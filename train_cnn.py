#!/usr/bin/env python3
"""
Script de Entrenamiento CNN 2D
================================
Pipeline completo: Reutiliza espectrogramas ya calculados,
entrena CNN 2D con SpecAugment, evalúa con MC Dropout y Grad-CAM.

Uso:
    python train_cnn.py --hc_dir data/vowels_healthy --pd_dir data/vowels_pk
"""

from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Importar módulos propios - REUTILIZAMOS PIPELINE EXISTENTE
from modules.dataset import build_full_pipeline
from modules.cnn_utils import (
    split_by_speaker,
    create_dataloaders_from_existing,
    compute_class_weights_from_dataset,
)
from modules.cnn_model import CNN2D, print_model_summary
from modules.cnn_training import (
    train_model,
    detailed_evaluation,
    print_evaluation_report,
    save_training_results,
)
from modules.cnn_inference import (
    mc_dropout_inference,
    aggregate_by_file,
    aggregate_by_patient,
    analyze_uncertainty,
    print_inference_report,
)
from modules.cnn_visualization import generate_visual_report


# ============================================================
# CONFIGURACIÓN
# ============================================================


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de CNN 2D para detección de Parkinson"
    )

    # Datos
    parser.add_argument(
        "--hc_dir",
        type=str,
        default="data/vowels_healthy",
        help="Directorio con datos HC",
    )
    parser.add_argument(
        "--pd_dir", type=str, default="data/vowels_pk", help="Directorio con datos PD"
    )

    # Split
    parser.add_argument(
        "--train_ratio", type=float, default=0.6, help="Proporción para training"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Proporción para validation"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.25, help="Proporción para test"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla para reproducibilidad"
    )

    # Modelo
    parser.add_argument(
        "--dropout_conv",
        type=float,
        default=0.3,
        help="Dropout en capas convolucionales",
    )
    parser.add_argument(
        "--dropout_fc", type=float, default=0.5, help="Dropout en capas fully connected"
    )

    # SpecAugment
    parser.add_argument(
        "--freq_mask", type=int, default=8, help="Parámetro de frequency masking"
    )
    parser.add_argument(
        "--time_mask", type=int, default=6, help="Parámetro de time masking"
    )
    parser.add_argument(
        "--spec_augment_prob",
        type=float,
        default=0.5,
        help="Probabilidad de aplicar cada máscara",
    )

    # Entrenamiento
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Número máximo de épocas"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Paciencia para early stopping"
    )
    parser.add_argument(
        "--use_class_weights", action="store_true", help="Usar class weights en loss"
    )

    # MC Dropout
    parser.add_argument(
        "--mc_samples", type=int, default=30, help="Número de muestras para MC Dropout"
    )

    # Guardado
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cnn_training",
        help="Directorio de salida",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    """Pipeline principal."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("PIPELINE DE ENTRENAMIENTO CNN 2D PARA DETECCIÓN DE PARKINSON")
    print("=" * 70)
    print(f"\n🔧 Configuración:")
    print(f"  - HC dir: {args.hc_dir}")
    print(f"  - PD dir: {args.pd_dir}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Épocas máximas: {args.epochs}")
    print(f"  - MC samples: {args.mc_samples}")
    print(f"  - Device: {args.device}")
    print(f"  - Output: {args.output_dir}")

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n💾 Configuración guardada: {config_path}")

    device = torch.device(args.device)

    # ================================================================
    # 1. CARGAR DATOS CON PIPELINE EXISTENTE (Reutiliza preprocesamiento)
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 1: CARGA DE DATOS (Reutilizando pipeline existente)")
    print("=" * 70)

    # Cargar archivos
    hc_files = list(Path(args.hc_dir).glob("*.egg"))
    pd_files = list(Path(args.pd_dir).glob("*.egg"))

    print("\n📁 Archivos encontrados:")
    print(f"  - HC: {len(hc_files)} archivos")
    print(f"  - PD: {len(pd_files)} archivos")

    # Procesar HC
    print("\n🟢 Procesando HC...")
    hc_result = build_full_pipeline(hc_files, max_files=None)
    hc_dataset = hc_result["torch_ds"]

    # CORREGIR ETIQUETAS: Todos HC deben ser 0
    print("   Corrigiendo etiquetas HC → 0...")
    for i in range(len(hc_dataset.y_task)):
        hc_dataset.y_task[i] = 0

    # Procesar PD
    print("\n🔴 Procesando PD...")
    pd_result = build_full_pipeline(pd_files, max_files=None)
    pd_dataset = pd_result["torch_ds"]

    # CORREGIR ETIQUETAS: Todos PD deben ser 1
    print("   Corrigiendo etiquetas PD → 1...")
    for i in range(len(pd_dataset.y_task)):
        pd_dataset.y_task[i] = 1

    # Combinar datasets
    from torch.utils.data import ConcatDataset

    combined_dataset = ConcatDataset([hc_dataset, pd_dataset])

    print(f"\n✅ Dataset combinado: {len(combined_dataset)} segmentos")

    # Verificar distribución de etiquetas
    hc_count = len(hc_dataset)
    pd_count = len(pd_dataset)
    print(f"   📊 Distribución: HC={hc_count} (0), PD={pd_count} (1)")

    # ================================================================
    # 2. SPLIT SPEAKER-INDEPENDENT
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 2: SPLIT SPEAKER-INDEPENDENT")
    print("=" * 70)

    # Combinar metadatos
    all_metas = hc_result["metadata"] + pd_result["metadata"]

    split_indices = split_by_speaker(
        all_metas,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ================================================================
    # 3. CREAR DATALOADERS CON SPECAUGMENT
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 3: CREACIÓN DE DATALOADERS (con SpecAugment)")
    print("=" * 70)

    spec_augment_params = {
        "freq_mask_param": args.freq_mask,
        "time_mask_param": args.time_mask,
        "prob": args.spec_augment_prob,
    }

    loaders = create_dataloaders_from_existing(
        base_dataset=combined_dataset,
        split_indices=split_indices,
        batch_size=args.batch_size,
        spec_augment_params=spec_augment_params,
        num_workers=0,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # ================================================================
    # 4. CREAR MODELO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 4: CREACIÓN DEL MODELO")
    print("=" * 70)

    model = CNN2D(n_classes=2, p_drop_conv=args.dropout_conv, p_drop_fc=args.dropout_fc)
    model = model.to(device)

    print_model_summary(model)

    # ================================================================
    # 5. CONFIGURAR ENTRENAMIENTO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 5: CONFIGURACIÓN DE ENTRENAMIENTO")
    print("=" * 70)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function (con class weights si se especifica)
    if args.use_class_weights:
        class_weights = compute_class_weights_from_dataset(
            combined_dataset, indices=split_indices["train"]
        )
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("\n✅ Configuración:")
    print(f"  - Optimizer: Adam (lr={args.lr})")
    loss_msg = "CrossEntropyLoss"
    if args.use_class_weights:
        loss_msg += " (con class weights)"
    print(f"  - Loss: {loss_msg}")
    print(f"  - Early stopping: {args.patience} épocas")

    # ================================================================
    # 6. ENTRENAR MODELO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 6: ENTRENAMIENTO")
    print("=" * 70)

    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_dir=output_dir,
        verbose=True,
    )

    model = training_results["model"]
    history = training_results["history"]

    # Guardar historial
    save_training_results(training_results, output_dir, prefix="training")

    # ================================================================
    # 7. EVALUACIÓN BÁSICA (sin MC Dropout)
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 7: EVALUACIÓN BÁSICA (Test Set)")
    print("=" * 70)

    basic_eval = detailed_evaluation(
        model=model, loader=test_loader, device=device, class_names=["HC", "PD"]
    )

    print_evaluation_report(basic_eval, class_names=["HC", "PD"])

    # ================================================================
    # 8. MC DROPOUT INFERENCE
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 8: INFERENCIA CON MC DROPOUT")
    print("=" * 70)

    mc_results = mc_dropout_inference(
        model=model,
        loader=test_loader,
        device=device,
        n_samples=args.mc_samples,
        verbose=True,
    )

    # Guardar resultados MC
    mc_results_path = output_dir / "mc_dropout_results.npz"
    import numpy as np

    np.savez(
        mc_results_path,
        predictions=mc_results["predictions"],
        probabilities_mean=mc_results["probabilities_mean"],
        probabilities_std=mc_results["probabilities_std"],
        entropy=mc_results["entropy"],
        variance=mc_results["variance"],
        labels=mc_results["labels"],
        file_ids=mc_results["file_ids"],
    )
    print(f"💾 Resultados MC Dropout guardados: {mc_results_path}")

    # ================================================================
    # 9. AGREGACIÓN POR ARCHIVO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 9: AGREGACIÓN POR ARCHIVO")
    print("=" * 70)

    file_results = aggregate_by_file(mc_results, aggregation_method="mean")

    # Guardar
    file_results_path = output_dir / "file_level_results.npz"
    np.savez(
        file_results_path,
        file_predictions=file_results["file_predictions"],
        file_probabilities=file_results["file_probabilities"],
        file_uncertainty=file_results["file_uncertainty"],
        file_labels=file_results["file_labels"],
        file_ids=file_results["file_ids"],
    )
    print(f"💾 Resultados por archivo guardados: {file_results_path}")

    # ================================================================
    # 10. AGREGACIÓN POR PACIENTE
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 10: AGREGACIÓN POR PACIENTE")
    print("=" * 70)

    patient_results = aggregate_by_patient(mc_results, aggregation_method="mean")

    # Guardar
    patient_results_path = output_dir / "patient_level_results.npz"
    np.savez(
        patient_results_path,
        patient_predictions=patient_results["patient_predictions"],
        patient_probabilities=patient_results["patient_probabilities"],
        patient_uncertainty=patient_results["patient_uncertainty"],
        patient_labels=patient_results["patient_labels"],
    )
    print(f"💾 Resultados por paciente guardados: {patient_results_path}")

    # Imprimir reporte completo
    print_inference_report(mc_results, file_results, patient_results)

    # ================================================================
    # 11. ANÁLISIS DE INCERTIDUMBRE
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 11: ANÁLISIS DE INCERTIDUMBRE")
    print("=" * 70)

    uncertainty_analysis = analyze_uncertainty(
        mc_results, aggregated_results=file_results
    )

    # Guardar análisis
    uncertainty_path = output_dir / "uncertainty_analysis.json"
    with open(uncertainty_path, "w") as f:
        json.dump(uncertainty_analysis, f, indent=2)
    print(f"💾 Análisis de incertidumbre guardado: {uncertainty_path}")

    # Imprimir resumen
    print("\n📊 RESUMEN DE INCERTIDUMBRE:")
    print(f"  Entropía promedio: {uncertainty_analysis['entropy']['mean']:.4f}")
    print(
        f"  Entropía (correctos): {uncertainty_analysis['uncertainty_vs_correctness']['correct_mean_entropy']:.4f}"
    )
    print(
        f"  Entropía (incorrectos): {uncertainty_analysis['uncertainty_vs_correctness']['incorrect_mean_entropy']:.4f}"
    )

    # ================================================================
    # 12. GENERAR VISUALIZACIONES
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 12: GENERACIÓN DE VISUALIZACIONES")
    print("=" * 70)

    viz_dir = output_dir / "visualizations"

    generate_visual_report(
        model=model,
        loader=test_loader,
        mc_results=mc_results,
        file_results=file_results,
        history=history,
        save_dir=viz_dir,
    )

    # ================================================================
    # RESUMEN FINAL
    # ================================================================

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)

    from sklearn.metrics import accuracy_score, f1_score

    seg_acc = accuracy_score(mc_results["labels"], mc_results["predictions"])
    seg_f1 = f1_score(mc_results["labels"], mc_results["predictions"])

    file_acc = accuracy_score(
        file_results["file_labels"], file_results["file_predictions"]
    )
    file_f1 = f1_score(file_results["file_labels"], file_results["file_predictions"])

    patient_acc = accuracy_score(
        patient_results["patient_labels"], patient_results["patient_predictions"]
    )
    patient_f1 = f1_score(
        patient_results["patient_labels"], patient_results["patient_predictions"]
    )

    print("\n📊 RESULTADOS FINALES:")
    print(f"\n  Nivel Segmento:")
    print(f"    - Accuracy: {seg_acc:.4f}")
    print(f"    - F1-Score: {seg_f1:.4f}")

    print(f"\n  Nivel Archivo:")
    print(f"    - Accuracy: {file_acc:.4f}")
    print(f"    - F1-Score: {file_f1:.4f}")

    print(f"\n  Nivel Paciente:")
    print(f"    - Accuracy: {patient_acc:.4f}")
    print(f"    - F1-Score: {patient_f1:.4f}")

    print(f"\n📁 Todos los resultados guardados en: {output_dir}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
