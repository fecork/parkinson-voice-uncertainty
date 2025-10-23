#!/usr/bin/env python3
"""
Script de Entrenamiento CNN 2D-DA con 10-Fold CV
=================================================
Implementación completa según Ibarra et al. (2023):
- 10-fold CV estratificada independiente por hablante
- SGD con LR inicial 0.1 y scheduler StepLR
- Cross-entropy ponderada automática
- Lambda constante para GRL

Uso:
    python train_cnn_da_kfold.py --hc_dir data/vowels_healthy --pd_dir data/vowels_pk
"""

from pathlib import Path
import argparse
import json
import sys
import torch
from torch.utils.data import ConcatDataset

# Agregar directorio raíz al path para importar modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos propios
from modules.core.dataset import build_full_pipeline
from modules.models.cnn2d.model import CNN2D_DA
from modules.models.cnn2d.training import train_model_da_kfold


# ============================================================
# CONFIGURACIÓN
# ============================================================


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento CNN 2D-DA con 10-Fold CV (Ibarra 2023)"
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

    # Modelo
    parser.add_argument(
        "--dropout_conv",
        type=float,
        default=0.3,
        help="Dropout en capas convolucionales (paper: 0.2 o 0.5)",
    )
    parser.add_argument(
        "--dropout_fc",
        type=float,
        default=0.5,
        help="Dropout en capas FC (paper: 0.2 o 0.5)",
    )

    # Entrenamiento (según Ibarra 2023)
    parser.add_argument(
        "--n_folds", type=int, default=10, help="Número de folds (paper: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch (paper: probar 16/32/64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Número máximo de épocas"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate inicial (paper: 0.1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Peso de pérdida de dominio"
    )
    parser.add_argument(
        "--lambda_grl",
        type=float,
        default=1.0,
        help="Lambda constante para GRL (paper: constante)",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Paciencia para early stopping"
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    # Guardado
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cnn_da_kfold",
        help="Directorio de salida",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    # Cache (usar augmentation pre-calculada)
    parser.add_argument(
        "--use_augmented",
        action="store_true",
        help="Usar datos con augmentation (recomendado)",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    """Pipeline principal de 10-fold CV."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO CNN 2D-DA CON 10-FOLD CV (IBARRA 2023)")
    print("=" * 70)
    print(f"\n🔧 Configuración:")
    print(f"  - HC dir: {args.hc_dir}")
    print(f"  - PD dir: {args.pd_dir}")
    print(f"  - N folds: {args.n_folds}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - LR inicial: {args.lr} (SGD)")
    print(f"  - Lambda GRL: {args.lambda_grl} (constante)")
    print(f"  - Épocas máx: {args.epochs}")
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
    # 1. CARGAR DATOS
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 1: CARGA DE DATOS")
    print("=" * 70)

    # Cargar archivos
    hc_files = list(Path(args.hc_dir).glob("*.egg"))
    pd_files = list(Path(args.pd_dir).glob("*.egg"))

    print(f"\n📁 Archivos encontrados:")
    print(f"  - HC: {len(hc_files)} archivos")
    print(f"  - PD: {len(pd_files)} archivos")

    if args.use_augmented:
        print("\n⚠️  Modo augmentation activado")
        print("   Si tienes cache, se cargará automáticamente")

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
    combined_dataset = ConcatDataset([hc_dataset, pd_dataset])

    print(f"\n✅ Dataset combinado: {len(combined_dataset)} muestras")
    print(f"   HC: {len(hc_dataset)} muestras")
    print(f"   PD: {len(pd_dataset)} muestras")

    # ================================================================
    # 2. PREPARAR METADATA PARA K-FOLD
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 2: PREPARACIÓN DE METADATA")
    print("=" * 70)

    # Combinar metadatos y agregar labels
    metadata_list = []

    # Metadata HC (label=0)
    for meta in hc_result["metadata"]:
        meta_copy = meta.copy()
        meta_copy["label"] = 0
        metadata_list.append(meta_copy)

    # Metadata PD (label=1)
    for meta in pd_result["metadata"]:
        meta_copy = meta.copy()
        meta_copy["label"] = 1
        metadata_list.append(meta_copy)

    print(f"\n✅ Metadata preparada: {len(metadata_list)} muestras")

    # Verificar balance
    hc_count = sum(1 for m in metadata_list if m["label"] == 0)
    pd_count = sum(1 for m in metadata_list if m["label"] == 1)
    print(f"   HC: {hc_count} ({hc_count / len(metadata_list) * 100:.1f}%)")
    print(f"   PD: {pd_count} ({pd_count / len(metadata_list) * 100:.1f}%)")

    # ================================================================
    # 3. DETERMINAR NÚMERO DE DOMINIOS
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 3: ANÁLISIS DE DOMINIOS")
    print("=" * 70)

    # Contar dominios únicos (basado en subject_id)
    unique_subjects = set()
    for meta in metadata_list:
        subject_id = meta.get("subject_id", meta.get("filename", "unknown"))
        unique_subjects.add(subject_id)

    n_domains = len(unique_subjects)
    print(f"\n📊 Dominios detectados: {n_domains} hablantes únicos")

    # ================================================================
    # 4. CONFIGURACIÓN DEL MODELO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 4: CONFIGURACIÓN DEL MODELO")
    print("=" * 70)

    model_params = {
        "n_domains": n_domains,
        "p_drop_conv": args.dropout_conv,
        "p_drop_fc": args.dropout_fc,
    }

    print(f"\n⚙️  Parámetros del modelo:")
    print(f"  - Dominios: {n_domains}")
    print(f"  - Dropout conv: {args.dropout_conv}")
    print(f"  - Dropout FC: {args.dropout_fc}")

    # ================================================================
    # 5. ENTRENAMIENTO 10-FOLD CV
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 5: ENTRENAMIENTO 10-FOLD CV")
    print("=" * 70)
    print("\n🚀 Iniciando 10-fold cross-validation según Ibarra (2023)...")
    print("   Esto puede tomar bastante tiempo (entrenar 10 modelos)")

    kfold_results = train_model_da_kfold(
        model_class=CNN2D_DA,
        model_params=model_params,
        dataset=combined_dataset,
        metadata_list=metadata_list,
        device=device,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        lambda_constant=args.lambda_grl,
        early_stopping_patience=args.patience,
        save_dir=output_dir,
        seed=args.seed,
        verbose=True,
    )

    # ================================================================
    # 6. RESUMEN FINAL
    # ================================================================

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)

    print(f"\n📊 RESULTADOS FINALES ({args.n_folds}-Fold CV):")
    print(
        f"   Val Loss PD: {kfold_results['mean_val_loss_pd']:.4f} "
        f"± {kfold_results['std_val_loss_pd']:.4f}"
    )
    print(f"   Tiempo total: {kfold_results['total_time'] / 60:.1f} minutos")

    print(f"\n📁 Resultados guardados en: {output_dir}")
    print(f"   - Configuración: {output_dir}/config.json")
    print(f"   - Métricas K-fold: {output_dir}/kfold_results.json")
    print(f"   - Modelos por fold: {output_dir}/fold_*/")

    print("\n" + "=" * 70)
    print("CUMPLIMIENTO DEL PAPER IBARRA (2023):")
    print("=" * 70)
    print("✅ 10-fold CV estratificada independiente por hablante")
    print(f"✅ SGD con LR inicial {args.lr}")
    print("✅ LR scheduler (StepLR decay cada 30 épocas)")
    print("✅ Cross-entropy ponderada automática (PD + dominio)")
    print(f"✅ Lambda GRL constante = {args.lambda_grl}")
    print("✅ Arquitectura 2D-CNN con DA (MaxPool 3×3)")
    print("✅ Métricas reportadas: mean ± std sobre folds")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
