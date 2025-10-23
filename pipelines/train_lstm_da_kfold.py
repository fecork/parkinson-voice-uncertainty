#!/usr/bin/env python3
"""
Script de Entrenamiento Time-CNN-BiLSTM-DA con 10-Fold CV
==========================================================
Implementación completa según Ibarra et al. (2023):
- 10-fold CV estratificada independiente por hablante
- SGD con LR inicial 0.1 y scheduler StepLR
- Cross-entropy ponderada automática
- Lambda warm-up para GRL (0→1 en 5 épocas)
- Secuencias de n espectrogramas con zero-padding + masking

Uso:
    python train_lstm_da_kfold.py --n_frames 7 --lstm_units 64
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

# Agregar directorio raíz al path para importar modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos propios
from modules.core.sequence_dataset import (
    create_sequence_dataset_from_cache,
)
from modules.models.lstm_da.model import TimeCNNBiLSTM_DA
from modules.models.lstm_da.training import train_model_da_kfold


# ============================================================
# CONFIGURACIÓN
# ============================================================


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento Time-CNN-BiLSTM-DA con 10-Fold CV (Ibarra 2023)"
    )

    # Datos
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directorio base de cache",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=7,
        help="Número de frames por secuencia (paper: 3, 5, 7, 9)",
    )

    # Modelo
    parser.add_argument(
        "--lstm_units",
        type=int,
        default=64,
        help="Unidades LSTM por dirección (paper: 16, 32, 64)",
    )
    parser.add_argument(
        "--dropout_conv",
        type=float,
        default=0.3,
        help="Dropout en capas convolucionales",
    )
    parser.add_argument(
        "--dropout_fc",
        type=float,
        default=0.5,
        help="Dropout en capas FC",
    )

    # Entrenamiento (según Ibarra 2023)
    parser.add_argument(
        "--n_folds", type=int, default=10, help="Número de folds (paper: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch",
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
        "--lambda_warmup",
        type=int,
        default=5,
        help="Épocas de warm-up para lambda GRL (paper: 5)",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Paciencia para early stopping"
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    # Guardado
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/lstm_da_kfold",
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
    """Pipeline principal de 10-fold CV."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO TIME-CNN-BILSTM-DA CON 10-FOLD CV (IBARRA 2023)")
    print("=" * 70)
    print("\n🔧 Configuración:")
    print(f"  - Cache dir: {args.cache_dir}")
    print(f"  - N frames: {args.n_frames}")
    print(f"  - LSTM units: {args.lstm_units}")
    print(f"  - N folds: {args.n_folds}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - LR inicial: {args.lr} (SGD)")
    print(f"  - Lambda warm-up: {args.lambda_warmup} épocas")
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
    # 1. CARGAR DATOS DE CACHE
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 1: CARGA DE DATOS DESDE CACHE")
    print("=" * 70)

    # Paths de cache
    cache_healthy = Path(args.cache_dir) / "healthy" / f"lstm_sequences_n{args.n_frames}.pkl"
    cache_parkinson = Path(args.cache_dir) / "parkinson" / f"lstm_sequences_n{args.n_frames}.pkl"

    print(f"\n📁 Cargando secuencias (n={args.n_frames})...")
    print(f"  - HC: {cache_healthy}")
    print(f"  - PD: {cache_parkinson}")

    # Verificar que existan los caches
    if not cache_healthy.exists():
        print(f"\n❌ ERROR: No existe cache: {cache_healthy}")
        print("   Ejecuta data_preprocessing.ipynb primero")
        sys.exit(1)

    if not cache_parkinson.exists():
        print(f"\n❌ ERROR: No existe cache: {cache_parkinson}")
        print("   Ejecuta data_preprocessing.ipynb primero")
        sys.exit(1)

    # Cargar datasets
    print("\n🟢 Cargando secuencias Healthy...")
    hc_dataset = create_sequence_dataset_from_cache(
        cache_path=str(cache_healthy),
        label_value=0,  # HC
    )

    print("\n🔴 Cargando secuencias Parkinson...")
    pd_dataset = create_sequence_dataset_from_cache(
        cache_path=str(cache_parkinson),
        label_value=1,  # PD
    )

    # Combinar datasets
    combined_dataset = ConcatDataset([hc_dataset, pd_dataset])

    print(f"\n✅ Dataset combinado: {len(combined_dataset)} secuencias")
    print(f"   HC: {len(hc_dataset)} secuencias")
    print(f"   PD: {len(pd_dataset)} secuencias")

    # ================================================================
    # 2. PREPARAR METADATA PARA K-FOLD
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 2: PREPARACIÓN DE METADATA")
    print("=" * 70)

    # Combinar metadatos y agregar labels
    metadata_list = []

    # Metadata HC (label=0)
    for i in range(len(hc_dataset)):
        sample = hc_dataset[i]
        meta = sample["meta"]
        metadata_list.append({
            "subject_id": meta.subject_id,
            "filename": meta.filename,
            "label": 0,
            "domain_label": sample["y_domain"],
        })

    # Metadata PD (label=1)
    for i in range(len(pd_dataset)):
        sample = pd_dataset[i]
        meta = sample["meta"]
        metadata_list.append({
            "subject_id": meta.subject_id,
            "filename": meta.filename,
            "label": 1,
            "domain_label": sample["y_domain"],
        })

    print(f"\n✅ Metadata preparada: {len(metadata_list)} secuencias")

    # Verificar balance
    hc_count = sum(1 for m in metadata_list if m["label"] == 0)
    pd_count = sum(1 for m in metadata_list if m["label"] == 1)
    hc_pct = hc_count / len(metadata_list) * 100
    pd_pct = pd_count / len(metadata_list) * 100
    print(f"   HC: {hc_count} ({hc_pct:.1f}%)")
    print(f"   PD: {pd_count} ({pd_pct:.1f}%)")

    # ================================================================
    # 3. DETERMINAR NÚMERO DE DOMINIOS
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 3: ANÁLISIS DE DOMINIOS")
    print("=" * 70)

    # Según paper Ibarra et al. (2023): 4 corpus fijos
    n_domains = 4
    print(f"\n📊 Dominios configurados: {n_domains} (según paper)")
    print("   • GITA, Neurovoz, German, Czech")

    # Contar dominios únicos en datos
    unique_domains = set(m["domain_label"] for m in metadata_list)
    print(f"\n📌 Dominios detectados: {len(unique_domains)}")

    # ================================================================
    # 4. CONFIGURACIÓN DEL MODELO
    # ================================================================

    print("\n" + "=" * 70)
    print("PASO 4: CONFIGURACIÓN DEL MODELO")
    print("=" * 70)

    model_params = {
        "n_frames": args.n_frames,
        "lstm_units": args.lstm_units,
        "n_domains": n_domains,
        "p_drop_conv": args.dropout_conv,
        "p_drop_fc": args.dropout_fc,
    }

    print("\n⚙️  Parámetros del modelo:")
    print(f"  - N frames: {args.n_frames}")
    print(f"  - LSTM units: {args.lstm_units} (bidirectional)")
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
        model_class=TimeCNNBiLSTM_DA,
        model_params=model_params,
        dataset=combined_dataset,
        metadata_list=metadata_list,
        device=device,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        lambda_warmup_epochs=args.lambda_warmup,
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
    print(
        f"   Val Acc PD:  {kfold_results['mean_val_acc_pd']:.4f} "
        f"± {kfold_results['std_val_acc_pd']:.4f}"
    )
    print(
        f"   Val F1 PD:   {kfold_results['mean_val_f1_pd']:.4f} "
        f"± {kfold_results['std_val_f1_pd']:.4f}"
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
    print(f"✅ SGD con LR inicial {args.lr} + momentum 0.9")
    print("✅ LR scheduler (StepLR decay cada 30 épocas)")
    print("✅ Cross-entropy ponderada automática (PD + dominio)")
    print(f"✅ Lambda GRL warm-up 0→1 en {args.lambda_warmup} épocas")
    print("✅ Arquitectura Time-CNN-BiLSTM con DA")
    print(f"✅ Secuencias de n={args.n_frames} frames con masking")
    print("✅ Métricas reportadas: mean ± std sobre folds")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
