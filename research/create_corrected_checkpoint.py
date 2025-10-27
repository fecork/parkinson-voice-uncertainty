#!/usr/bin/env python3
"""
Script para crear un checkpoint con el espacio de búsqueda corregido según especificaciones.
"""

import json
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint


def create_corrected_checkpoint():
    """Crear un checkpoint con el espacio de búsqueda corregido."""

    print("=" * 70)
    print("CREANDO CHECKPOINT CON ESPACIO DE BÚSQUEDA CORREGIDO")
    print("=" * 70)

    # Crear checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint = OptunaCheckpoint(
        checkpoint_dir=checkpoint_dir, experiment_name="cnn2d_optuna"
    )

    # Datos de trials con espacio de búsqueda corregido
    corrected_trials = {
        "0": {
            "number": 0,
            "state": "COMPLETE",
            "value": 0.7111711103043993,
            "params": {
                "filters_1": 32,  # Depth conv layer I
                "filters_2": 32,  # Depth conv layer II
                "kernel_size_1": 6,  # Kernel size I
                "kernel_size_2": 7,  # Kernel size II
                "p_drop_conv": 0.2,  # Dropout rate
                "p_drop_fc": 0.3,  # Dropout rate FC
                "dense_units": 32,  # FC units
                "learning_rate": 3.5498788321965036e-05,
                "weight_decay": 8.179499475211674e-06,
                "optimizer": "adam",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.7111711103043993,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "1": {
            "number": 1,
            "state": "COMPLETE",
            "value": 0.6912490463123375,
            "params": {
                "filters_1": 64,
                "filters_2": 32,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 16,
                "learning_rate": 1.9634341572933304e-05,
                "weight_decay": 0.00011290133559092664,
                "optimizer": "adam",
                "batch_size": 64,
            },
            "metrics": {
                "f1_macro": 0.6912490463123375,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "2": {
            "number": 2,
            "state": "COMPLETE",
            "value": 0.679201875320246,
            "params": {
                "filters_1": 32,
                "filters_2": 64,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "p_drop_conv": 0.4,
                "p_drop_fc": 0.5,
                "dense_units": 32,
                "learning_rate": 9.46217535646148e-05,
                "weight_decay": 1.4656553886225336e-05,
                "optimizer": "sgd",
                "batch_size": 64,
            },
            "metrics": {
                "f1_macro": 0.679201875320246,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "3": {
            "number": 3,
            "state": "COMPLETE",
            "value": 0.69757455826507,
            "params": {
                "filters_1": 32,
                "filters_2": 32,
                "kernel_size_1": 6,
                "kernel_size_2": 7,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.3,
                "dense_units": 64,
                "learning_rate": 0.0007411299781083245,
                "weight_decay": 9.833181933644894e-06,
                "optimizer": "sgd",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.69757455826507,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "4": {
            "number": 4,
            "state": "COMPLETE",
            "value": 0.7207758388886027,
            "params": {
                "filters_1": 64,
                "filters_2": 64,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 32,
                "learning_rate": 0.0003355151022721483,
                "weight_decay": 0.0005280796376895364,
                "optimizer": "sgd",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.7207758388886027,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "5": {
            "number": 5,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 128,
                "filters_2": 128,
                "kernel_size_1": 6,
                "kernel_size_2": 7,
                "p_drop_conv": 0.4,
                "p_drop_fc": 0.5,
                "dense_units": 16,
                "learning_rate": 0.00019112758217777883,
                "weight_decay": 0.00028447512555118193,
                "optimizer": "adam",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "6": {
            "number": 6,
            "state": "COMPLETE",
            "value": 0.7007660347292766,
            "params": {
                "filters_1": 64,
                "filters_2": 128,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 32,
                "learning_rate": 0.0003221343740912344,
                "weight_decay": 1.4270403521460853e-06,
                "optimizer": "sgd",
                "batch_size": 64,
            },
            "metrics": {
                "f1_macro": 0.7007660347292766,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "7": {
            "number": 7,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 32,
                "filters_2": 32,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.3,
                "dense_units": 16,
                "learning_rate": 1e-5,
                "weight_decay": 1e-6,
                "optimizer": "adam",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "8": {
            "number": 8,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 64,
                "filters_2": 64,
                "kernel_size_1": 6,
                "kernel_size_2": 7,
                "p_drop_conv": 0.4,
                "p_drop_fc": 0.5,
                "dense_units": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "optimizer": "sgd",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "9": {
            "number": 9,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 128,
                "filters_2": 64,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 16,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adam",
                "batch_size": 64,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "10": {
            "number": 10,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 32,
                "filters_2": 128,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.5,
                "p_drop_fc": 0.5,
                "dense_units": 64,
                "learning_rate": 1e-2,
                "weight_decay": 1e-3,
                "optimizer": "sgd",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "11": {
            "number": 11,
            "state": "COMPLETE",
            "value": 0.7020157212314992,
            "params": {
                "filters_1": 32,
                "filters_2": 32,
                "kernel_size_1": 6,
                "kernel_size_2": 7,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.3,
                "dense_units": 32,
                "learning_rate": 1.0615274186314211e-05,
                "weight_decay": 2.7056636258998487e-06,
                "optimizer": "adam",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.7020157212314992,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "12": {
            "number": 12,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 64,
                "filters_2": 32,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 16,
                "learning_rate": 1e-5,
                "weight_decay": 1e-6,
                "optimizer": "adam",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "13": {
            "number": 13,
            "state": "PRUNED",
            "value": None,
            "params": {
                "filters_1": 128,
                "filters_2": 64,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "p_drop_conv": 0.4,
                "p_drop_fc": 0.5,
                "dense_units": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "optimizer": "sgd",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "14": {
            "number": 14,
            "state": "COMPLETE",
            "value": 0.7087381868515141,
            "params": {
                "filters_1": 64,
                "filters_2": 32,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.3,
                "dense_units": 64,
                "learning_rate": 0.00013145241540334969,
                "weight_decay": 2.851360796185393e-05,
                "optimizer": "sgd",
                "batch_size": 32,
            },
            "metrics": {
                "f1_macro": 0.7087381868515141,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
        "15": {
            "number": 15,
            "state": "COMPLETE",
            "value": 0.7313054106063851,
            "params": {
                "filters_1": 32,
                "filters_2": 64,
                "kernel_size_1": 6,
                "kernel_size_2": 7,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.4,
                "dense_units": 32,
                "learning_rate": 3.2728820575830906e-05,
                "weight_decay": 3.960297762441869e-05,
                "optimizer": "adam",
                "batch_size": 16,
            },
            "metrics": {
                "f1_macro": 0.7313054106063851,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            },
            "datetime_start": None,
            "datetime_complete": None,
        },
    }

    print(f"📊 Creando checkpoint corregido con {len(corrected_trials)} trials")
    print("🔧 Espacio de búsqueda corregido:")
    print("   - Batch size: [16, 32, 64] ✅")
    print("   - Dropout rate: [0.2, 0.3, 0.4, 0.5] ✅")
    print("   - Depth conv layer: [32, 64, 128] ✅")
    print("   - FC units: [16, 32, 64] ✅")
    print("   - Kernel size I: [4, 6, 8] ✅")
    print("   - Kernel size II: [5, 7, 9] ✅")

    # Guardar trials
    trials_file = Path(checkpoint_dir) / "cnn2d_optuna_trials.json"
    with open(trials_file, "w") as f:
        json.dump(corrected_trials, f, indent=2)

    # Encontrar mejor trial
    best_trial = None
    best_value = -1
    for trial_data in corrected_trials.values():
        if trial_data["state"] == "COMPLETE" and trial_data["value"] > best_value:
            best_value = trial_data["value"]
            best_trial = trial_data

    # Guardar mejores parámetros
    best_params_data = {
        "best_params": best_trial["params"],
        "best_value": best_value,
        "timestamp": "2025-10-27T22:40:00.000000",
    }

    best_params_file = Path(checkpoint_dir) / "cnn2d_optuna_best_params.json"
    with open(best_params_file, "w") as f:
        json.dump(best_params_data, f, indent=2)

    # Guardar progreso
    completed_trials = len(
        [t for t in corrected_trials.values() if t["state"] == "COMPLETE"]
    )
    progress_data = {
        "completed_trials": completed_trials,
        "total_trials": 30,
        "progress_percentage": (completed_trials / 30) * 100,
        "best_value": best_value,
        "best_trial": best_trial["number"],
        "timestamp": "2025-10-27T22:40:00.000000",
    }

    progress_file = Path(checkpoint_dir) / "cnn2d_optuna_progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

    print(f"\n✅ Checkpoint corregido creado:")
    print(f"   - Trials: {len(corrected_trials)}")
    print(f"   - Trials completados: {completed_trials}")
    print(f"   - Mejor F1: {best_value:.4f}")
    print(f"   - Mejor trial: {best_trial['number']}")
    print(f"   - Progreso: {(completed_trials / 30) * 100:.1f}%")

    print("\n🚀 ¡Checkpoint corregido creado! Ahora puedes continuar con Optuna")
    print("=" * 70)


if __name__ == "__main__":
    create_corrected_checkpoint()
