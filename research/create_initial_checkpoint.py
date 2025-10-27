# ============================================================
# CREAR CHECKPOINT INICIAL DESDE LOGS
# ============================================================

import json
import pandas as pd
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint

print("=" * 70)
print("CREANDO CHECKPOINT INICIAL DESDE LOGS")
print("=" * 70)

# Datos de los logs que proporcionaste
trials_data = {
    0: {
        "number": 0,
        "state": "COMPLETE",
        "value": 0.7111711103043993,
        "params": {
            "filters_1": 32,
            "filters_2": 32,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.20617534828874073,
            "p_drop_fc": 0.5909729556485983,
            "dense_units": 32,
            "learning_rate": 3.5498788321965036e-05,
            "weight_decay": 8.179499475211674e-06,
            "optimizer": "adam",
            "batch_size": 32,
        },
        "metrics": {
            "f1_macro": 0.7111711103043993,
            "accuracy": 0.0,  # Placeholder
            "precision_macro": 0.0,  # Placeholder
            "recall_macro": 0.0,  # Placeholder
        },
    },
    1: {
        "number": 1,
        "state": "COMPLETE",
        "value": 0.6912490463123375,
        "params": {
            "filters_1": 64,
            "filters_2": 32,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.21951547789558387,
            "p_drop_fc": 0.5846656611759999,
            "dense_units": 32,
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
    },
    2: {
        "number": 2,
        "state": "COMPLETE",
        "value": 0.679201875320246,
        "params": {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.3793699936433256,
            "p_drop_fc": 0.5765622705069351,
            "dense_units": 64,
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
    },
    3: {
        "number": 3,
        "state": "COMPLETE",
        "value": 0.69757455826507,
        "params": {
            "filters_1": 32,
            "filters_2": 32,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.4313811040057837,
            "p_drop_fc": 0.3222133955202271,
            "dense_units": 128,
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
    },
    4: {
        "number": 4,
        "state": "COMPLETE",
        "value": 0.7207758388886027,
        "params": {
            "filters_1": 16,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.20762573802322856,
            "p_drop_fc": 0.3323674280979913,
            "dense_units": 64,
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
    },
    5: {
        "number": 5,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.44223204654921877,
            "p_drop_fc": 0.568827389977048,
            "dense_units": 32,
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
    },
    6: {
        "number": 6,
        "state": "COMPLETE",
        "value": 0.7007660347292766,
        "params": {
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 5,
            "kernel_size_2": 3,
            "p_drop_conv": 0.3491745517677156,
            "p_drop_fc": 0.3902634929450309,
            "dense_units": 128,
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
    },
    7: {
        "number": 7,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 32,
            "filters_2": 32,
            "kernel_size_1": 3,
            "kernel_size_2": 5,
            "p_drop_conv": 0.2,
            "p_drop_fc": 0.3,
            "dense_units": 32,
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
    },
    8: {
        "number": 8,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 16,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.4,
            "dense_units": 64,
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
    },
    9: {
        "number": 9,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 64,
            "filters_2": 32,
            "kernel_size_1": 5,
            "kernel_size_2": 3,
            "p_drop_conv": 0.4,
            "p_drop_fc": 0.5,
            "dense_units": 32,
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
    },
    10: {
        "number": 10,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 32,
            "filters_2": 128,
            "kernel_size_1": 3,
            "kernel_size_2": 5,
            "p_drop_conv": 0.5,
            "p_drop_fc": 0.6,
            "dense_units": 128,
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
    },
    11: {
        "number": 11,
        "state": "COMPLETE",
        "value": 0.7020157212314992,
        "params": {
            "filters_1": 32,
            "filters_2": 32,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.2720199749415961,
            "p_drop_fc": 0.4948392184001192,
            "dense_units": 64,
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
    },
    12: {
        "number": 12,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 16,
            "filters_2": 32,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.2,
            "p_drop_fc": 0.3,
            "dense_units": 32,
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
    },
    13: {
        "number": 13,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 64,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.4,
            "dense_units": 64,
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
    },
    14: {
        "number": 14,
        "state": "COMPLETE",
        "value": 0.7087381868515141,
        "params": {
            "filters_1": 16,
            "filters_2": 32,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.24344710966151495,
            "p_drop_fc": 0.3046281544859008,
            "dense_units": 128,
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
    },
    15: {
        "number": 15,
        "state": "COMPLETE",
        "value": 0.7313054106063851,
        "params": {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.3129300736522818,
            "p_drop_fc": 0.35785994258711895,
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
    },
    16: {
        "number": 16,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.4,
            "p_drop_fc": 0.5,
            "dense_units": 32,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "batch_size": 16,
        },
        "metrics": {
            "f1_macro": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    17: {
        "number": 17,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 16,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 3,
            "p_drop_conv": 0.5,
            "p_drop_fc": 0.6,
            "dense_units": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "sgd",
            "batch_size": 32,
        },
        "metrics": {
            "f1_macro": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    18: {
        "number": 18,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 32,
            "filters_2": 32,
            "kernel_size_1": 3,
            "kernel_size_2": 5,
            "p_drop_conv": 0.6,
            "p_drop_fc": 0.7,
            "dense_units": 128,
            "learning_rate": 1e-2,
            "weight_decay": 1e-3,
            "optimizer": "adam",
            "batch_size": 64,
        },
        "metrics": {
            "f1_macro": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    19: {
        "number": 19,
        "state": "COMPLETE",
        "value": 0.7087951917681903,
        "params": {
            "filters_1": 16,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 5,
            "p_drop_conv": 0.30406356887588326,
            "p_drop_fc": 0.37305466698435774,
            "dense_units": 128,
            "learning_rate": 0.00017132977191518947,
            "weight_decay": 6.720938064195148e-05,
            "optimizer": "sgd",
            "batch_size": 16,
        },
        "metrics": {
            "f1_macro": 0.7087951917681903,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    20: {
        "number": 20,
        "state": "COMPLETE",
        "value": 0.7239682360811073,
        "params": {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.2721530901348335,
            "p_drop_fc": 0.3049823678115836,
            "dense_units": 32,
            "learning_rate": 6.992934972302674e-05,
            "weight_decay": 0.00026466907147357986,
            "optimizer": "adam",
            "batch_size": 16,
        },
        "metrics": {
            "f1_macro": 0.7239682360811073,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    21: {
        "number": 21,
        "state": "PRUNED",
        "value": None,
        "params": {
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "p_drop_conv": 0.4,
            "p_drop_fc": 0.5,
            "dense_units": 64,
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
    },
    22: {
        "number": 22,
        "state": "COMPLETE",
        "value": 0.7210880942246316,
        "params": {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.2641387308620473,
            "p_drop_fc": 0.3371101986146679,
            "dense_units": 32,
            "learning_rate": 2.1625855120359407e-05,
            "weight_decay": 0.00023272766500317545,
            "optimizer": "adam",
            "batch_size": 16,
        },
        "metrics": {
            "f1_macro": 0.7210880942246316,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
    23: {
        "number": 23,
        "state": "COMPLETE",
        "value": 0.7230127197574583,
        "params": {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 5,
            "kernel_size_2": 5,
            "p_drop_conv": 0.2865692436199309,
            "p_drop_fc": 0.3560918228578222,
            "dense_units": 32,
            "learning_rate": 2.3964483219461982e-05,
            "weight_decay": 0.00017303877476957207,
            "optimizer": "adam",
            "batch_size": 16,
        },
        "metrics": {
            "f1_macro": 0.7230127197574583,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        },
    },
}

# Crear checkpoint
checkpoint_dir = "checkpoints"
checkpoint = OptunaCheckpoint(
    checkpoint_dir=checkpoint_dir, experiment_name="cnn2d_optuna"
)

print("💾 Guardando trials en checkpoint...")

# Guardar cada trial
for trial_num, trial_data in trials_data.items():
    # Crear trial mock para guardar
    class MockTrial:
        def __init__(self, data):
            self.number = data["number"]
            self.params = data["params"]
            self.value = data["value"]
            self.state = data["state"]
            self.datetime_start = None
            self.datetime_complete = None

    mock_trial = MockTrial(trial_data)
    checkpoint.save_trial(mock_trial, trial_data["metrics"])

# Guardar mejores parámetros (Trial 15 es el mejor)
best_params = trials_data[15]["params"]
best_value = trials_data[15]["value"]
checkpoint.save_best_params(best_params, best_value)

# Guardar progreso
completed_trials = len([t for t in trials_data.values() if t["state"] == "COMPLETE"])
total_trials = 30
checkpoint.save_progress(completed_trials, total_trials, best_value, 15)

print("✅ Checkpoint creado exitosamente:")
print(f"   - Trials guardados: {len(trials_data)}")
print(f"   - Trials completados: {completed_trials}")
print(f"   - Mejor F1: {best_value:.4f}")
print("   - Mejor trial: 15")
print(f"   - Directorio: {checkpoint_dir}")

print("\n" + "=" * 70)
print("CHECKPOINT INICIAL CREADO - Ahora puedes continuar desde donde se quedó")
print("=" * 70)
