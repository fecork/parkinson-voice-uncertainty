#!/usr/bin/env python3
"""
Validación simple de los valores en el checkpoint actual vs especificaciones de Ibarra.
"""

import json
from pathlib import Path


def validate_checkpoint_values():
    """Validar los valores del checkpoint actual."""

    print("=" * 70)
    print("VALIDACIÓN DE VALORES DEL CHECKPOINT ACTUAL")
    print("=" * 70)

    # Especificaciones del paper de Ibarra
    ibarra_specs = {
        "batch_size": [16, 32, 64],
        "dropout_rate": [0.2, 0.5],
        "depth_conv_layer": [32, 64, 128],
        "fc_units": [16, 32, 64],
        "kernel_size_i": [4, 6, 8],
        "kernel_size_ii": [5, 7, 9],
    }

    # Cargar checkpoint actual
    checkpoint_file = Path("checkpoints/cnn2d_optuna_trials.json")

    if not checkpoint_file.exists():
        print("❌ No se encontró el archivo de checkpoint")
        return False

    with open(checkpoint_file, "r") as f:
        trials_data = json.load(f)

    print(f"📊 Trials en el checkpoint: {len(trials_data)}")

    # Analizar valores únicos
    unique_values = {
        "batch_size": set(),
        "p_drop_conv": set(),
        "p_drop_fc": set(),
        "filters_1": set(),
        "filters_2": set(),
        "dense_units": set(),
        "kernel_size_1": set(),
        "kernel_size_2": set(),
    }

    # Recopilar valores de trials completados
    completed_trials = 0
    for trial_id, trial_data in trials_data.items():
        if trial_data.get("state") == "COMPLETE":
            completed_trials += 1
            params = trial_data.get("params", {})

            for key in unique_values.keys():
                if key in params:
                    unique_values[key].add(params[key])

    print(f"📈 Trials completados: {completed_trials}")

    # Mostrar valores encontrados vs esperados
    print(f"\n🔍 ANÁLISIS DE VALORES:")

    issues_found = []

    # Batch size
    batch_values = sorted(list(unique_values["batch_size"]))
    expected_batch = ibarra_specs["batch_size"]
    if batch_values == expected_batch:
        print(f"✅ batch_size: {batch_values} - CORRECTO")
    else:
        print(f"❌ batch_size: {batch_values} vs {expected_batch} - INCORRECTO")
        issues_found.append("batch_size")

    # Dropout rates
    conv_dropout_values = sorted(list(unique_values["p_drop_conv"]))
    fc_dropout_values = sorted(list(unique_values["p_drop_fc"]))
    expected_dropout = ibarra_specs["dropout_rate"]

    if conv_dropout_values == expected_dropout:
        print(f"✅ p_drop_conv: {conv_dropout_values} - CORRECTO")
    else:
        print(
            f"❌ p_drop_conv: {conv_dropout_values} vs {expected_dropout} - INCORRECTO"
        )
        issues_found.append("p_drop_conv")

    if fc_dropout_values == expected_dropout:
        print(f"✅ p_drop_fc: {fc_dropout_values} - CORRECTO")
    else:
        print(f"❌ p_drop_fc: {fc_dropout_values} vs {expected_dropout} - INCORRECTO")
        issues_found.append("p_drop_fc")

    # Depth conv layers
    filters_1_values = sorted(list(unique_values["filters_1"]))
    filters_2_values = sorted(list(unique_values["filters_2"]))
    expected_filters = ibarra_specs["depth_conv_layer"]

    if filters_1_values == expected_filters:
        print(f"✅ filters_1: {filters_1_values} - CORRECTO")
    else:
        print(f"❌ filters_1: {filters_1_values} vs {expected_filters} - INCORRECTO")
        issues_found.append("filters_1")

    if filters_2_values == expected_filters:
        print(f"✅ filters_2: {filters_2_values} - CORRECTO")
    else:
        print(f"❌ filters_2: {filters_2_values} vs {expected_filters} - INCORRECTO")
        issues_found.append("filters_2")

    # FC units
    dense_units_values = sorted(list(unique_values["dense_units"]))
    expected_dense = ibarra_specs["fc_units"]

    if dense_units_values == expected_dense:
        print(f"✅ dense_units: {dense_units_values} - CORRECTO")
    else:
        print(f"❌ dense_units: {dense_units_values} vs {expected_dense} - INCORRECTO")
        issues_found.append("dense_units")

    # Kernel sizes
    kernel_1_values = sorted(list(unique_values["kernel_size_1"]))
    kernel_2_values = sorted(list(unique_values["kernel_size_2"]))
    expected_kernel_1 = ibarra_specs["kernel_size_i"]
    expected_kernel_2 = ibarra_specs["kernel_size_ii"]

    if kernel_1_values == expected_kernel_1:
        print(f"✅ kernel_size_1: {kernel_1_values} - CORRECTO")
    else:
        print(
            f"❌ kernel_size_1: {kernel_1_values} vs {expected_kernel_1} - INCORRECTO"
        )
        issues_found.append("kernel_size_1")

    if kernel_2_values == expected_kernel_2:
        print(f"✅ kernel_size_2: {kernel_2_values} - CORRECTO")
    else:
        print(
            f"❌ kernel_size_2: {kernel_2_values} vs {expected_kernel_2} - INCORRECTO"
        )
        issues_found.append("kernel_size_2")

    # Conclusión
    print(f"\n" + "=" * 70)
    if not issues_found:
        print("🎉 ¡CHECKPOINT COMPLETAMENTE COMPATIBLE CON IBARRA!")
    else:
        print(f"❌ CHECKPOINT INCOMPATIBLE CON IBARRA:")
        print(f"   - Parámetros con problemas: {', '.join(issues_found)}")
        print(f"   - El checkpoint fue creado con valores incorrectos")
        print(f"   - Se necesita crear un nuevo checkpoint con valores correctos")
    print("=" * 70)

    return len(issues_found) == 0


if __name__ == "__main__":
    validate_checkpoint_values()
