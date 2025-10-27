#!/usr/bin/env python3
"""
Pruebas para validar si el checkpoint actual es compatible con las especificaciones de Ibarra.
"""

import json
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint


def validate_checkpoint_ibarra_compatibility():
    """Validar si el checkpoint actual es compatible con las especificaciones de Ibarra."""

    print("=" * 70)
    print("VALIDACIÓN DE COMPATIBILIDAD CHECKPOINT vs PAPER IBARRA")
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
    checkpoint = OptunaCheckpoint(
        checkpoint_dir="checkpoints", experiment_name="cnn2d_optuna"
    )

    try:
        # Cargar trials del checkpoint
        trials_df = checkpoint.create_dataframe_from_checkpoint()
        print(f"📊 Trials cargados del checkpoint: {len(trials_df)}")

        # Analizar cada trial
        incompatible_trials = []
        compatible_trials = []

        for _, trial in trials_df.iterrows():
            if trial["state"] == "COMPLETE":
                trial_params = trial["params"]

                # Verificar compatibilidad con Ibarra
                is_compatible = True
                issues = []

                # Verificar batch_size
                if trial_params.get("batch_size") not in ibarra_specs["batch_size"]:
                    is_compatible = False
                    issues.append(
                        f"batch_size: {trial_params.get('batch_size')} not in {ibarra_specs['batch_size']}"
                    )

                # Verificar dropout rates
                if trial_params.get("p_drop_conv") not in ibarra_specs["dropout_rate"]:
                    is_compatible = False
                    issues.append(
                        f"p_drop_conv: {trial_params.get('p_drop_conv')} not in {ibarra_specs['dropout_rate']}"
                    )

                if trial_params.get("p_drop_fc") not in ibarra_specs["dropout_rate"]:
                    is_compatible = False
                    issues.append(
                        f"p_drop_fc: {trial_params.get('p_drop_fc')} not in {ibarra_specs['dropout_rate']}"
                    )

                # Verificar depth conv layers
                if (
                    trial_params.get("filters_1")
                    not in ibarra_specs["depth_conv_layer"]
                ):
                    is_compatible = False
                    issues.append(
                        f"filters_1: {trial_params.get('filters_1')} not in {ibarra_specs['depth_conv_layer']}"
                    )

                if (
                    trial_params.get("filters_2")
                    not in ibarra_specs["depth_conv_layer"]
                ):
                    is_compatible = False
                    issues.append(
                        f"filters_2: {trial_params.get('filters_2')} not in {ibarra_specs['depth_conv_layer']}"
                    )

                # Verificar FC units
                if trial_params.get("dense_units") not in ibarra_specs["fc_units"]:
                    is_compatible = False
                    issues.append(
                        f"dense_units: {trial_params.get('dense_units')} not in {ibarra_specs['fc_units']}"
                    )

                # Verificar kernel sizes
                if (
                    trial_params.get("kernel_size_1")
                    not in ibarra_specs["kernel_size_i"]
                ):
                    is_compatible = False
                    issues.append(
                        f"kernel_size_1: {trial_params.get('kernel_size_1')} not in {ibarra_specs['kernel_size_i']}"
                    )

                if (
                    trial_params.get("kernel_size_2")
                    not in ibarra_specs["kernel_size_ii"]
                ):
                    is_compatible = False
                    issues.append(
                        f"kernel_size_2: {trial_params.get('kernel_size_2')} not in {ibarra_specs['kernel_size_ii']}"
                    )

                if is_compatible:
                    compatible_trials.append(trial["number"])
                else:
                    incompatible_trials.append(
                        {
                            "trial": trial["number"],
                            "issues": issues,
                            "params": trial_params,
                        }
                    )

        # Mostrar resultados
        print(f"\n📈 RESULTADOS DE COMPATIBILIDAD:")
        print(f"   - Trials compatibles: {len(compatible_trials)}")
        print(f"   - Trials incompatibles: {len(incompatible_trials)}")

        if compatible_trials:
            print(f"\n✅ Trials compatibles: {compatible_trials}")

        if incompatible_trials:
            print(f"\n❌ Trials incompatibles:")
            for trial_info in incompatible_trials:
                print(f"   - Trial {trial_info['trial']}:")
                for issue in trial_info["issues"]:
                    print(f"     * {issue}")

        # Análisis de valores únicos encontrados
        print(f"\n🔍 ANÁLISIS DE VALORES ÚNICOS EN EL CHECKPOINT:")

        # Recopilar todos los valores únicos
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

        for _, trial in trials_df.iterrows():
            if trial["state"] == "COMPLETE":
                trial_params = trial["params"]
                for key in unique_values.keys():
                    if key in trial_params:
                        unique_values[key].add(trial_params[key])

        # Mostrar valores únicos vs especificaciones
        for param, values in unique_values.items():
            if param == "batch_size":
                expected = ibarra_specs["batch_size"]
            elif param in ["p_drop_conv", "p_drop_fc"]:
                expected = ibarra_specs["dropout_rate"]
            elif param in ["filters_1", "filters_2"]:
                expected = ibarra_specs["depth_conv_layer"]
            elif param == "dense_units":
                expected = ibarra_specs["fc_units"]
            elif param == "kernel_size_1":
                expected = ibarra_specs["kernel_size_i"]
            elif param == "kernel_size_2":
                expected = ibarra_specs["kernel_size_ii"]
            else:
                expected = "N/A"

            values_list = sorted(list(values))
            expected_list = sorted(expected) if expected != "N/A" else expected

            if values_list == expected_list:
                status = "✅"
            else:
                status = "❌"

            print(f"   {status} {param}: {values_list} vs {expected_list}")

        # Conclusión
        print(f"\n" + "=" * 70)
        if len(incompatible_trials) == 0:
            print("🎉 ¡CHECKPOINT COMPLETAMENTE COMPATIBLE CON IBARRA!")
        else:
            print(f"⚠️  CHECKPOINT PARCIALMENTE INCOMPATIBLE:")
            print(
                f"   - {len(compatible_trials)}/{len(compatible_trials) + len(incompatible_trials)} trials compatibles"
            )
            print(f"   - Se recomienda crear un nuevo checkpoint con valores correctos")
        print("=" * 70)

        return len(incompatible_trials) == 0

    except Exception as e:
        print(f"❌ Error al validar checkpoint: {e}")
        return False


if __name__ == "__main__":
    success = validate_checkpoint_ibarra_compatibility()
    sys.exit(0 if success else 1)
