#!/usr/bin/env python3
"""
Script para corregir la compatibilidad del checkpoint con el espacio de búsqueda actual.
"""

import json
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint


def fix_checkpoint_compatibility():
    """Corregir la compatibilidad del checkpoint."""

    print("=" * 70)
    print("CORRIGIENDO COMPATIBILIDAD DEL CHECKPOINT")
    print("=" * 70)

    # Cargar checkpoint actual
    checkpoint = OptunaCheckpoint(
        checkpoint_dir="checkpoints", experiment_name="cnn2d_optuna"
    )

    # Cargar trials actuales
    trials_data = checkpoint.load_trials()
    print(f"📊 Trials cargados: {len(trials_data)}")

    # Verificar valores problemáticos
    problematic_trials = []
    for trial_id, trial_data in trials_data.items():
        params = trial_data["params"]

        # Verificar filters_1
        if params["filters_1"] not in [16, 32, 64]:
            problematic_trials.append((trial_id, "filters_1", params["filters_1"]))

        # Verificar filters_2
        if params["filters_2"] not in [32, 64, 128]:
            problematic_trials.append((trial_id, "filters_2", params["filters_2"]))

        # Verificar dense_units
        if params["dense_units"] not in [32, 64, 128]:
            problematic_trials.append((trial_id, "dense_units", params["dense_units"]))

    if problematic_trials:
        print(
            f"⚠️  Encontrados {len(problematic_trials)} trials con valores problemáticos:"
        )
        for trial_id, param, value in problematic_trials:
            print(f"   - Trial {trial_id}: {param} = {value}")
    else:
        print("✅ Todos los trials son compatibles con el espacio de búsqueda actual")
        return

    # Crear nuevo checkpoint compatible
    print("\n🔧 Creando nuevo checkpoint compatible...")

    # Filtrar solo trials compatibles
    compatible_trials = {}
    for trial_id, trial_data in trials_data.items():
        params = trial_data["params"]

        # Verificar si el trial es compatible
        is_compatible = (
            params["filters_1"] in [16, 32, 64]
            and params["filters_2"] in [32, 64, 128]
            and params["dense_units"] in [32, 64, 128]
        )

        if is_compatible:
            compatible_trials[trial_id] = trial_data
        else:
            print(f"   - Excluyendo trial {trial_id} (incompatible)")

    print(f"✅ Trials compatibles: {len(compatible_trials)}/{len(trials_data)}")

    # Guardar nuevo checkpoint
    checkpoint_dir = Path("checkpoints")
    backup_dir = checkpoint_dir / "backup"
    backup_dir.mkdir(exist_ok=True)

    # Hacer backup del checkpoint actual
    print(f"💾 Haciendo backup del checkpoint actual en {backup_dir}")
    import shutil

    for file_name in [
        "cnn2d_optuna_trials.json",
        "cnn2d_optuna_best_params.json",
        "cnn2d_optuna_progress.json",
    ]:
        src = checkpoint_dir / file_name
        dst = backup_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)

    # Guardar trials compatibles
    trials_file = checkpoint_dir / "cnn2d_optuna_trials.json"
    with open(trials_file, "w") as f:
        json.dump(compatible_trials, f, indent=2)

    # Encontrar mejor trial compatible
    best_trial = None
    best_value = -1
    for trial_data in compatible_trials.values():
        if trial_data["state"] == "COMPLETE" and trial_data["value"] > best_value:
            best_value = trial_data["value"]
            best_trial = trial_data

    if best_trial:
        # Guardar mejores parámetros
        best_params_data = {
            "best_params": best_trial["params"],
            "best_value": best_value,
            "timestamp": "2025-10-27T22:30:00.000000",
        }

        best_params_file = checkpoint_dir / "cnn2d_optuna_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(best_params_data, f, indent=2)

        # Guardar progreso
        completed_trials = len(
            [t for t in compatible_trials.values() if t["state"] == "COMPLETE"]
        )
        progress_data = {
            "completed_trials": completed_trials,
            "total_trials": 30,
            "progress_percentage": (completed_trials / 30) * 100,
            "best_value": best_value,
            "best_trial": best_trial["number"],
            "timestamp": "2025-10-27T22:30:00.000000",
        }

        progress_file = checkpoint_dir / "cnn2d_optuna_progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

        print(f"\n✅ Checkpoint corregido:")
        print(f"   - Trials compatibles: {len(compatible_trials)}")
        print(f"   - Trials completados: {completed_trials}")
        print(f"   - Mejor F1: {best_value:.4f}")
        print(f"   - Mejor trial: {best_trial['number']}")
        print(f"   - Backup guardado en: {backup_dir}")

    print("\n🚀 ¡Checkpoint corregido! Ahora puedes continuar con Optuna")
    print("=" * 70)


if __name__ == "__main__":
    fix_checkpoint_compatibility()
