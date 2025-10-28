#!/usr/bin/env python3
"""
Prueba rápida para verificar que el guardado de trials funciona correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint
import optuna
from optuna.trial import Trial


def test_trial_saving():
    """Probar el guardado de trials con diferentes estados."""

    print("=" * 70)
    print("PRUEBA DE GUARDADO DE TRIALS")
    print("=" * 70)

    # Crear checkpoint de prueba
    checkpoint = OptunaCheckpoint(
        checkpoint_dir="test_checkpoints", experiment_name="test_trial_saving"
    )

    # Crear un trial simulado
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    # Simular parámetros
    trial.params = {
        "filters_1": 32,
        "filters_2": 64,
        "kernel_size_1": 5,
        "kernel_size_2": 7,
        "p_drop_conv": 0.2,
        "p_drop_fc": 0.5,
        "dense_units": 32,
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
    }

    # Simular métricas
    metrics = {
        "f1_macro": 0.75,
        "accuracy": 0.80,
        "precision_macro": 0.78,
        "recall_macro": 0.72,
    }

    try:
        # Probar guardado de trial completado
        print("📊 Probando guardado de trial completado...")
        checkpoint.save_trial(trial, metrics, state="COMPLETE")
        print("✅ Trial completado guardado correctamente")

        # Probar guardado de trial pruned
        print("\n📊 Probando guardado de trial pruned...")
        checkpoint.save_pruned_trial(trial, epoch=5, f1_value=0.65)
        print("✅ Trial pruned guardado correctamente")

        # Verificar que se guardaron
        trials_data = checkpoint.load_trials()
        print(f"\n📈 Trials guardados: {len(trials_data)}")

        for trial_id, trial_data in trials_data.items():
            print(
                f"   - Trial {trial_data['number']}: {trial_data['state']} (F1={trial_data['value']:.4f})"
            )

        print("\n🎉 ¡Guardado de trials funcionando correctamente!")
        return True

    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        return False


if __name__ == "__main__":
    success = test_trial_saving()
    sys.exit(0 if success else 1)
