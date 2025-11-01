#!/usr/bin/env python3
"""
Prueba para verificar que el manejo del valor del trial funciona correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint


def test_trial_value_handling():
    """Probar el manejo del valor del trial."""

    print("=" * 70)
    print("PRUEBA DE MANEJO DE VALOR DE TRIAL")
    print("=" * 70)

    # Crear checkpoint de prueba
    checkpoint = OptunaCheckpoint(
        checkpoint_dir="test_checkpoints", experiment_name="test_value_handling"
    )

    # Simular un trial con diferentes escenarios
    class MockTrial:
        def __init__(self, number, has_value=True, value=None):
            self.number = number
            self.params = {
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
            self.datetime_start = None
            self.datetime_complete = None
            self._has_value = has_value
            self._value = value

        @property
        def value(self):
            if self._has_value:
                return self._value
            else:
                raise AttributeError("'Trial' object has no attribute 'value'")

    # Métricas de prueba
    metrics = {
        "f1_macro": 0.75,
        "accuracy": 0.80,
        "precision_macro": 0.78,
        "recall_macro": 0.72,
    }

    try:
        # Escenario 1: Trial con valor
        print("📊 Escenario 1: Trial con valor...")
        trial1 = MockTrial(1, has_value=True, value=0.75)
        checkpoint.save_trial(trial1, metrics, state="COMPLETE")
        print("✅ Trial con valor guardado correctamente")

        # Escenario 2: Trial sin valor (simulando el error)
        print("\n📊 Escenario 2: Trial sin valor...")
        trial2 = MockTrial(2, has_value=False)
        checkpoint.save_trial(trial2, metrics, state="COMPLETE")
        print("✅ Trial sin valor guardado correctamente (usando F1 de metrics)")

        # Escenario 3: Trial pruned
        print("\n📊 Escenario 3: Trial pruned...")
        trial3 = MockTrial(3, has_value=False)
        checkpoint.save_pruned_trial(trial3, epoch=5, f1_value=0.65)
        print("✅ Trial pruned guardado correctamente")

        # Verificar que se guardaron
        trials_data = checkpoint.load_trials()
        print(f"\n📈 Trials guardados: {len(trials_data)}")

        for trial_id, trial_data in trials_data.items():
            print(
                f"   - Trial {trial_data['number']}: {trial_data['state']} (F1={trial_data['value']:.4f})"
            )

        print("\n🎉 ¡Manejo de valor de trial funcionando correctamente!")
        return True

    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trial_value_handling()
    sys.exit(0 if success else 1)
