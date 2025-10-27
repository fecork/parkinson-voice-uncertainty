#!/usr/bin/env python3
"""
Prueba para verificar la configuración de pruning agresivo en Optuna.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.cnn2d_optuna_wrapper import optimize_cnn2d
import torch
import numpy as np


def test_optuna_pruning_config():
    """Probar la configuración de pruning agresivo."""

    print("=" * 70)
    print("PRUEBA DE CONFIGURACIÓN DE PRUNING AGRESIVO")
    print("=" * 70)

    # Crear datos de prueba pequeños
    print("📊 Creando datos de prueba...")
    n_samples = 100
    X_train = torch.randn(n_samples, 1, 65, 41)
    y_train = torch.randint(0, 2, (n_samples,))
    X_val = torch.randn(20, 1, 65, 41)
    y_val = torch.randint(0, 2, (20,))

    print(f"   - Train: {X_train.shape}")
    print(f"   - Val: {X_val.shape}")

    # Configuración de prueba
    print("\n⚙️  Configuración de prueba:")
    print("   - Épocas por trial: 10 (reducido de 20)")
    print("   - Pruning patience: 3 épocas")
    print("   - Min épocas antes de pruning: 2")
    print("   - Trials: 5 (para prueba rápida)")

    try:
        # Ejecutar optimización con configuración reducida
        results = optimize_cnn2d(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_shape=(1, 65, 41),
            n_trials=5,  # Solo 5 trials para prueba
            n_epochs_per_trial=10,  # 10 épocas por trial
            device="cpu",  # Usar CPU para prueba rápida
            save_dir="test_results",
            checkpoint_dir="test_checkpoints",
            resume=False,  # Empezar desde cero
        )

        print("\n✅ Optimización completada exitosamente")
        print(f"   - Trials ejecutados: {len(results['results_df'])}")
        print(f"   - Mejor F1: {results['best_value']:.4f}")

        # Analizar trials pruned
        results_df = results["results_df"]
        pruned_trials = results_df[results_df["state"] == "PRUNED"]
        completed_trials = results_df[results_df["state"] == "COMPLETE"]

        print(f"\n📈 Análisis de pruning:")
        print(f"   - Trials completados: {len(completed_trials)}")
        print(f"   - Trials pruned: {len(pruned_trials)}")
        print(
            f"   - Tasa de pruning: {len(pruned_trials) / len(results_df) * 100:.1f}%"
        )

        if len(pruned_trials) > 0:
            print(f"\n🛑 Trials pruned (primeros 3):")
            for i, (_, trial) in enumerate(pruned_trials.head(3).iterrows()):
                print(f"   - Trial {trial['number']}: F1={trial['value']:.4f}")

        print("\n🎉 ¡Configuración de pruning agresivo funcionando correctamente!")

    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_optuna_pruning_config()
    sys.exit(0 if success else 1)
