"""
Tests para el wrapper de CNN2D con Optuna.

Verifica que la integración de CNN2D con Optuna funcione correctamente.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.cnn2d_optuna_wrapper import (
    CNN2DOptunaWrapper,
    create_cnn2d_optimizer,
    optimize_cnn2d,
)


class TestCNN2DOptunaWrapper(unittest.TestCase):
    """Tests para CNN2DOptunaWrapper."""

    def setUp(self):
        """Configuración inicial."""
        torch.manual_seed(42)

        # Datos de prueba (espectrogramas pequeños)
        self.input_shape = (1, 32, 32)  # (C, H, W)
        self.n_samples = 50

        self.X_train = torch.randn(self.n_samples, *self.input_shape)
        self.y_train = torch.randint(0, 2, (self.n_samples,))
        self.X_val = torch.randn(20, *self.input_shape)
        self.y_val = torch.randint(0, 2, (20,))

    def test_create_wrapper(self):
        """Verificar creación de wrapper."""
        wrapper = CNN2DOptunaWrapper(input_shape=self.input_shape, device="cpu")
        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.input_shape, self.input_shape)
        self.assertEqual(wrapper.device, "cpu")

    def test_suggest_hyperparameters(self):
        """Verificar que sugiere hiperparámetros correctos."""
        import optuna

        wrapper = CNN2DOptunaWrapper(input_shape=self.input_shape, device="cpu")

        # Crear un trial de prueba
        study = optuna.create_study()
        trial = study.ask()

        params = wrapper.suggest_hyperparameters(trial)

        # Verificar que contiene los parámetros esperados
        expected_params = [
            "filters_1",
            "filters_2",
            "kernel_size_1",
            "kernel_size_2",
            "p_drop_conv",
            "p_drop_fc",
            "dense_units",
            "learning_rate",
            "weight_decay",
            "optimizer",
        ]

        for param in expected_params:
            self.assertIn(param, params)

    def test_create_model(self):
        """Verificar creación de modelo."""
        import optuna

        wrapper = CNN2DOptunaWrapper(input_shape=self.input_shape, device="cpu")

        study = optuna.create_study()
        trial = study.ask()

        model = wrapper.create_model(trial)

        self.assertIsInstance(model, nn.Module)

        # Verificar que el modelo puede procesar datos
        with torch.no_grad():
            output = model(self.X_train[:5])
            self.assertEqual(output.shape, (5, 2))  # 2 clases

    def test_create_cnn2d_optimizer(self):
        """Verificar función de conveniencia."""
        optimizer = create_cnn2d_optimizer(
            input_shape=self.input_shape,
            experiment_name="test_cnn2d",
            n_trials=2,
            n_epochs_per_trial=2,
            device="cpu",
        )

        self.assertIsNotNone(optimizer)


class TestOptimizeCNN2D(unittest.TestCase):
    """Tests para la función optimize_cnn2d."""

    def setUp(self):
        """Configuración inicial."""
        torch.manual_seed(42)

        self.input_shape = (1, 32, 32)
        self.n_samples = 50

        self.X_train = torch.randn(self.n_samples, *self.input_shape)
        self.y_train = torch.randint(0, 2, (self.n_samples,))
        self.X_val = torch.randn(20, *self.input_shape)
        self.y_val = torch.randint(0, 2, (20,))

    def test_optimize_cnn2d_runs(self):
        """Verificar que optimize_cnn2d se ejecuta (test rápido)."""
        results = optimize_cnn2d(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            input_shape=self.input_shape,
            n_trials=2,  # Pocos trials para test rápido
            n_epochs_per_trial=2,  # Pocas épocas
            device="cpu",
        )

        # Verificar estructura de resultados
        self.assertIsInstance(results, dict)
        self.assertIn("best_params", results)
        self.assertIn("best_value", results)
        self.assertIn("best_trial", results)
        self.assertIn("analysis", results)

    def test_optimize_cnn2d_best_params(self):
        """Verificar que retorna mejores parámetros válidos."""
        results = optimize_cnn2d(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            input_shape=self.input_shape,
            n_trials=2,
            n_epochs_per_trial=2,
            device="cpu",
        )

        best_params = results["best_params"]

        # Verificar que tiene los parámetros esperados
        self.assertIn("filters_1", best_params)
        self.assertIn("learning_rate", best_params)
        self.assertIn("batch_size", best_params)

        # Verificar rangos válidos
        self.assertIn(best_params["filters_1"], [16, 32, 64])
        self.assertGreater(best_params["learning_rate"], 0)


def run_tests():
    """Ejecutar todos los tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
