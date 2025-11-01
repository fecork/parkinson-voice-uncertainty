"""
Tests básicos para el módulo de optimización con Optuna.

Verifica que las funciones principales de optimización funcionen correctamente.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.optuna_optimization import (
    OptunaOptimizer,
    OptunaModelWrapper,
)
from optuna.trial import Trial


class DummyModelWrapper(OptunaModelWrapper):
    """Wrapper de prueba simple."""

    def __init__(self, input_size=10, output_size=2):
        self.input_size = input_size
        self.output_size = output_size

    def create_model(self, trial: Trial) -> nn.Module:
        """Crear modelo simple de prueba."""
        hidden_size = trial.suggest_int("hidden_size", 16, 32)

        model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )
        return model

    def train_model(self, model, train_loader, val_loader, trial, n_epochs=2):
        """Entrenar modelo simple."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training rápido
        for epoch in range(n_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            accuracy = correct / total

            # Reportar métrica intermedia
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return accuracy, {"accuracy": accuracy}

    def suggest_hyperparameters(self, trial: Trial):
        """Sugerir hiperparámetros."""
        return {
            "hidden_size": trial.suggest_int("hidden_size", 16, 32),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }


class TestOptunaBasic(unittest.TestCase):
    """Tests básicos de Optuna."""

    def setUp(self):
        """Configuración inicial."""
        # Crear datos de prueba
        torch.manual_seed(42)
        self.n_samples = 100
        self.input_size = 10
        self.output_size = 2

        self.X_train = torch.randn(self.n_samples, self.input_size)
        self.y_train = torch.randint(0, self.output_size, (self.n_samples,))
        self.X_val = torch.randn(50, self.input_size)
        self.y_val = torch.randint(0, self.output_size, (50,))

    def test_optuna_installation(self):
        """Verificar que Optuna está instalado."""
        try:
            import optuna

            self.assertTrue(True)
        except ImportError:
            self.fail("Optuna no está instalado")

    def test_create_optimizer(self):
        """Verificar creación de OptunaOptimizer."""
        wrapper = DummyModelWrapper()
        optimizer = OptunaOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_optuna",
            n_trials=2,
            n_epochs_per_trial=2,
        )
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.n_trials, 2)

    def test_optimization_runs(self):
        """Verificar que la optimización se ejecuta."""
        wrapper = DummyModelWrapper()
        optimizer = OptunaOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_optuna_run",
            n_trials=3,
            n_epochs_per_trial=2,
        )

        # Ejecutar optimización
        results_df = optimizer.optimize(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            show_progress=False,
        )

        # Verificar resultados
        self.assertIsNotNone(results_df)
        self.assertEqual(len(results_df), 3)  # 3 trials

    def test_get_best_params(self):
        """Verificar obtención de mejores parámetros."""
        wrapper = DummyModelWrapper()
        optimizer = OptunaOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_best_params",
            n_trials=3,
            n_epochs_per_trial=2,
        )

        optimizer.optimize(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            show_progress=False,
        )

        best_params = optimizer.get_best_params()
        self.assertIsInstance(best_params, dict)
        self.assertIn("hidden_size", best_params)
        self.assertIn("batch_size", best_params)

    def test_get_best_value(self):
        """Verificar obtención del mejor valor."""
        wrapper = DummyModelWrapper()
        optimizer = OptunaOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_best_value",
            n_trials=3,
            n_epochs_per_trial=2,
        )

        optimizer.optimize(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            show_progress=False,
        )

        best_value = optimizer.get_best_value()
        self.assertIsInstance(best_value, float)
        self.assertGreater(best_value, 0)

    def test_analyze_results(self):
        """Verificar análisis de resultados."""
        wrapper = DummyModelWrapper()
        optimizer = OptunaOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_analyze",
            n_trials=3,
            n_epochs_per_trial=2,
        )

        optimizer.optimize(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            show_progress=False,
        )

        analysis = optimizer.analyze_results()
        self.assertIsInstance(analysis, dict)
        self.assertIn("best_params", analysis)
        self.assertIn("best_value", analysis)
        self.assertIn("n_trials", analysis)


class TestOptunaModelWrapper(unittest.TestCase):
    """Tests para OptunaModelWrapper."""

    def test_wrapper_interface(self):
        """Verificar que wrapper implementa interfaz correcta."""
        wrapper = DummyModelWrapper()

        # Verificar que tiene los métodos requeridos
        self.assertTrue(hasattr(wrapper, "create_model"))
        self.assertTrue(hasattr(wrapper, "train_model"))
        self.assertTrue(hasattr(wrapper, "suggest_hyperparameters"))


def run_tests():
    """Ejecutar todos los tests."""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
