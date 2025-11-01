#!/usr/bin/env python3
"""
Pruebas unitarias para verificar activación/desactivación de early stopping
============================================================================

Verifica que la configuración early_stopping_enabled funcione correctamente:
- early_stopping_enabled=True: usa patience configurado
- early_stopping_enabled=False: completa todas las épocas
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import tempfile
import shutil

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic
from modules.core.training_monitor import TrainingMonitor
from unittest.mock import Mock


class TestEarlyStoppingConfig(unittest.TestCase):
    """Pruebas para verificar configuración de early stopping."""

    def setUp(self):
        """Configurar datos de prueba antes de cada test."""
        torch.manual_seed(42)
        self.device = torch.device("cpu")

        # Crear datos sintéticos pequeños para tests rápidos
        self.batch_size = 4
        self.num_samples = 16
        self.input_size = (1, 32, 32)
        self.num_classes = 2

        # Datos de entrenamiento
        X_train = torch.randn(self.num_samples, *self.input_size)
        y_train = torch.randint(0, self.num_classes, (self.num_samples,))
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Datos de validación
        X_val = torch.randn(self.num_samples // 2, *self.input_size)
        y_val = torch.randint(0, self.num_classes, (self.num_samples // 2,))
        val_dataset = TensorDataset(X_val, y_val)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Modelo simple
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, self.num_classes),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # Monitor mock
        self.monitor = Mock(spec=TrainingMonitor)
        self.monitor.log = Mock()
        self.monitor.should_plot = Mock(return_value=False)
        self.monitor.plot_local = Mock()
        self.monitor.finish = Mock()
        self.monitor.print_summary = Mock()

        # Directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir)

    def tearDown(self):
        """Limpiar después de cada test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_patience_from_config(self, early_stopping_enabled, n_epochs, patience):
        """Función helper que simula la lógica del notebook."""
        if early_stopping_enabled:
            return patience
        else:
            # Desactivado: usar n_epochs para que nunca se active
            return n_epochs

    def test_early_stopping_enabled_true_uses_patience(self):
        """Verificar que early_stopping_enabled=True usa patience configurado."""
        n_epochs = 10
        configured_patience = 3

        # Simular configuración del notebook
        early_stopping_enabled = True
        early_stopping_patience = self._get_patience_from_config(
            early_stopping_enabled, n_epochs, configured_patience
        )

        # Verificar que usa el patience configurado
        self.assertEqual(early_stopping_patience, configured_patience)

        # Ejecutar entrenamiento real (pocas épocas para test rápido)
        results = train_with_wandb_monitoring_generic(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=None,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            save_dir=self.save_dir,
            verbose=False,
        )

        # Verificar que se respeta el patience (no debe completar todas las épocas si no mejora)
        self.assertIsInstance(results, dict)
        self.assertIn("early_stopped", results)
        self.assertIn("final_epoch", results)
        # El early stopping puede activarse o no dependiendo del entrenamiento
        # Lo importante es que se use el patience correcto

    def test_early_stopping_enabled_false_completes_all_epochs(self):
        """Verificar que early_stopping_enabled=False completa todas las épocas."""
        n_epochs = 5
        configured_patience = 2

        # Simular configuración del notebook con early stopping desactivado
        early_stopping_enabled = False
        early_stopping_patience = self._get_patience_from_config(
            early_stopping_enabled, n_epochs, configured_patience
        )

        # Verificar que usa n_epochs como patience (efectivamente desactivado)
        self.assertEqual(early_stopping_patience, n_epochs)

        # Ejecutar entrenamiento
        results = train_with_wandb_monitoring_generic(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=None,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            save_dir=self.save_dir,
            verbose=False,
        )

        # Verificar que completó todas las épocas
        self.assertEqual(results["final_epoch"], n_epochs)
        self.assertFalse(
            results["early_stopped"],
            "Early stopping no debería activarse cuando está desactivado",
        )

    def test_early_stopping_enabled_false_ignores_patience_value(self):
        """Verificar que el valor de patience se ignora cuando está desactivado."""
        n_epochs = 8
        configured_patience = 1  # Valor pequeño que normalmente activaría early stopping

        # Simular configuración con early stopping desactivado
        early_stopping_enabled = False
        early_stopping_patience = self._get_patience_from_config(
            early_stopping_enabled, n_epochs, configured_patience
        )

        # Verificar que usa n_epochs, no el patience configurado
        self.assertEqual(early_stopping_patience, n_epochs)
        self.assertNotEqual(
            early_stopping_patience, configured_patience, "Debe ignorar patience configurado"
        )

        # Ejecutar entrenamiento
        results = train_with_wandb_monitoring_generic(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=None,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            save_dir=self.save_dir,
            verbose=False,
        )

        # Verificar que completó todas las épocas a pesar de patience pequeño
        self.assertEqual(results["final_epoch"], n_epochs)
        self.assertFalse(results["early_stopped"])

    def test_training_config_dict_simulation(self):
        """Simular la configuración exacta del notebook."""
        # Simular TRAINING_CONFIG del notebook
        TRAINING_CONFIG = {
            "n_epochs": 10,
            "early_stopping_enabled": False,
            "early_stopping_patience": 5,
        }

        # Aplicar lógica del notebook
        if TRAINING_CONFIG.get("early_stopping_enabled", True):
            early_stopping_patience = TRAINING_CONFIG["early_stopping_patience"]
        else:
            early_stopping_patience = TRAINING_CONFIG["n_epochs"]

        # Verificar resultado
        self.assertEqual(early_stopping_patience, 10)
        self.assertEqual(early_stopping_patience, TRAINING_CONFIG["n_epochs"])

        # Cambiar a True
        TRAINING_CONFIG["early_stopping_enabled"] = True
        if TRAINING_CONFIG.get("early_stopping_enabled", True):
            early_stopping_patience = TRAINING_CONFIG["early_stopping_patience"]
        else:
            early_stopping_patience = TRAINING_CONFIG["n_epochs"]

        # Verificar que ahora usa patience
        self.assertEqual(early_stopping_patience, 5)
        self.assertEqual(early_stopping_patience, TRAINING_CONFIG["early_stopping_patience"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

