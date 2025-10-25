"""
Pruebas unitarias para el scheduler de learning rate
===================================================

Verifica que el ReduceLROnPlateau scheduler funciona correctamente
con los parámetros configurados en el CNN2D Talos wrapper.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from modules.core.cnn2d_talos_wrapper import CNN2DTalosWrapper


class DictDataset(torch.utils.data.Dataset):
    """Dataset que devuelve diccionarios en lugar de tuplas."""

    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return {"spectrogram": self.spectrograms[idx], "label": self.labels[idx]}


class TestLearningRateScheduler(unittest.TestCase):
    """Pruebas para el scheduler de learning rate."""

    def setUp(self):
        """Configuración inicial."""
        self.wrapper = CNN2DTalosWrapper()
        self.device = torch.device("cpu")  # Usar CPU para pruebas

        # Crear datos sintéticos
        self.n_samples = 100
        self.input_shape = (65, 41)
        self.n_classes = 2

        # Generar datos de entrenamiento
        X_train = torch.randn(self.n_samples, 1, *self.input_shape)
        y_train = torch.randint(0, self.n_classes, (self.n_samples,))

        # Crear DataLoaders usando DictDataset
        train_dataset = DictDataset(X_train, y_train)
        val_dataset = DictDataset(X_train, y_train)  # Mismo para simplicidad

        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    def test_scheduler_initialization(self):
        """Verificar que el scheduler se inicializa correctamente."""
        # Crear modelo
        params = {
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.3,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 6,
            "kernel_size_2": 7,
            "dense_units": 64,
        }

        model = self.wrapper.create_model(params)
        model = model.to(self.device)

        # Crear optimizer y scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Verificar configuración inicial
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 5)
        self.assertEqual(scheduler.mode, "min")

    def test_scheduler_reduction_behavior(self):
        """Verificar que el scheduler reduce el learning rate correctamente."""
        # Crear modelo
        params = {
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.3,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 6,
            "kernel_size_2": 7,
            "dense_units": 64,
        }

        model = self.wrapper.create_model(params)
        model = model.to(self.device)

        # Crear optimizer y scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Simular entrenamiento con pérdidas que no mejoran
        initial_lr = optimizer.param_groups[0]["lr"]
        lr_history = [initial_lr]

        # Simular 15 épocas con pérdidas que no mejoran
        for epoch in range(15):
            # Simular pérdida que no mejora (plateau)
            val_loss = 0.5 + epoch * 0.01  # Pérdida que empeora ligeramente
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)

            # El scheduler reduce después de patience+1 épocas sin mejora
            # Primera reducción en época 6 (después de 5+1 épocas sin mejora)
            # Segunda reducción en época 12 (después de otras 5+1 épocas)
            if epoch < 6:
                expected_lr = 0.1
            elif epoch < 12:
                expected_lr = 0.05
            else:
                expected_lr = 0.025
            self.assertAlmostEqual(current_lr, expected_lr, places=6)

        # Verificar que el LR se redujo
        self.assertLess(lr_history[-1], lr_history[0])
        print(f"Learning rate inicial: {lr_history[0]}")
        print(f"Learning rate final: {lr_history[-1]}")
        print(f"Historial de LR: {lr_history}")

    def test_scheduler_with_improving_loss(self):
        """Verificar que el scheduler no reduce LR cuando la pérdida mejora."""
        # Crear modelo
        params = {
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.3,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 6,
            "kernel_size_2": 7,
            "dense_units": 64,
        }

        model = self.wrapper.create_model(params)
        model = model.to(self.device)

        # Crear optimizer y scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Simular entrenamiento con pérdidas que mejoran
        initial_lr = optimizer.param_groups[0]["lr"]
        lr_history = [initial_lr]

        # Simular 10 épocas con pérdidas que mejoran
        for epoch in range(10):
            # Simular pérdida que mejora
            val_loss = 0.5 - epoch * 0.01  # Pérdida que mejora
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)

            # El LR no debería cambiar si la pérdida mejora
            self.assertEqual(current_lr, initial_lr)

        print(f"Learning rate se mantuvo constante: {lr_history[0]}")

    def test_scheduler_integration_with_training(self):
        """Verificar que el scheduler se integra correctamente con el entrenamiento."""
        # Crear modelo
        params = {
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.3,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 6,
            "kernel_size_2": 7,
            "dense_units": 64,
        }

        # Simular entrenamiento con el wrapper
        try:
            f1_score, metrics = self.wrapper.train_model(
                model=self.wrapper.create_model(params),
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                params=params,
                n_epochs=3,  # Pocas épocas para prueba rápida
            )

            # Verificar que el entrenamiento completó sin errores
            self.assertIsInstance(f1_score, float)
            self.assertIsInstance(metrics, dict)
            self.assertIn("f1", metrics)
            self.assertIn("val_loss", metrics)

            print(f"Entrenamiento completado - F1: {f1_score:.4f}")
            print(f"Métricas: {metrics}")

        except Exception as e:
            self.fail(f"Error en entrenamiento con scheduler: {e}")

    def test_scheduler_parameters_validation(self):
        """Verificar que los parámetros del scheduler son correctos."""
        # Crear optimizer
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        # Crear scheduler con parámetros específicos
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            threshold_mode="rel",
        )

        # Verificar parámetros
        self.assertEqual(scheduler.mode, "min")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 5)
        self.assertEqual(scheduler.threshold, 1e-4)
        self.assertEqual(scheduler.threshold_mode, "rel")

    def test_scheduler_with_different_factors(self):
        """Verificar que diferentes factores de reducción funcionan correctamente."""
        model = nn.Linear(10, 2)

        # Probar diferentes factores
        factors = [0.1, 0.5, 0.8]

        for factor in factors:
            with self.subTest(factor=factor):
                optimizer = optim.Adam(model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=factor,
                    patience=1,
                )

                # Simular pérdida que no mejora
                scheduler.step(0.5)  # Primera época (establece baseline)
                scheduler.step(0.6)  # Segunda época (peor, cuenta=1)
                scheduler.step(0.65)  # Tercera época (peor, cuenta=2, activa reducción)

                # Verificar que el LR se redujo con el factor correcto
                expected_lr = 0.1 * factor
                actual_lr = optimizer.param_groups[0]["lr"]
                self.assertAlmostEqual(actual_lr, expected_lr, places=6)

                print(f"Factor {factor}: LR {0.1} -> {actual_lr}")


if __name__ == "__main__":
    unittest.main()
