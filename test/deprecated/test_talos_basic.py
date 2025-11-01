"""
Pruebas básicas para verificar que los componentes de Talos funcionan correctamente
sin depender de la instalación de Talos.

Este módulo verifica que:
1. Los wrappers se crean correctamente
2. Los modelos se pueden crear con diferentes parámetros
3. Las métricas se calculan correctamente
4. Los archivos se pueden guardar y cargar
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import json
from pathlib import Path
import pandas as pd

# Importar módulos a probar
from modules.core.cnn2d_talos_wrapper import CNN2DTalosWrapper
from modules.models.cnn2d.model import CNN2D


class TestTalosBasic(unittest.TestCase):
    """Pruebas básicas para verificar que los componentes funcionan correctamente."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos sintéticos para pruebas
        np.random.seed(42)
        torch.manual_seed(42)

        self.n_samples = 100
        self.input_shape = (65, 41)

        # Datos sintéticos
        self.X_train = np.random.randn(self.n_samples, 1, *self.input_shape).astype(
            np.float32
        )
        self.y_train = np.random.randint(0, 2, self.n_samples).astype(np.int64)
        self.X_val = np.random.randn(20, 1, *self.input_shape).astype(np.float32)
        self.y_val = np.random.randint(0, 2, 20).astype(np.int64)

    def test_cnn2d_wrapper_creation(self):
        """Verificar que CNN2DTalosWrapper se crea correctamente."""
        wrapper = CNN2DTalosWrapper()
        self.assertIsNotNone(wrapper)

        # Verificar parámetros de búsqueda
        search_params = wrapper.get_search_params()
        self.assertIsInstance(search_params, dict)
        self.assertIn("batch_size", search_params)
        self.assertIn("filters_1", search_params)
        self.assertIn("filters_2", search_params)
        # learning_rate fue removido - ahora se usa scheduler
        self.assertNotIn("learning_rate", search_params)

    def test_model_creation_with_different_params(self):
        """Verificar que se pueden crear modelos con diferentes parámetros."""
        wrapper = CNN2DTalosWrapper()

        # Probar diferentes combinaciones de parámetros
        test_params = [
            {
                "batch_size": 16,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.2,
                "filters_1": 32,
                "filters_2": 32,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "dense_units": 16,
            },
            {
                "batch_size": 64,
                "p_drop_conv": 0.5,
                "p_drop_fc": 0.5,
                "filters_1": 128,
                "filters_2": 128,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "dense_units": 64,
            },
        ]

        for params in test_params:
            # Crear modelo
            model = wrapper.create_model(params)
            self.assertIsInstance(model, CNN2D)

            # Verificar forward pass
            test_input = torch.randn(2, 1, 65, 41)
            with torch.no_grad():
                output = model(test_input)
                self.assertEqual(output.shape, (2, 2))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_hyperparameter_space_coverage(self):
        """Verificar que el espacio de hiperparámetros cubre los valores esperados."""
        wrapper = CNN2DTalosWrapper()
        search_params = wrapper.get_search_params()

        # Verificar que tiene todos los parámetros esperados
        expected_params = [
            "batch_size",
            "p_drop_conv",
            "p_drop_fc",
            "filters_1",
            "filters_2",
            "kernel_size_1",
            "kernel_size_2",
            "dense_units",
        ]

        for param in expected_params:
            self.assertIn(param, search_params)
            self.assertIsInstance(search_params[param], list)
            self.assertGreater(len(search_params[param]), 0)

        # Verificar valores específicos según la tabla del paper
        self.assertEqual(search_params["batch_size"], [16, 32, 64])
        self.assertEqual(search_params["p_drop_conv"], [0.2, 0.5])
        self.assertEqual(search_params["p_drop_fc"], [0.2, 0.5])
        self.assertEqual(search_params["filters_1"], [32, 64, 128])
        self.assertEqual(search_params["filters_2"], [32, 64, 128])
        self.assertEqual(search_params["kernel_size_1"], [4, 6, 8])
        self.assertEqual(search_params["kernel_size_2"], [5, 7, 9])
        self.assertEqual(search_params["dense_units"], [16, 32, 64])
        # learning_rate fue removido - ahora se usa scheduler con lr fijo

    def test_model_training_basic(self):
        """Verificar que se puede entrenar un modelo básico."""
        wrapper = CNN2DTalosWrapper()

        # Parámetros de prueba
        params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
        }

        # Crear modelo
        model = wrapper.create_model(params)

        # Crear DataLoaders con DictDataset
        from modules.core.dataset import DictDataset

        train_dataset = DictDataset(
            torch.FloatTensor(self.X_train), torch.LongTensor(self.y_train)
        )
        val_dataset = DictDataset(
            torch.FloatTensor(self.X_val), torch.LongTensor(self.y_val)
        )

        # Función para manejar diccionarios en el DataLoader
        def dict_collate_fn(batch):
            spectrograms = torch.stack([item["spectrogram"] for item in batch])
            labels = torch.stack([item["label"] for item in batch])
            return {"spectrogram": spectrograms, "label": labels}

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=dict_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            collate_fn=dict_collate_fn,
        )

        # Entrenar modelo (pocas épocas para prueba rápida)
        f1_score, metrics = wrapper.train_model(
            model, train_loader, val_loader, params, n_epochs=2
        )

        # Verificar que retorna métricas válidas
        self.assertIsInstance(f1_score, float)
        self.assertIsInstance(metrics, dict)
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

        # Verificar métricas esperadas
        expected_metrics = [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "val_loss",
            "train_loss",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)

    def test_metrics_consistency(self):
        """Verificar que las métricas son consistentes."""
        wrapper = CNN2DTalosWrapper()

        # Parámetros fijos
        params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
        }

        # Crear DataLoaders con DictDataset
        from modules.core.dataset import DictDataset

        train_dataset = DictDataset(
            torch.FloatTensor(self.X_train), torch.LongTensor(self.y_train)
        )
        val_dataset = DictDataset(
            torch.FloatTensor(self.X_val), torch.LongTensor(self.y_val)
        )

        # Función para manejar diccionarios en el DataLoader
        def dict_collate_fn(batch):
            spectrograms = torch.stack([item["spectrogram"] for item in batch])
            labels = torch.stack([item["label"] for item in batch])
            return {"spectrogram": spectrograms, "label": labels}

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=dict_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            collate_fn=dict_collate_fn,
        )

        # Ejecutar entrenamiento múltiples veces
        results = []
        for i in range(3):
            # Crear nuevo modelo para cada ejecución
            model = wrapper.create_model(params)

            f1_score, metrics = wrapper.train_model(
                model, train_loader, val_loader, params, n_epochs=1
            )

            results.append((f1_score, metrics))

        # Verificar que todas las ejecuciones retornan métricas válidas
        for f1_score, metrics in results:
            self.assertIsInstance(f1_score, float)
            self.assertIsInstance(metrics, dict)
            self.assertGreaterEqual(f1_score, 0.0)
            self.assertLessEqual(f1_score, 1.0)

    def test_data_types_compatibility(self):
        """Verificar compatibilidad con diferentes tipos de datos."""
        # Verificar que acepta numpy arrays
        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertIsInstance(self.y_train, np.ndarray)

        # Verificar que los arrays tienen las dimensiones correctas
        self.assertEqual(
            self.X_train.shape[1:], (1, 65, 41)
        )  # (batch, channels, height, width)
        self.assertEqual(len(self.y_train), self.n_samples)

        # Verificar tipos de datos
        self.assertEqual(self.X_train.dtype, np.float32)
        self.assertEqual(self.y_train.dtype, np.int64)

    def test_error_handling_invalid_params(self):
        """Verificar que se manejan correctamente parámetros inválidos."""
        wrapper = CNN2DTalosWrapper()

        # Parámetros con valores inválidos
        invalid_params = {
            "batch_size": -1,  # Inválido
            "p_drop_conv": 1.5,  # Inválido (> 1)
            "p_drop_fc": -0.1,  # Inválido (< 0)
            "filters_1": 0,  # Inválido
            "filters_2": -1,  # Inválido
            "kernel_size_1": 0,  # Inválido
            "kernel_size_2": -1,  # Inválido
            "dense_units": 0,  # Inválido
            # learning_rate removido - se usa scheduler
        }

        # Debería fallar de manera controlada
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            wrapper.create_model(invalid_params)

    def test_results_saving_and_loading(self):
        """Verificar que se pueden guardar y cargar resultados correctamente."""
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Crear datos simulados
            mock_results = {
                "f1": [0.8, 0.85, 0.82],
                "accuracy": [0.8, 0.85, 0.82],
                "precision": [0.8, 0.85, 0.82],
                "recall": [0.8, 0.85, 0.82],
                "batch_size": [16, 32, 16],
                "p_drop_conv": [0.2, 0.5, 0.2],
                "filters_1": [32, 64, 32],
                "filters_2": [64, 128, 64],
                "kernel_size_1": [3, 5, 3],
                "kernel_size_2": [3, 5, 3],
                "dense_units": [32, 64, 32],
                # learning_rate removido - se usa scheduler
            }

            results_df = pd.DataFrame(mock_results)
            best_params = {"f1": 0.85, "batch_size": 32, "filters_1": 64}

            # Guardar CSV
            csv_path = temp_path / "test_results.csv"
            results_df.to_csv(csv_path, index=False)
            self.assertTrue(csv_path.exists())

            # Guardar JSON
            json_path = temp_path / "test_best_params.json"
            with open(json_path, "w") as f:
                json.dump(best_params, f, indent=2)
            self.assertTrue(json_path.exists())

            # Cargar y verificar CSV
            loaded_df = pd.read_csv(csv_path)
            self.assertEqual(len(loaded_df), len(results_df))
            self.assertTrue(loaded_df.equals(results_df))

            # Cargar y verificar JSON
            with open(json_path, "r") as f:
                loaded_params = json.load(f)
            self.assertEqual(loaded_params, best_params)

    def test_model_architecture_validation(self):
        """Verificar que las arquitecturas de modelo son válidas."""
        wrapper = CNN2DTalosWrapper()

        # Probar diferentes configuraciones de arquitectura
        test_configs = [
            {"filters_1": 32, "filters_2": 64, "dense_units": 32},
            {"filters_1": 64, "filters_2": 128, "dense_units": 64},
            {"filters_1": 128, "filters_2": 256, "dense_units": 128},
        ]

        for config in test_configs:
            params = {
                "batch_size": 16,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.5,
                "kernel_size_1": 3,
                "kernel_size_2": 3,
                **config,
            }

            # Crear modelo
            model = wrapper.create_model(params)
            self.assertIsInstance(model, CNN2D)

            # Verificar que el modelo tiene la arquitectura correcta
            self.assertEqual(model.filters_1, config["filters_1"])
            self.assertEqual(model.filters_2, config["filters_2"])
            self.assertEqual(model.dense_units, config["dense_units"])

            # Verificar forward pass
            test_input = torch.randn(2, 1, 65, 41)
            with torch.no_grad():
                output = model(test_input)
                self.assertEqual(output.shape, (2, 2))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_hyperparameter_combinations(self):
        """Verificar que se pueden probar diferentes combinaciones de hiperparámetros."""
        wrapper = CNN2DTalosWrapper()
        search_params = wrapper.get_search_params()

        # Calcular número total de combinaciones
        total_combinations = 1
        for param, values in search_params.items():
            # learning_rate fue removido - se usa scheduler
            total_combinations *= len(values)

        print(f"Total de combinaciones posibles: {total_combinations}")

        # Verificar que el número de combinaciones es razonable
        self.assertGreater(total_combinations, 0)
        self.assertLess(total_combinations, 10000)  # No demasiadas combinaciones

        # Probar algunas combinaciones aleatorias
        import random

        random.seed(42)

        for _ in range(5):  # Probar 5 combinaciones aleatorias
            params = {}
            for param, values in search_params.items():
                # learning_rate fue removido - se usa scheduler
                params[param] = random.choice(values)

            # learning_rate fue removido - se usa scheduler con lr fijo

            # Crear modelo con esta combinación
            model = wrapper.create_model(params)
            self.assertIsInstance(model, CNN2D)

            # Verificar forward pass
            test_input = torch.randn(1, 1, 65, 41)
            with torch.no_grad():
                output = model(test_input)
                self.assertEqual(output.shape, (1, 2))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())


if __name__ == "__main__":
    # Configurar el runner de pruebas
    unittest.main(verbosity=2)
