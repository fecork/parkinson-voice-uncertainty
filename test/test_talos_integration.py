"""
Pruebas de integración para Talos - Verificar que funciona correctamente
y genera métricas válidas.

Este módulo verifica que la integración completa con Talos funciona
y que se pueden guardar y cargar resultados correctamente.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.core.dataset import DictDataset
import tempfile
import os
import json
from pathlib import Path
import pandas as pd

# Importar módulos a probar
from modules.core.cnn2d_talos_wrapper import (
    CNN2DTalosWrapper,
    create_cnn2d_optimizer,
    optimize_cnn2d,
)
from modules.core.talos_optimization import TalosOptimizer
from modules.models.cnn2d.model import CNN2D


class TestTalosIntegration(unittest.TestCase):
    """Pruebas de integración para verificar que Talos funciona correctamente."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos sintéticos para pruebas
        np.random.seed(42)
        torch.manual_seed(42)

        self.n_samples = 200
        self.input_shape = (65, 41)

        # Datos sintéticos más realistas
        self.X_train = np.random.randn(self.n_samples, 1, *self.input_shape).astype(
            np.float32
        )
        self.y_train = np.random.randint(0, 2, self.n_samples).astype(np.int64)
        self.X_val = np.random.randn(50, 1, *self.input_shape).astype(np.float32)
        self.y_val = np.random.randint(0, 2, 50)
        self.X_test = np.random.randn(30, 1, *self.input_shape).astype(np.float32)
        self.y_test = np.random.randint(0, 2, 30)

    def test_cnn2d_wrapper_training(self):
        """Verificar que CNN2DTalosWrapper puede entrenar modelos correctamente."""
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
        self.assertIsInstance(model, CNN2D)

        # Crear DataLoaders
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
            model, train_loader, val_loader, params, n_epochs=3
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

    def test_talos_optimizer_creation(self):
        """Verificar que TalosOptimizer se crea correctamente."""
        wrapper = CNN2DTalosWrapper()
        optimizer = TalosOptimizer(
            model_wrapper=wrapper,
            experiment_name="test_integration",
            fraction_limit=0.1,
            search_method="random",
            seed=42,
        )

        # Verificar propiedades
        self.assertEqual(optimizer.experiment_name, "test_integration")
        self.assertEqual(optimizer.fraction_limit, 0.1)
        self.assertEqual(optimizer.search_method, "random")
        self.assertEqual(optimizer.seed, 42)
        self.assertIsNone(optimizer.results)

    def test_create_cnn2d_optimizer(self):
        """Verificar que create_cnn2d_optimizer funciona correctamente."""
        optimizer = create_cnn2d_optimizer(
            experiment_name="test_cnn2d", fraction_limit=0.1, search_method="random"
        )

        # Verificar que es instancia de TalosOptimizer
        self.assertIsInstance(optimizer, TalosOptimizer)

        # Verificar configuración
        self.assertEqual(optimizer.experiment_name, "test_cnn2d")
        self.assertEqual(optimizer.fraction_limit, 0.1)
        self.assertEqual(optimizer.search_method, "random")

    def test_optimize_cnn2d_basic(self):
        """Verificar que optimize_cnn2d funciona con configuración básica."""
        # Usar configuración muy pequeña para prueba rápida
        results = optimize_cnn2d(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            experiment_name="test_optimization",
            fraction_limit=0.05,  # Solo 5% para prueba rápida
            search_method="random",
            n_epochs=2,  # Solo 2 épocas para prueba rápida
        )

        # Verificar que retorna diccionario con claves esperadas
        self.assertIsInstance(results, dict)
        expected_keys = ["results_df", "best_params", "analysis", "optimizer"]
        for key in expected_keys:
            self.assertIn(key, results)

        # Verificar results_df
        results_df = results["results_df"]
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 0)

        # Verificar que tiene columnas esperadas
        expected_columns = ["f1", "accuracy", "precision", "recall"]
        for col in expected_columns:
            self.assertIn(col, results_df.columns)

        # Verificar best_params
        best_params = results["best_params"]
        self.assertIsInstance(best_params, dict)
        self.assertIn("f1", best_params)
        self.assertIsInstance(best_params["f1"], (int, float))

        # Verificar analysis
        analysis = results["analysis"]
        self.assertIsInstance(analysis, dict)

    def test_optimize_cnn2d_with_round_limit(self):
        """Verificar que optimize_cnn2d funciona con round_limit."""
        # Usar round_limit para controlar exactamente cuántas configuraciones probar
        results = optimize_cnn2d(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            experiment_name="test_round_limit",
            round_limit=3,  # Exactamente 3 configuraciones
            n_epochs=2,
        )

        # Verificar que se evaluaron exactamente 3 configuraciones
        results_df = results["results_df"]
        self.assertEqual(len(results_df), 3)

        # Verificar que todas las configuraciones tienen métricas válidas
        for _, row in results_df.iterrows():
            self.assertGreaterEqual(row["f1"], 0.0)
            self.assertLessEqual(row["f1"], 1.0)
            self.assertGreaterEqual(row["accuracy"], 0.0)
            self.assertLessEqual(row["accuracy"], 1.0)

    def test_results_saving_and_loading(self):
        """Verificar que se pueden guardar y cargar resultados correctamente."""
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Ejecutar optimización pequeña
            results = optimize_cnn2d(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                experiment_name="test_save_load",
                round_limit=2,
                n_epochs=1,
            )

            # Guardar resultados
            results_df = results["results_df"]
            best_params = results["best_params"]

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
        # learning_rate fue removido - ahora se usa scheduler

    def test_model_creation_with_all_params(self):
        """Verificar que se pueden crear modelos con todos los parámetros del espacio de búsqueda."""
        wrapper = CNN2DTalosWrapper()
        search_params = wrapper.get_search_params()

        # Probar diferentes combinaciones de parámetros
        test_combinations = [
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

        for params in test_combinations:
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

    def test_metrics_consistency(self):
        """Verificar que las métricas son consistentes entre diferentes ejecuciones."""
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

        # Crear modelo
        model = wrapper.create_model(params)

        # Crear DataLoaders
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

        # Ejecutar entrenamiento múltiples veces con la misma configuración
        results = []
        for i in range(3):
            # Crear nuevo modelo para cada ejecución
            model_copy = wrapper.create_model(params)

            f1_score, metrics = wrapper.train_model(
                model_copy, train_loader, val_loader, params, n_epochs=2
            )

            results.append((f1_score, metrics))

        # Verificar que todas las ejecuciones retornan métricas válidas
        for f1_score, metrics in results:
            self.assertIsInstance(f1_score, float)
            self.assertIsInstance(metrics, dict)
            self.assertGreaterEqual(f1_score, 0.0)
            self.assertLessEqual(f1_score, 1.0)

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

    def test_optimization_completeness(self):
        """Verificar que la optimización completa funciona end-to-end."""
        # Ejecutar optimización completa pero pequeña
        results = optimize_cnn2d(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            experiment_name="test_complete",
            round_limit=2,
            n_epochs=2,
        )

        # Verificar estructura completa de resultados
        self.assertIn("results_df", results)
        self.assertIn("best_params", results)
        self.assertIn("analysis", results)
        self.assertIn("optimizer", results)

        # Verificar que se encontraron mejores parámetros
        best_params = results["best_params"]
        self.assertIn("f1", best_params)
        self.assertGreater(best_params["f1"], 0.0)

        # Verificar que el análisis funciona
        analysis = results["analysis"]
        self.assertIsInstance(analysis, dict)
        self.assertIn("correlations", analysis)
        self.assertIn("sorted_importance", analysis)
        self.assertIn("top_important", analysis)


if __name__ == "__main__":
    # Configurar el runner de pruebas
    unittest.main(verbosity=2)
