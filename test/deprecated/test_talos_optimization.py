"""
Pruebas unitarias para el módulo de optimización con Talos.

Este módulo verifica que las funciones de Talos funcionen correctamente
y que la integración con CNN2D flexible sea exitosa.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

# Importar módulos a probar
from modules.core.talos_optimization import (
    TalosOptimizer,
    get_default_search_params,
    create_talos_model_wrapper,
    evaluate_best_model,
    analyze_hyperparameter_importance,
    print_optimization_summary,
)
from modules.core.cnn2d_talos_wrapper import (
    CNN2DTalosWrapper,
    create_cnn2d_optimizer,
    optimize_cnn2d,
)
from modules.models.cnn2d.model import CNN2D


class TestTalosOptimization(unittest.TestCase):
    """Pruebas para las funciones de optimización con Talos."""

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
        self.y_train = np.random.randint(0, 2, self.n_samples)
        self.X_val = np.random.randn(20, 1, *self.input_shape).astype(np.float32)
        self.y_val = np.random.randint(0, 2, 20)
        self.X_test = np.random.randn(20, 1, *self.input_shape).astype(np.float32)
        self.y_test = np.random.randint(0, 2, 20)

    def test_get_default_search_params(self):
        """Verificar que get_default_search_params retorna el diccionario correcto."""
        params = get_default_search_params()

        # Verificar que es un diccionario
        self.assertIsInstance(params, dict)

        # Verificar parámetros esperados
        expected_params = [
            "batch_size",
            "learning_rate",
            "dropout_rate",
            "weight_decay",
            "optimizer",
        ]

        for param in expected_params:
            self.assertIn(param, params)
            self.assertIsInstance(params[param], list)
            self.assertGreater(len(params[param]), 0)

        # Verificar valores específicos
        self.assertEqual(params["batch_size"], [16, 32, 64])
        self.assertEqual(params["learning_rate"], [0.001, 0.0001, 0.00001])
        self.assertEqual(params["dropout_rate"], [0.2, 0.3, 0.5])
        self.assertEqual(params["weight_decay"], [0.0, 1e-4, 1e-5])
        self.assertEqual(params["optimizer"], ["adam", "sgd"])

    def test_cnn2d_wrapper_basic(self):
        """Verificar que CNN2DTalosWrapper funciona con parámetros básicos."""
        # Crear wrapper
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
            "learning_rate": 0.001,
        }

        # Crear modelo
        model = wrapper.create_model(params)
        self.assertIsInstance(model, CNN2D)

        # Verificar parámetros de búsqueda
        search_params = wrapper.get_search_params()
        self.assertIsInstance(search_params, dict)
        self.assertIn("batch_size", search_params)
        self.assertIn("filters_1", search_params)

    def test_create_talos_model_different_params(self):
        """Verificar que create_talos_model funciona con diferentes parámetros."""
        # Parámetros alternativos
        params = {
            "batch_size": 32,
            "p_drop_conv": 0.2,
            "p_drop_fc": 0.3,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 5,
            "kernel_size_2": 7,
            "dense_units": 32,
            "learning_rate": 0.0001,
        }

        # Ejecutar función
        f1_score, metrics = create_talos_model(
            self.X_train, self.y_train, self.X_val, self.y_val, params
        )

        # Verificar que funciona con parámetros diferentes
        self.assertIsInstance(f1_score, float)
        self.assertIsInstance(metrics, dict)
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

    def test_create_talos_model_edge_cases(self):
        """Verificar que create_talos_model maneja casos edge."""
        # Parámetros extremos
        params = {
            "batch_size": 64,
            "p_drop_conv": 0.5,
            "p_drop_fc": 0.5,
            "filters_1": 128,
            "filters_2": 128,
            "kernel_size_1": 8,
            "kernel_size_2": 9,
            "dense_units": 16,
            "learning_rate": 0.0001,
        }

        # Ejecutar función
        f1_score, metrics = create_talos_model(
            self.X_train, self.y_train, self.X_val, self.y_val, params
        )

        # Verificar que maneja parámetros extremos
        self.assertIsInstance(f1_score, float)
        self.assertIsInstance(metrics, dict)

    def test_evaluate_best_model(self):
        """Verificar que evaluate_best_model funciona correctamente."""
        # Parámetros de prueba
        best_params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
            "learning_rate": 0.001,
        }

        # Crear directorio temporal para guardar modelo
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.pth")

            # Ejecutar función
            results = evaluate_best_model(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
                best_params,
                save_path=save_path,
            )

            # Verificar que retorna resultados válidos
            self.assertIsInstance(results, dict)

            # Verificar claves esperadas
            expected_keys = [
                "test_metrics",
                "train_history",
                "val_history",
                "final_epoch",
                "best_params",
            ]
            for key in expected_keys:
                self.assertIn(key, results)

            # Verificar métricas de test
            test_metrics = results["test_metrics"]
            expected_test_metrics = [
                "f1",
                "accuracy",
                "precision",
                "recall",
                "auc",
                "loss",
            ]
            for metric in expected_test_metrics:
                self.assertIn(metric, test_metrics)
                self.assertIsInstance(test_metrics[metric], (int, float))

            # Verificar que el archivo se guardó
            self.assertTrue(os.path.exists(save_path))

            # Verificar que se creó el archivo de configuración
            config_path = save_path.replace(".pth", "_config.json")
            self.assertTrue(os.path.exists(config_path))

    def test_analyze_hyperparameter_importance(self):
        """Verificar que analyze_hyperparameter_importance funciona correctamente."""
        # Crear DataFrame simulado con resultados
        import pandas as pd

        # Datos simulados
        data = {
            "f1": [0.8, 0.85, 0.82, 0.87, 0.83],
            "accuracy": [0.8, 0.85, 0.82, 0.87, 0.83],
            "precision": [0.8, 0.85, 0.82, 0.87, 0.83],
            "recall": [0.8, 0.85, 0.82, 0.87, 0.83],
            "batch_size": [16, 32, 16, 64, 32],
            "p_drop_conv": [0.2, 0.5, 0.2, 0.5, 0.3],
            "filters_1": [32, 64, 32, 128, 64],
            "filters_2": [64, 128, 64, 128, 128],
            "kernel_size_1": [3, 5, 3, 7, 5],
            "kernel_size_2": [3, 5, 3, 7, 5],
            "dense_units": [32, 64, 32, 128, 64],
            "learning_rate": [0.001, 0.0001, 0.001, 0.0001, 0.001],
        }

        results_df = pd.DataFrame(data)

        # Ejecutar análisis
        importance = analyze_hyperparameter_importance(results_df)

        # Verificar que retorna diccionario con claves esperadas
        self.assertIsInstance(importance, dict)
        expected_keys = ["correlations", "sorted_importance", "top_5_important"]
        for key in expected_keys:
            self.assertIn(key, importance)

        # Verificar correlaciones
        correlations = importance["correlations"]
        self.assertIsInstance(correlations, dict)

        # Verificar sorted_importance
        sorted_importance = importance["sorted_importance"]
        self.assertIsInstance(sorted_importance, list)
        self.assertGreater(len(sorted_importance), 0)

        # Verificar top_5_important
        top_5 = importance["top_5_important"]
        self.assertIsInstance(top_5, list)
        self.assertLessEqual(len(top_5), 5)

    def test_print_optimization_summary(self):
        """Verificar que print_optimization_summary no falla."""
        # Crear DataFrame simulado
        import pandas as pd

        data = {
            "f1": [0.8, 0.85, 0.82, 0.87, 0.83],
            "accuracy": [0.8, 0.85, 0.82, 0.87, 0.83],
            "precision": [0.8, 0.85, 0.82, 0.87, 0.83],
            "recall": [0.8, 0.85, 0.82, 0.87, 0.83],
            "batch_size": [16, 32, 16, 64, 32],
            "p_drop_conv": [0.2, 0.5, 0.2, 0.5, 0.3],
            "filters_1": [32, 64, 32, 128, 64],
            "filters_2": [64, 128, 64, 128, 128],
            "kernel_size_1": [3, 5, 3, 7, 5],
            "kernel_size_2": [3, 5, 3, 7, 5],
            "dense_units": [32, 64, 32, 128, 64],
            "learning_rate": [0.001, 0.0001, 0.001, 0.0001, 0.001],
        }

        results_df = pd.DataFrame(data)

        # Verificar que la función no falla (captura output)
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            print_optimization_summary(results_df, top_n=3)
            # Si llegamos aquí, la función no falló
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"print_optimization_summary falló con error: {e}")
        finally:
            sys.stdout = sys.__stdout__

    def test_model_creation_with_talos_params(self):
        """Verificar que se puede crear un modelo CNN2D con parámetros de Talos."""
        # Parámetros típicos de Talos
        talos_params = {
            "batch_size": 32,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 64,
            "filters_2": 128,
            "kernel_size_1": 5,
            "kernel_size_2": 7,
            "dense_units": 32,
            "learning_rate": 0.001,
        }

        # Crear modelo con parámetros de Talos
        model = CNN2D(
            n_classes=2,
            p_drop_conv=talos_params["p_drop_conv"],
            p_drop_fc=talos_params["p_drop_fc"],
            input_shape=(65, 41),
            filters_1=talos_params["filters_1"],
            filters_2=talos_params["filters_2"],
            kernel_size_1=talos_params["kernel_size_1"],
            kernel_size_2=talos_params["kernel_size_2"],
            dense_units=talos_params["dense_units"],
        )

        # Verificar que el modelo se creó correctamente
        self.assertIsInstance(model, CNN2D)

        # Verificar parámetros
        self.assertEqual(model.filters_1, talos_params["filters_1"])
        self.assertEqual(model.filters_2, talos_params["filters_2"])
        self.assertEqual(model.kernel_size_1, talos_params["kernel_size_1"])
        self.assertEqual(model.kernel_size_2, talos_params["kernel_size_2"])
        self.assertEqual(model.dense_units, talos_params["dense_units"])

        # Verificar forward pass
        test_input = torch.randn(2, 1, 65, 41)
        with torch.no_grad():
            output = model(test_input)
            self.assertEqual(output.shape, (2, 2))
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_talos_integration_compatibility(self):
        """Verificar que las funciones son compatibles con Talos."""
        # Verificar que create_talos_model retorna el formato esperado por Talos
        params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
            "learning_rate": 0.001,
        }

        # Ejecutar función
        result = create_talos_model(
            self.X_train, self.y_train, self.X_val, self.y_val, params
        )

        # Verificar que retorna tupla (f1_score, metrics_dict)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        f1_score, metrics = result

        # Verificar tipos
        self.assertIsInstance(f1_score, float)
        self.assertIsInstance(metrics, dict)

        # Verificar que f1_score es un valor válido para Talos
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

        # Verificar que no hay NaN o Inf
        self.assertFalse(np.isnan(f1_score))
        self.assertFalse(np.isinf(f1_score))

    def test_error_handling(self):
        """Verificar que las funciones manejan errores correctamente."""
        # Probar con parámetros inválidos
        invalid_params = {
            "batch_size": -1,  # Inválido
            "p_drop_conv": 1.5,  # Inválido (> 1)
            "p_drop_fc": -0.1,  # Inválido (< 0)
            "filters_1": 0,  # Inválido
            "filters_2": -1,  # Inválido
            "kernel_size_1": 0,  # Inválido
            "kernel_size_2": -1,  # Inválido
            "dense_units": 0,  # Inválido
            "learning_rate": -0.001,  # Inválido
        }

        # La función debería manejar parámetros inválidos o fallar de manera controlada
        try:
            result = create_talos_model(
                self.X_train, self.y_train, self.X_val, self.y_val, invalid_params
            )
            # Si no falla, verificar que al menos retorna algo
            self.assertIsNotNone(result)
        except Exception as e:
            # Si falla, verificar que es un error esperado
            self.assertIsInstance(e, (ValueError, RuntimeError, AssertionError))

    def test_data_types_compatibility(self):
        """Verificar compatibilidad de tipos de datos."""
        # Verificar que acepta numpy arrays
        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertIsInstance(self.y_train, np.ndarray)

        # Verificar que acepta diferentes tipos de datos
        params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
            "learning_rate": 0.001,
        }

        # Debería funcionar con numpy arrays
        result = create_talos_model(
            self.X_train, self.y_train, self.X_val, self.y_val, params
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)


class TestTalosIntegration(unittest.TestCase):
    """Pruebas de integración con Talos."""

    def setUp(self):
        """Configuración inicial."""
        self.n_samples = 50
        self.input_shape = (65, 41)

        # Datos sintéticos más pequeños para pruebas rápidas
        np.random.seed(42)
        self.X_train = np.random.randn(self.n_samples, 1, *self.input_shape).astype(
            np.float32
        )
        self.y_train = np.random.randint(0, 2, self.n_samples)
        self.X_val = np.random.randn(10, 1, *self.input_shape).astype(np.float32)
        self.y_val = np.random.randint(0, 2, 10)

    @patch("talos.Scan")
    def test_talos_scan_integration(self, mock_scan):
        """Verificar integración con Talos Scan."""
        # Configurar mock
        mock_scan_instance = MagicMock()
        mock_scan_instance.data = MagicMock()
        mock_scan.return_value = mock_scan_instance

        # Simular resultados de Talos
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
            "learning_rate": [0.001, 0.0001, 0.001],
        }

        import pandas as pd

        mock_scan_instance.data = pd.DataFrame(mock_results)

        # Verificar que se puede usar con Talos
        from modules.models.cnn2d.talos_optimization import get_search_params

        params = get_search_params()
        self.assertIsInstance(params, dict)
        self.assertIn("batch_size", params)
        self.assertIn("filters_1", params)

    def test_talos_model_function_signature(self):
        """Verificar que la función de modelo tiene la firma correcta para Talos."""
        from modules.models.cnn2d.talos_optimization import create_talos_model

        # Verificar que la función acepta los parámetros correctos
        import inspect

        sig = inspect.signature(create_talos_model)
        params = list(sig.parameters.keys())

        # Debería tener: x_train, y_train, x_val, y_val, params
        expected_params = ["x_train", "y_train", "x_val", "y_val", "params"]
        for param in expected_params:
            self.assertIn(param, params)

    def test_talos_output_format(self):
        """Verificar que el output es compatible con Talos."""
        params = {
            "batch_size": 16,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
            "learning_rate": 0.001,
        }

        result = create_talos_model(
            self.X_train, self.y_train, self.X_val, self.y_val, params
        )

        # Talos espera que la función retorne (métrica_principal, métricas_dict)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        f1_score, metrics = result

        # La métrica principal debe ser un float
        self.assertIsInstance(f1_score, float)
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

        # Las métricas adicionales deben ser un dict
        self.assertIsInstance(metrics, dict)


if __name__ == "__main__":
    # Configurar el runner de pruebas
    unittest.main(verbosity=2)
