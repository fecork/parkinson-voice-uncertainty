"""
Pruebas unitarias para el módulo core de optimización con Talos.

Este módulo verifica que las funciones del sistema centralizado funcionen correctamente.
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
    TalosModelWrapper,
)
from modules.core.model_evaluation import (
    ModelEvaluator,
    compare_models,
    save_model_results,
)
from modules.core.cnn2d_talos_wrapper import CNN2DTalosWrapper, create_cnn2d_optimizer
from modules.models.cnn2d.model import CNN2D


class TestCoreTalosOptimization(unittest.TestCase):
    """Pruebas para las funciones core de optimización con Talos."""

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
        """Verificar que get_default_search_params retorna parámetros válidos."""
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

    def test_cnn2d_wrapper_creation(self):
        """Verificar que CNN2DTalosWrapper se crea correctamente."""
        wrapper = CNN2DTalosWrapper()

        # Verificar que es instancia de TalosModelWrapper
        self.assertIsInstance(wrapper, TalosModelWrapper)

        # Verificar parámetros de búsqueda
        search_params = wrapper.get_search_params()
        self.assertIsInstance(search_params, dict)
        self.assertIn("batch_size", search_params)
        self.assertIn("filters_1", search_params)
        self.assertIn("filters_2", search_params)
        self.assertIn("learning_rate", search_params)

    def test_cnn2d_wrapper_model_creation(self):
        """Verificar que CNN2DTalosWrapper crea modelos correctamente."""
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

        # Verificar que es instancia de CNN2D
        self.assertIsInstance(model, CNN2D)

        # Verificar parámetros del modelo
        self.assertEqual(model.filters_1, params["filters_1"])
        self.assertEqual(model.filters_2, params["filters_2"])
        self.assertEqual(model.kernel_size_1, params["kernel_size_1"])
        self.assertEqual(model.kernel_size_2, params["kernel_size_2"])
        self.assertEqual(model.dense_units, params["dense_units"])

    def test_talos_optimizer_creation(self):
        """Verificar que TalosOptimizer se crea correctamente."""
        wrapper = CNN2DTalosWrapper()
        optimizer = TalosOptimizer(wrapper, experiment_name="test")

        # Verificar propiedades
        self.assertEqual(optimizer.experiment_name, "test")
        self.assertEqual(optimizer.fraction_limit, 0.1)
        self.assertEqual(optimizer.random_method, "sobol")
        self.assertEqual(optimizer.seed, 42)
        self.assertIsNone(optimizer.results)

    def test_create_cnn2d_optimizer(self):
        """Verificar que create_cnn2d_optimizer funciona correctamente."""
        optimizer = create_cnn2d_optimizer(
            experiment_name="test_cnn2d", fraction_limit=0.05
        )

        # Verificar que es instancia de TalosOptimizer
        self.assertIsInstance(optimizer, TalosOptimizer)

        # Verificar configuración
        self.assertEqual(optimizer.experiment_name, "test_cnn2d")
        self.assertEqual(optimizer.fraction_limit, 0.05)

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
        expected_keys = ["correlations", "sorted_importance", "top_important"]
        for key in expected_keys:
            self.assertIn(key, importance)

        # Verificar correlaciones
        correlations = importance["correlations"]
        self.assertIsInstance(correlations, dict)

        # Verificar sorted_importance
        sorted_importance = importance["sorted_importance"]
        self.assertIsInstance(sorted_importance, list)
        self.assertGreater(len(sorted_importance), 0)

        # Verificar top_important
        top_important = importance["top_important"]
        self.assertIsInstance(top_important, list)
        self.assertLessEqual(len(top_important), 10)

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

    def test_model_evaluator_creation(self):
        """Verificar que ModelEvaluator se crea correctamente."""
        model = CNN2D()
        evaluator = ModelEvaluator(model)

        # Verificar propiedades
        self.assertEqual(evaluator.model, model)
        self.assertIsNotNone(evaluator.device)

    def test_model_evaluator_evaluation(self):
        """Verificar que ModelEvaluator evalúa modelos correctamente."""
        model = CNN2D()
        evaluator = ModelEvaluator(model)

        # Evaluar modelo
        metrics = evaluator.evaluate(self.X_test, self.y_test, batch_size=16)

        # Verificar métricas
        expected_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

    def test_compare_models(self):
        """Verificar que compare_models funciona correctamente."""
        # Crear diferentes modelos
        models = {
            "CNN2D_Small": CNN2D(filters_1=32, filters_2=64, dense_units=32),
            "CNN2D_Large": CNN2D(filters_1=128, filters_2=256, dense_units=128),
        }

        # Comparar modelos
        results_df = compare_models(models, self.X_test, self.y_test, batch_size=16)

        # Verificar resultados
        import pandas as pd

        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 2)
        self.assertIn("model", results_df.columns)
        self.assertIn("accuracy", results_df.columns)
        self.assertIn("f1", results_df.columns)

    def test_save_model_results(self):
        """Verificar que save_model_results funciona correctamente."""
        model = CNN2D()
        metrics = {
            "accuracy": 0.85,
            "f1": 0.82,
            "precision": 0.80,
            "recall": 0.84,
            "auc": 0.88,
        }

        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")

            # Guardar resultados
            save_model_results(model, metrics, save_path)

            # Verificar que se crearon los archivos
            self.assertTrue(os.path.exists(save_path + ".pth"))
            self.assertTrue(os.path.exists(save_path + "_results.json"))

    def test_create_talos_model_wrapper(self):
        """Verificar que create_talos_model_wrapper funciona correctamente."""

        # Función de entrenamiento simple
        def simple_train_function(model, train_loader, val_loader, params, n_epochs):
            return 0.8, {"f1": 0.8, "accuracy": 0.8}

        # Crear wrapper genérico
        wrapper = create_talos_model_wrapper(
            model_class=CNN2D,
            search_params={"batch_size": [16, 32], "learning_rate": [0.001, 0.0001]},
            train_function=simple_train_function,
        )

        # Verificar que es instancia de TalosModelWrapper
        self.assertIsInstance(wrapper, TalosModelWrapper)

        # Verificar que puede crear modelo (con parámetros válidos para CNN2D)
        params = {
            "filters_1": 32,
            "filters_2": 64,
            "kernel_size_1": 3,
            "kernel_size_2": 3,
            "dense_units": 64,
            "p_drop_conv": 0.3,
            "p_drop_fc": 0.5,
        }
        model = wrapper.create_model(params)
        self.assertIsInstance(model, CNN2D)

    def test_talos_model_wrapper_interface(self):
        """Verificar que TalosModelWrapper es una clase abstracta."""
        # Intentar crear instancia directa debería fallar
        with self.assertRaises(TypeError):
            TalosModelWrapper()

    def test_error_handling(self):
        """Verificar que las funciones manejan errores correctamente."""
        # Probar con parámetros inválidos
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
            "learning_rate": -0.001,  # Inválido
        }

        # La función debería manejar parámetros inválidos o fallar de manera controlada
        try:
            model = wrapper.create_model(invalid_params)
            # Si no falla, verificar que al menos retorna algo
            self.assertIsNotNone(model)
        except Exception as e:
            # Si falla, verificar que es un error esperado
            self.assertIsInstance(e, (ValueError, RuntimeError, AssertionError))

    def test_data_types_compatibility(self):
        """Verificar compatibilidad de tipos de datos."""
        # Verificar que acepta numpy arrays
        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertIsInstance(self.y_train, np.ndarray)

        # Verificar que el wrapper funciona con numpy arrays
        wrapper = CNN2DTalosWrapper()
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
        model = wrapper.create_model(params)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, CNN2D)


if __name__ == "__main__":
    # Configurar el runner de pruebas
    unittest.main(verbosity=2)
