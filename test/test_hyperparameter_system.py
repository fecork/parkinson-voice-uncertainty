#!/usr/bin/env python3
"""
Pruebas para el sistema de configuración de hiperparámetros.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# Agregar módulos al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.hyperparameter_config import (
    HyperparameterManager,
    IbarraHyperparameters,
    get_hyperparameters,
    compare_hyperparameters,
)


class TestHyperparameterSystem(unittest.TestCase):
    """Pruebas para el sistema de hiperparámetros."""

    def setUp(self):
        """Configuración inicial."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_config_path = Path(self.temp_dir) / "test_config.json"
        self.manager = HyperparameterManager(self.temp_config_path)

    def test_ibarra_hyperparameters(self):
        """Probar que los hiperparámetros de Ibarra son correctos."""
        ibarra_params = self.manager.get_ibarra_hyperparameters()

        # Verificar valores exactos del paper
        self.assertEqual(ibarra_params["kernel_size_1"], 6)
        self.assertEqual(ibarra_params["kernel_size_2"], 9)
        self.assertEqual(ibarra_params["filters_2"], 64)  # depth_CL
        self.assertEqual(ibarra_params["dense_units"], 32)  # neurons_MLP
        self.assertEqual(ibarra_params["p_drop_conv"], 0.2)  # drop_out
        self.assertEqual(ibarra_params["batch_size"], 64)
        self.assertEqual(ibarra_params["learning_rate"], 0.1)
        self.assertEqual(ibarra_params["momentum"], 0.9)
        self.assertEqual(ibarra_params["source"], "ibarra_2023_paper")

    def test_config_save_load(self):
        """Probar guardar y cargar configuración."""
        # Guardar configuración
        self.manager.save_config(use_ibarra=True)

        # Verificar que el archivo se creó
        self.assertTrue(self.temp_config_path.exists())

        # Cargar configuración
        config = self.manager.load_config()
        self.assertTrue(config["use_ibarra_hyperparameters"])

    def test_ibarra_overrides_from_config(self):
        """Los overrides en config deben reflejarse en get_ibarra_hyperparameters."""
        override_config = {
            "use_ibarra_hyperparameters": True,
            "ibarra_hyperparameters": {
                "filters_1": 64,
                "kernel_size_1": 6,
                "kernel_size_2": 9,
            },
        }

        with open(self.temp_config_path, "w") as f:
            json.dump(override_config, f)

        params = self.manager.get_ibarra_hyperparameters()
        self.assertEqual(params["filters_1"], 64)

    def test_hyperparameter_selection(self):
        """Probar selección de hiperparámetros."""
        # Probar con Ibarra
        ibarra_params = self.manager.get_hyperparameters(use_ibarra=True)
        self.assertEqual(ibarra_params["source"], "ibarra_2023_paper")

        # Probar con Optuna (debería fallar y usar Ibarra como fallback)
        optuna_params = self.manager.get_hyperparameters(use_ibarra=False)
        # Como no hay archivo de Optuna, debería usar Ibarra como fallback
        self.assertIn("source", optuna_params)

    def test_ibarra_dataclass(self):
        """Probar la clase IbarraHyperparameters."""
        ibarra = IbarraHyperparameters()

        # Verificar valores por defecto
        self.assertEqual(ibarra.kernel_size_1, 6)
        self.assertEqual(ibarra.kernel_size_2, 9)
        self.assertEqual(ibarra.depth_CL, 64)
        self.assertEqual(ibarra.neurons_MLP, 32)
        self.assertEqual(ibarra.drop_out, 0.2)
        self.assertEqual(ibarra.batch_size, 64)

        # Probar conversión a diccionario
        params_dict = ibarra.to_dict()
        self.assertIn("kernel_size_1", params_dict)
        self.assertIn("filters_2", params_dict)  # Mapeado correcto
        self.assertEqual(params_dict["kernel_size_1"], 6)

    def test_parameter_completion(self):
        """Probar completado de parámetros de Optuna."""
        # Parámetros incompletos
        incomplete_params = {
            "kernel_size_1": 4,
            "filters_2": 32,
            "learning_rate": 0.001,
        }

        # Completar parámetros
        complete_params = self.manager._complete_optuna_params(incomplete_params)

        # Verificar que se completaron los parámetros faltantes
        self.assertIn("kernel_size_2", complete_params)
        self.assertIn("dense_units", complete_params)
        self.assertIn("batch_size", complete_params)
        self.assertIn("optimizer", complete_params)

        # Verificar que se mantuvieron los valores originales
        self.assertEqual(complete_params["kernel_size_1"], 4)
        self.assertEqual(complete_params["filters_2"], 32)
        self.assertEqual(complete_params["learning_rate"], 0.001)

    def test_convenience_function(self):
        """Probar función de conveniencia."""
        # Probar con Ibarra
        ibarra_params = get_hyperparameters(use_ibarra=True)
        self.assertEqual(ibarra_params["source"], "ibarra_2023_paper")

        # Probar con Optuna
        optuna_params = get_hyperparameters(use_ibarra=False)
        self.assertIn("source", optuna_params)


class TestIbarraHyperparameters(unittest.TestCase):
    """Pruebas específicas para la clase IbarraHyperparameters."""

    def test_custom_values(self):
        """Probar valores personalizados."""
        ibarra = IbarraHyperparameters(
            kernel_size_1=8, depth_CL=128, neurons_MLP=64, batch_size=32
        )

        self.assertEqual(ibarra.kernel_size_1, 8)
        self.assertEqual(ibarra.depth_CL, 128)
        self.assertEqual(ibarra.neurons_MLP, 64)
        self.assertEqual(ibarra.batch_size, 32)

        # Valores por defecto no especificados
        self.assertEqual(ibarra.kernel_size_2, 9)
        self.assertEqual(ibarra.drop_out, 0.2)

    def test_dict_conversion(self):
        """Probar conversión a diccionario."""
        ibarra = IbarraHyperparameters()
        params_dict = ibarra.to_dict()

        # Verificar mapeo correcto
        self.assertEqual(params_dict["kernel_size_1"], ibarra.kernel_size_1)
        self.assertEqual(params_dict["kernel_size_2"], ibarra.kernel_size_2)
        self.assertEqual(params_dict["filters_2"], ibarra.depth_CL)
        self.assertEqual(params_dict["dense_units"], ibarra.neurons_MLP)
        self.assertEqual(params_dict["p_drop_conv"], ibarra.drop_out)
        self.assertEqual(params_dict["batch_size"], ibarra.batch_size)

        # Verificar que se incluyen todos los parámetros necesarios
        required_keys = [
            "kernel_size_1",
            "kernel_size_2",
            "filters_1",
            "filters_2",
            "dense_units",
            "p_drop_conv",
            "p_drop_fc",
            "batch_size",
            "learning_rate",
            "momentum",
            "weight_decay",
            "n_epochs",
            "early_stopping_patience",
            "step_size",
            "gamma",
            "optimizer",
            "source",
        ]

        for key in required_keys:
            self.assertIn(key, params_dict)


def run_comparison_test():
    """Ejecutar prueba de comparación."""
    print("=" * 80)
    print("PRUEBA DE COMPARACIÓN DE HIPERPARÁMETROS")
    print("=" * 80)

    try:
        compare_hyperparameters()
        print("✅ Comparación ejecutada exitosamente")
    except Exception as e:
        print(f"❌ Error en comparación: {e}")


if __name__ == "__main__":
    print("🧪 EJECUTANDO PRUEBAS DEL SISTEMA DE HIPERPARÁMETROS")
    print("=" * 60)

    # Ejecutar pruebas unitarias
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Ejecutar prueba de comparación
    print("\n" + "=" * 60)
    run_comparison_test()

    print("\n🎉 TODAS LAS PRUEBAS COMPLETADAS")
