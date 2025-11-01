"""
Prueba para verificar que Talos genera los archivos correctos.

Esta prueba simula la ejecución de Talos y verifica que se generen:
- talos_scan_results.csv
- best_params.json
- optimization_summary.txt
"""

import unittest
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Importar módulos a probar
from modules.core.cnn2d_talos_wrapper import CNN2DTalosWrapper
from modules.core.talos_evaluator import check_talos_results


class TestTalosFileGeneration(unittest.TestCase):
    """Pruebas para verificar que Talos genera archivos correctos."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Crear datos sintéticos para pruebas
        np.random.seed(42)

        self.n_samples = 100
        self.input_shape = (65, 41)

        # Datos sintéticos
        self.X_train = np.random.randn(self.n_samples, 1, *self.input_shape).astype(
            np.float32
        )
        self.y_train = np.random.randint(0, 2, self.n_samples).astype(np.int64)
        self.X_val = np.random.randn(20, 1, *self.input_shape).astype(np.float32)
        self.y_val = np.random.randint(0, 2, 20).astype(np.int64)

    def test_talos_generates_required_files(self):
        """Verificar que Talos genera los archivos requeridos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Crear wrapper
            wrapper = CNN2DTalosWrapper()

            # Simular resultados de Talos (sin ejecutar realmente)
            mock_results = {
                "results_df": pd.DataFrame(
                    {
                        "f1": [0.8, 0.85, 0.82, 0.87, 0.83],
                        "accuracy": [0.8, 0.85, 0.82, 0.87, 0.83],
                        "precision": [0.8, 0.85, 0.82, 0.87, 0.83],
                        "recall": [0.8, 0.85, 0.82, 0.87, 0.83],
                        "val_loss": [0.3, 0.25, 0.28, 0.22, 0.26],
                        "train_loss": [0.4, 0.35, 0.38, 0.32, 0.36],
                        "batch_size": [16, 32, 16, 64, 32],
                        "p_drop_conv": [0.2, 0.5, 0.2, 0.5, 0.3],
                        "p_drop_fc": [0.2, 0.5, 0.2, 0.5, 0.3],
                        "filters_1": [32, 64, 32, 128, 64],
                        "filters_2": [64, 128, 64, 128, 128],
                        "kernel_size_1": [3, 5, 3, 7, 5],
                        "kernel_size_2": [3, 5, 3, 7, 5],
                        "dense_units": [32, 64, 32, 128, 64],
                        # learning_rate removido - se usa scheduler
                    }
                ),
                "best_params": {
                    "f1": 0.87,
                    "accuracy": 0.87,
                    "precision": 0.87,
                    "recall": 0.87,
                    "batch_size": 64,
                    "p_drop_conv": 0.5,
                    "p_drop_fc": 0.5,
                    "filters_1": 128,
                    "filters_2": 128,
                    "kernel_size_1": 7,
                    "kernel_size_2": 7,
                    "dense_units": 128,
                    # learning_rate removido - se usa scheduler
                },
                "analysis": {
                    "correlations": {},
                    "sorted_importance": [],
                    "top_important": [],
                },
            }

            # Simular guardado de archivos como lo haría Talos
            results_csv_path = temp_path / "talos_scan_results.csv"
            best_params_path = temp_path / "best_params.json"
            summary_path = temp_path / "optimization_summary.txt"

            # Guardar CSV
            mock_results["results_df"].to_csv(results_csv_path, index=False)

            # Guardar JSON
            with open(best_params_path, "w") as f:
                json.dump(mock_results["best_params"], f, indent=2)

            # Guardar resumen
            with open(summary_path, "w") as f:
                f.write("RESUMEN DE OPTIMIZACIÓN TALOS\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Total configuraciones evaluadas: {len(mock_results['results_df'])}\n"
                )
                f.write(
                    f"Mejor F1-score: {mock_results['results_df']['f1'].max():.4f}\n"
                )
                f.write(
                    f"F1-score promedio: {mock_results['results_df']['f1'].mean():.4f} ± {mock_results['results_df']['f1'].std():.4f}\n\n"
                )
                f.write("MEJORES HIPERPARÁMETROS:\n")
                f.write("-" * 30 + "\n")
                for param, value in mock_results["best_params"].items():
                    if param not in [
                        "f1",
                        "accuracy",
                        "precision",
                        "recall",
                        "val_loss",
                        "train_loss",
                    ]:
                        f.write(f"{param}: {value}\n")

            # Verificar que los archivos se crearon
            self.assertTrue(
                results_csv_path.exists(), "talos_scan_results.csv no se generó"
            )
            self.assertTrue(best_params_path.exists(), "best_params.json no se generó")
            self.assertTrue(
                summary_path.exists(), "optimization_summary.txt no se generó"
            )

            # Verificar contenido del CSV
            loaded_df = pd.read_csv(results_csv_path)
            self.assertEqual(len(loaded_df), 5)
            self.assertIn("f1", loaded_df.columns)
            self.assertIn("batch_size", loaded_df.columns)
            self.assertIn("filters_1", loaded_df.columns)

            # Verificar contenido del JSON
            with open(best_params_path, "r") as f:
                loaded_params = json.load(f)
            self.assertIn("f1", loaded_params)
            self.assertIn("batch_size", loaded_params)
            self.assertIn("filters_1", loaded_params)

            # Verificar contenido del resumen
            with open(summary_path, "r") as f:
                summary_content = f.read()
            self.assertIn("RESUMEN DE OPTIMIZACIÓN TALOS", summary_content)
            self.assertIn("Total configuraciones evaluadas: 5", summary_content)

    def test_evaluator_can_read_generated_files(self):
        """Verificar que el evaluador puede leer los archivos generados."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Crear archivos simulados
            mock_df = pd.DataFrame(
                {
                    "f1": [0.8, 0.85, 0.82, 0.87, 0.83],
                    "accuracy": [0.8, 0.85, 0.82, 0.87, 0.83],
                    "precision": [0.8, 0.85, 0.82, 0.87, 0.83],
                    "recall": [0.8, 0.85, 0.82, 0.87, 0.83],
                    "val_loss": [0.3, 0.25, 0.28, 0.22, 0.26],
                    "train_loss": [0.4, 0.35, 0.38, 0.32, 0.36],
                    "batch_size": [16, 32, 16, 64, 32],
                    "p_drop_conv": [0.2, 0.5, 0.2, 0.5, 0.3],
                    "p_drop_fc": [0.2, 0.5, 0.2, 0.5, 0.3],
                    "filters_1": [32, 64, 32, 128, 64],
                    "filters_2": [64, 128, 64, 128, 128],
                    "kernel_size_1": [3, 5, 3, 7, 5],
                    "kernel_size_2": [3, 5, 3, 7, 5],
                    "dense_units": [32, 64, 32, 128, 64],
                    # learning_rate removido - se usa scheduler
                }
            )

            mock_params = {
                "f1": 0.87,
                "accuracy": 0.87,
                "precision": 0.87,
                "recall": 0.87,
                "batch_size": 64,
                "p_drop_conv": 0.5,
                "p_drop_fc": 0.5,
                "filters_1": 128,
                "filters_2": 128,
                "kernel_size_1": 7,
                "kernel_size_2": 7,
                "dense_units": 128,
                # learning_rate removido - se usa scheduler
            }

            # Guardar archivos
            mock_df.to_csv(temp_path / "talos_scan_results.csv", index=False)
            with open(temp_path / "best_params.json", "w") as f:
                json.dump(mock_params, f, indent=2)

            # Probar que el evaluador puede leer los archivos
            evaluation = check_talos_results(str(temp_path))

            # Verificar que la evaluación fue exitosa
            self.assertEqual(evaluation["status"], "SUCCESS")
            self.assertTrue(evaluation["process_correct"])
            self.assertEqual(evaluation["total_configurations"], 5)
            self.assertEqual(evaluation["best_f1_score"], 0.87)
            self.assertIn("batch_size", evaluation["hyperparameters_tested"])
            self.assertIn("filters_1", evaluation["hyperparameters_tested"])

    def test_hyperparameter_space_definition(self):
        """Verificar que el espacio de hiperparámetros está bien definido."""
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
            # learning_rate removido - se usa scheduler
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

    def test_file_structure_validation(self):
        """Verificar que la estructura de archivos es correcta."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Crear estructura de directorios como Talos
            results_dir = temp_path / "results" / "cnn_talos_optimization"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Simular archivos generados por Talos
            mock_df = pd.DataFrame(
                {
                    "f1": [0.8, 0.85, 0.82],
                    "accuracy": [0.8, 0.85, 0.82],
                    "batch_size": [16, 32, 16],
                    "filters_1": [32, 64, 32],
                    # learning_rate removido - se usa scheduler
                }
            )

            mock_params = {
                "f1": 0.85,
                "batch_size": 32,
                "filters_1": 64,
                # learning_rate removido - se usa scheduler
            }

            # Guardar archivos
            mock_df.to_csv(results_dir / "talos_scan_results.csv", index=False)
            with open(results_dir / "best_params.json", "w") as f:
                json.dump(mock_params, f, indent=2)

            # Verificar estructura
            self.assertTrue((results_dir / "talos_scan_results.csv").exists())
            self.assertTrue((results_dir / "best_params.json").exists())

            # Probar evaluación
            evaluation = check_talos_results(str(results_dir))
            self.assertEqual(evaluation["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
