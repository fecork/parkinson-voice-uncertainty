"""
Prueba que actualiza archivos reales para verificar que el sistema funciona.
"""

import unittest
import json
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Importar módulos a probar
from modules.core.talos_evaluator import check_talos_results


class TestTalosRealFiles(unittest.TestCase):
    """Pruebas que actualizan archivos reales."""

    def setUp(self):
        """Configuración inicial."""
        self.real_results_dir = Path("results/cnn_talos_optimization")
        self.test_results_dir = Path("results/cnn_talos_optimization_test")

        # Crear directorio de test
        self.test_results_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Limpiar archivos de test."""
        if self.test_results_dir.exists():
            shutil.rmtree(self.test_results_dir)

    def test_update_test_files(self):
        """Crear archivos de test con nuevos datos."""
        # Crear nuevos datos
        new_data = {
            "results_df": pd.DataFrame(
                {
                    "f1": [0.92, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77],
                    "accuracy": [0.92, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77],
                    "precision": [0.91, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76],
                    "recall": [0.93, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78],
                    "val_loss": [0.15, 0.18, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37],
                    "train_loss": [0.25, 0.28, 0.32, 0.35, 0.38, 0.41, 0.44, 0.47],
                    "batch_size": [64, 32, 16, 64, 32, 16, 32, 16],
                    "p_drop_conv": [0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2],
                    "p_drop_fc": [0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2],
                    "filters_1": [128, 64, 32, 128, 64, 32, 64, 32],
                    "filters_2": [256, 128, 64, 256, 128, 64, 128, 64],
                    "kernel_size_1": [8, 6, 4, 8, 6, 4, 6, 4],
                    "kernel_size_2": [9, 7, 5, 9, 7, 5, 7, 5],
                     "dense_units": [128, 64, 32, 128, 64, 32, 64, 32],
                     # learning_rate se maneja con scheduler, no es hiperparámetro
                }
            ),
            "best_params": {
                "f1": 0.92,
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "val_loss": 0.15,
                "train_loss": 0.25,
                "batch_size": 64,
                "p_drop_conv": 0.3,
                "p_drop_fc": 0.3,
                "filters_1": 128,
                "filters_2": 256,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "dense_units": 128,
                # learning_rate se maneja con scheduler, no es hiperparámetro
            },
        }

        # Guardar archivos de test
        csv_path = self.test_results_dir / "talos_scan_results.csv"
        json_path = self.test_results_dir / "best_params.json"

        new_data["results_df"].to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(new_data["best_params"], f, indent=2)

        # Verificar que los archivos se actualizaron
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())

        # Verificar contenido
        loaded_df = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_df), 8)  # 8 configuraciones nuevas
        self.assertEqual(loaded_df["f1"].max(), 0.92)  # Mejor F1-score

        with open(json_path, "r") as f:
            loaded_params = json.load(f)
        self.assertEqual(loaded_params["f1"], 0.92)
        self.assertEqual(loaded_params["batch_size"], 64)

        print(f"\nArchivos de test creados en: {self.test_results_dir}")
        print(f"Mejor F1-score: {loaded_df['f1'].max():.4f}")
        print(f"Total configuraciones: {len(loaded_df)}")

    def test_evaluate_test_files(self):
        """Evaluar los archivos de test."""
        # Primero crear los archivos de test
        self.test_update_test_files()

        # Luego evaluar
        evaluation = check_talos_results(str(self.test_results_dir))

        # Verificar que la evaluación fue exitosa
        self.assertEqual(evaluation["status"], "SUCCESS")
        self.assertTrue(evaluation["process_correct"])
        self.assertEqual(evaluation["total_configurations"], 8)
        self.assertEqual(evaluation["best_f1_score"], 0.92)

        print(f"\nEvaluacion exitosa:")
        print(f"Total configuraciones: {evaluation['total_configurations']}")
        print(f"Mejor F1-score: {evaluation['best_f1_score']:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
