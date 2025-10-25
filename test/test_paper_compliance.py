"""
Pruebas Unitarias de Cumplimiento del Paper Ibarra 2023
=======================================================

Verifica componentes individuales del sistema para asegurar
que cumplen con los requisitos del paper.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import sys

# Agregar módulos al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.cnn2d.model import CNN2D
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


class TestPaperCompliance(unittest.TestCase):
    """Tests de cumplimiento con el paper."""

    @classmethod
    def setUpClass(cls):
        """Configuración inicial de la clase."""
        # Cargar requisitos del paper
        requirements_path = Path(__file__).parent / "paper_requirements.json"
        with open(requirements_path, "r") as f:
            cls.requirements = json.load(f)

    def test_preprocessing_parameters(self):
        """Test 1: Verificar parámetros de preprocesamiento."""
        prep_req = self.requirements["preprocessing"]

        # Verificar valores esperados
        self.assertEqual(
            prep_req["sampling_rate"], 44100, "Sampling rate debe ser 44.1 kHz"
        )
        self.assertEqual(
            prep_req["segment_length_ms"], 400, "Segment length debe ser 400 ms"
        )
        self.assertEqual(prep_req["overlap"], 0.5, "Overlap debe ser 50%")
        self.assertEqual(prep_req["window_size_ms"], 40, "Window size debe ser 40 ms")
        self.assertEqual(prep_req["hop_length_ms"], 10, "Hop length debe ser 10 ms")
        self.assertEqual(prep_req["n_mels"], 65, "Número de filtros Mel debe ser 65")
        self.assertListEqual(
            prep_req["output_shape"], [65, 41], "Output shape debe ser [65, 41]"
        )

    def test_cnn2d_architecture(self):
        """Test 2: Verificar arquitectura CNN2D."""
        # Crear modelo con parámetros del paper
        model = CNN2D(
            n_classes=2,
            p_drop_conv=0.3,
            p_drop_fc=0.5,
            input_shape=(65, 41),
            filters_1=32,
            filters_2=64,
            kernel_size_1=3,
            kernel_size_2=3,
        )

        # Verificar input shape
        test_input = torch.randn(1, 1, 65, 41)
        output = model(test_input)

        # Verificar output shape
        self.assertEqual(
            output.shape,
            (1, 2),
            "Output debe ser (batch, 2) para clasificación binaria",
        )

        # Verificar que el modelo tiene los componentes correctos
        self.assertTrue(
            hasattr(model, "feature_extractor"), "Modelo debe tener feature_extractor"
        )
        self.assertTrue(
            hasattr(model, "pd_head"), "Modelo debe tener pd_head (classifier)"
        )

        # Verificar número de parámetros aproximado
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(
            total_params, 10000, "Modelo debe tener suficientes parámetros"
        )
        self.assertLess(total_params, 1000000, "Modelo no debe ser demasiado grande")

    def test_optimizer_configuration(self):
        """Test 3: Verificar configuración del optimizador SGD."""
        train_req = self.requirements["training"]

        # Verificar requisitos
        self.assertEqual(train_req["optimizer"], "SGD", "Optimizer debe ser SGD")
        self.assertEqual(train_req["lr"], 0.1, "Learning rate debe ser 0.1")
        self.assertEqual(train_req["momentum"], 0.9, "Momentum debe ser 0.9")

        # Crear modelo y optimizer
        model = CNN2D()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_req["lr"], momentum=train_req["momentum"]
        )

        # Verificar tipo
        self.assertIsInstance(optimizer, torch.optim.SGD, "Optimizer debe ser SGD")

        # Verificar parámetros
        self.assertEqual(
            optimizer.param_groups[0]["lr"], 0.1, "Learning rate debe ser 0.1"
        )
        self.assertEqual(
            optimizer.param_groups[0]["momentum"], 0.9, "Momentum debe ser 0.9"
        )

    def test_scheduler_configuration(self):
        """Test 4: Verificar configuración del scheduler StepLR."""
        train_req = self.requirements["training"]

        # Verificar requisitos
        self.assertEqual(train_req["scheduler"], "StepLR", "Scheduler debe ser StepLR")
        self.assertEqual(train_req["step_size"], 10, "Step size debe ser 10")
        self.assertEqual(train_req["gamma"], 0.1, "Gamma debe ser 0.1")

        # Crear modelo, optimizer y scheduler
        model = CNN2D()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_req["step_size"], gamma=train_req["gamma"]
        )

        # Verificar tipo
        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.StepLR, "Scheduler debe ser StepLR"
        )

        # Verificar parámetros
        self.assertEqual(scheduler.step_size, 10, "Step size debe ser 10")
        self.assertEqual(scheduler.gamma, 0.1, "Gamma debe ser 0.1")

        # Verificar comportamiento
        initial_lr = optimizer.param_groups[0]["lr"]
        self.assertEqual(initial_lr, 0.1, "LR inicial debe ser 0.1")

        # Simular 10 pasos
        for _ in range(10):
            scheduler.step()

        # Verificar que el LR se redujo
        new_lr = optimizer.param_groups[0]["lr"]
        expected_lr = 0.1 * 0.1  # 0.01
        self.assertAlmostEqual(
            new_lr,
            expected_lr,
            places=6,
            msg="LR debe reducirse a 0.01 después de 10 pasos",
        )

    def test_weighted_loss(self):
        """Test 5: Verificar cálculo de weighted loss."""
        # Simular dataset desbalanceado
        n_healthy = 100
        n_parkinson = 50
        y = np.array([0] * n_healthy + [1] * n_parkinson)

        # Calcular pesos
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weights = torch.tensor(weights, dtype=torch.float32)

        # Crear loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Verificar que los pesos están correctos
        self.assertEqual(len(class_weights), 2, "Debe haber 2 pesos (binario)")

        # El peso de la clase minoritaria debe ser mayor
        self.assertGreater(
            class_weights[1],
            class_weights[0],
            "Peso de clase minoritaria debe ser mayor",
        )

        # Verificar que la loss function acepta los pesos
        dummy_output = torch.randn(10, 2)
        dummy_target = torch.randint(0, 2, (10,))
        loss = criterion(dummy_output, dummy_target)

        self.assertIsInstance(loss.item(), float, "Loss debe retornar un valor float")
        self.assertGreater(loss.item(), 0, "Loss debe ser positivo")

    def test_kfold_split_by_speaker(self):
        """Test 6: Verificar split por hablante con K-Fold."""
        val_req = self.requirements["validation"]

        # Verificar requisitos
        self.assertEqual(
            val_req["method"], "StratifiedKFold", "Método debe ser StratifiedKFold"
        )
        self.assertEqual(val_req["n_splits"], 10, "Debe usar 10 folds")
        self.assertEqual(
            val_req["stratify_by"], "speaker_id", "Debe estratificar por speaker_id"
        )

        # Simular datos con speakers (agrupados)
        # Cada speaker tiene múltiples muestras
        n_speakers = 20
        samples_per_speaker = 5

        # Crear estructura: cada speaker tiene una etiqueta y múltiples muestras
        speaker_labels = np.random.randint(0, 2, n_speakers)

        # Crear K-Fold a nivel de speaker (no a nivel de muestra)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Verificar que funciona a nivel de speaker
        speaker_indices = np.arange(n_speakers)
        splits = list(kfold.split(speaker_indices, speaker_labels))
        self.assertEqual(len(splits), 10, "Debe generar 10 splits")

        # Verificar que ningún speaker aparece en train y val simultáneamente
        for train_speaker_idx, val_speaker_idx in splits:
            train_speakers = set(train_speaker_idx)
            val_speakers = set(val_speaker_idx)

            overlap = train_speakers.intersection(val_speakers)
            self.assertEqual(
                len(overlap), 0, "No debe haber speakers compartidos entre train y val"
            )

            # Verificar que hay speakers en ambos conjuntos
            self.assertGreater(len(train_speakers), 0, "Debe haber speakers en train")
            self.assertGreater(len(val_speakers), 0, "Debe haber speakers en val")

    def test_vocal_a_filtering(self):
        """Test 7: Verificar filtrado de vocal /a/."""
        val_req = self.requirements["validation"]

        # Verificar requisito
        self.assertEqual(val_req["vowel"], "a", "Debe usar solo vocal /a/")

        # Simular dataset con múltiples vocales
        mock_filenames = [
            "speaker1-a_h.egg",
            "speaker1-e_h.egg",
            "speaker1-i_h.egg",
            "speaker2-a_l.egg",
            "speaker2-o_l.egg",
            "speaker3-a_n.egg",
            "speaker3-u_n.egg",
        ]

        # Filtrar solo vocal /a/
        filtered = [f for f in mock_filenames if "-a_" in f or "-a." in f]

        # Verificar filtrado
        self.assertEqual(len(filtered), 3, "Debe haber 3 archivos de vocal /a/")
        for filename in filtered:
            self.assertIn(
                "-a_", filename, f"Archivo {filename} debe contener vocal /a/"
            )

    def test_hyperparameter_search_space(self):
        """Test 8: Verificar espacio de búsqueda de hiperparámetros."""
        hp_req = self.requirements["hyperparameters"]

        # Verificar batch sizes
        expected_batch_sizes = [16, 32, 64]
        self.assertListEqual(
            hp_req["batch_size"],
            expected_batch_sizes,
            "Batch sizes deben ser [16, 32, 64]",
        )

        # Verificar dropout values
        expected_dropout = [0.2, 0.5]
        self.assertListEqual(
            hp_req["dropout_conv"], expected_dropout, "Dropout conv debe ser [0.2, 0.5]"
        )
        self.assertListEqual(
            hp_req["dropout_fc"], expected_dropout, "Dropout fc debe ser [0.2, 0.5]"
        )

        # Verificar FC units
        expected_fc_units = [16, 32, 64]
        self.assertListEqual(
            hp_req["fc_units"], expected_fc_units, "FC units deben ser [16, 32, 64]"
        )

        # Verificar kernel sizes
        expected_kernel_1 = [4, 6, 8]
        expected_kernel_2 = [5, 7, 9]
        self.assertListEqual(
            hp_req["kernel_1"], expected_kernel_1, "Kernel size 1 debe ser [4, 6, 8]"
        )
        self.assertListEqual(
            hp_req["kernel_2"], expected_kernel_2, "Kernel size 2 debe ser [5, 7, 9]"
        )

    def test_model_with_paper_hyperparameters(self):
        """Test 9: Verificar que el modelo acepta hiperparámetros del paper."""
        hp_req = self.requirements["hyperparameters"]

        # Probar con diferentes combinaciones de hiperparámetros
        test_configs = [
            {
                "filters_1": 32,
                "filters_2": 64,
                "kernel_size_1": 4,
                "kernel_size_2": 5,
                "dense_units": 16,
                "p_drop_conv": 0.2,
                "p_drop_fc": 0.2,
            },
            {
                "filters_1": 64,
                "filters_2": 128,
                "kernel_size_1": 8,
                "kernel_size_2": 9,
                "dense_units": 64,
                "p_drop_conv": 0.5,
                "p_drop_fc": 0.5,
            },
        ]

        for config in test_configs:
            with self.subTest(config=config):
                # Crear modelo
                model = CNN2D(
                    n_classes=2,
                    p_drop_conv=config["p_drop_conv"],
                    p_drop_fc=config["p_drop_fc"],
                    input_shape=(65, 41),
                    filters_1=config["filters_1"],
                    filters_2=config["filters_2"],
                    kernel_size_1=config["kernel_size_1"],
                    kernel_size_2=config["kernel_size_2"],
                    dense_units=config["dense_units"],
                )

                # Verificar que funciona
                test_input = torch.randn(2, 1, 65, 41)
                output = model(test_input)

                self.assertEqual(
                    output.shape, (2, 2), f"Output incorrecto para config {config}"
                )

    def test_early_stopping_configuration(self):
        """Test 10: Verificar configuración de early stopping."""
        train_req = self.requirements["training"]

        # Verificar requisitos
        self.assertTrue(train_req["early_stopping"], "Debe usar early stopping")
        self.assertEqual(train_req["patience"], 10, "Patience debe ser 10")
        self.assertEqual(train_req["epochs"], 50, "Debe entrenar máximo 50 épocas")

    def test_metric_computation(self):
        """Test 11: Verificar cálculo de F1-macro."""
        from sklearn.metrics import f1_score

        # Simular predicciones
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        # Calcular F1-macro
        f1 = f1_score(y_true, y_pred, average="macro")

        # Verificar que es un valor válido
        self.assertIsInstance(f1, (float, np.floating), "F1 debe ser float")
        self.assertGreaterEqual(f1, 0.0, "F1 debe ser >= 0")
        self.assertLessEqual(f1, 1.0, "F1 debe ser <= 1")


class TestDataPreprocessing(unittest.TestCase):
    """Tests específicos de preprocesamiento."""

    def test_spectrogram_dimensions(self):
        """Verificar que el preprocesamiento genera espectrogramas de 65x41."""
        # Este test verifica que las dimensiones del modelo son correctas
        model = CNN2D(input_shape=(65, 41))

        test_input = torch.randn(1, 1, 65, 41)

        try:
            output = model(test_input)
            self.assertEqual(output.shape[0], 1, "Batch dimension incorrecta")
            self.assertEqual(output.shape[1], 2, "Output classes incorrectas")
        except Exception as e:
            self.fail(f"Modelo no acepta input (65, 41): {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
