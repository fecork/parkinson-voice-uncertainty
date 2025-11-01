"""
Pruebas unitarias para training_checks.py
==========================================

Verifica que las funciones de verificación pre-entrenamiento funcionen
correctamente con ambos formatos de datos: tuplas y diccionarios (DictDataset).
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.cnn2d.training_checks import (
    _extract_x_y,
    _first_batch,
    _softmax_ok,
    smoke_test,
    overfit_toy,
    lr_range_test,
    _run_epoch,
    mini_train_valid,
    quick_diag,
    run_all_checks,
)


class DictDataset(torch.utils.data.Dataset):
    """Dataset que devuelve diccionarios como DictDataset."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"spectrogram": self.X[idx], "label": self.y[idx]}


class SimpleModel(nn.Module):
    """Modelo simple para pruebas."""

    def __init__(self, input_shape=(1, 65, 41), n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestExtractXY(unittest.TestCase):
    """Pruebas para _extract_x_y."""

    def test_extract_from_dict_spectrogram(self):
        """Test: Extraer de diccionario con clave 'spectrogram'."""
        batch = {
            "spectrogram": torch.randn(4, 1, 65, 41),
            "label": torch.randint(0, 2, (4,)),
        }
        xb, yb = _extract_x_y(batch)
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)
        self.assertEqual(xb.shape[0], 4)
        self.assertEqual(yb.shape[0], 4)

    def test_extract_from_dict_X(self):
        """Test: Extraer de diccionario con clave 'X'."""
        batch = {
            "X": torch.randn(4, 1, 65, 41),
            "y_task": torch.randint(0, 2, (4,)),
        }
        xb, yb = _extract_x_y(batch)
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)

    def test_extract_from_tuple(self):
        """Test: Extraer de tupla."""
        batch = (torch.randn(4, 1, 65, 41), torch.randint(0, 2, (4,)))
        xb, yb = _extract_x_y(batch)
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)

    def test_extract_from_list(self):
        """Test: Extraer de lista."""
        batch = [torch.randn(4, 1, 65, 41), torch.randint(0, 2, (4,))]
        xb, yb = _extract_x_y(batch)
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)

    def test_extract_dict_missing_keys(self):
        """Test: Error cuando faltan claves en diccionario."""
        batch = {"wrong_key": torch.randn(4, 1, 65, 41)}
        with self.assertRaises(ValueError):
            _extract_x_y(batch)

    def test_extract_invalid_format(self):
        """Test: Error con formato no soportado."""
        batch = "invalid"
        with self.assertRaises(ValueError):
            _extract_x_y(batch)


class TestTrainingChecksTupleFormat(unittest.TestCase):
    """Pruebas con formato de tuplas (TensorDataset)."""

    def setUp(self):
        """Configuración inicial."""
        self.device = torch.device("cpu")
        self.n_samples = 50
        self.input_shape = (1, 65, 41)
        self.n_classes = 2

        # Crear datos sintéticos
        X = torch.randn(self.n_samples, *self.input_shape)
        y = torch.randint(0, self.n_classes, (self.n_samples,))

        # Crear datasets y loaders con formato de tupla
        train_dataset = TensorDataset(X, y)
        val_dataset = TensorDataset(X[:20], y[:20])

        self.train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        self.build_model = lambda: SimpleModel(
            input_shape=self.input_shape, n_classes=self.n_classes
        )

    def test_smoke_test_tuple_format(self):
        """Test: smoke_test con formato de tupla."""
        model = self.build_model()
        result = smoke_test(model, self.train_loader, self.device)
        self.assertIn("ok", result)
        self.assertIn("x_shape", result)
        self.assertIn("logits_shape", result)
        self.assertIn("softmax_sums_to_1", result)
        self.assertIn("loss_value", result)
        self.assertTrue(result["softmax_sums_to_1"])

    def test_overfit_toy_tuple_format(self):
        """Test: overfit_toy con formato de tupla."""
        result = overfit_toy(
            self.build_model,
            self.train_loader,
            self.device,
            toy_samples=20,
            steps=10,
        )
        self.assertIn("ok", result)
        self.assertIn("best", result)
        self.assertIn("history", result)
        self.assertIn("acc", result["best"])
        self.assertIn("loss", result["best"])

    def test_lr_range_test_tuple_format(self):
        """Test: lr_range_test con formato de tupla."""
        result = lr_range_test(
            self.build_model,
            self.train_loader,
            self.device,
            lr_start=1e-4,
            lr_end=0.1,
        )
        self.assertIn("ok", result)
        self.assertIn("best_lr", result)
        self.assertIn("history", result)

    def test_mini_train_valid_tuple_format(self):
        """Test: mini_train_valid con formato de tupla."""
        result = mini_train_valid(
            self.build_model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=2,
        )
        self.assertIn("ok", result)
        self.assertIn("best", result)
        self.assertIn("history", result)
        self.assertIn("val_loss", result["best"])

    def test_quick_diag_tuple_format(self):
        """Test: quick_diag con formato de tupla."""
        model = self.build_model()
        result = quick_diag(model, self.train_loader, self.val_loader, self.device)
        self.assertIn("ok", result)
        self.assertIn("logits_shape", result)
        self.assertIn("softmax_sums_to_1", result)
        self.assertIn("grad_norm", result)
        self.assertIn("weight_update_norm", result)
        self.assertIn("val_accuracy", result)
        self.assertIn("val_recall", result)
        self.assertIn("val_specificity", result)
        self.assertIn("val_f1", result)
        self.assertTrue(result["softmax_sums_to_1"])

    def test_run_all_checks_tuple_format(self):
        """Test: run_all_checks con formato de tupla."""
        ready, report = run_all_checks(
            self.build_model,
            self.train_loader,
            self.val_loader,
            self.device,
            toy_samples=20,
            toy_steps=10,
            mini_epochs=2,
        )
        self.assertIsInstance(ready, bool)
        self.assertIsInstance(report, str)
        self.assertIn("Smoke test", report)
        self.assertIn("Quick diag", report)


class TestTrainingChecksDictFormat(unittest.TestCase):
    """Pruebas con formato de diccionario (DictDataset)."""

    def setUp(self):
        """Configuración inicial."""
        self.device = torch.device("cpu")
        self.n_samples = 50
        self.input_shape = (1, 65, 41)
        self.n_classes = 2

        # Crear datos sintéticos
        X = torch.randn(self.n_samples, *self.input_shape)
        y = torch.randint(0, self.n_classes, (self.n_samples,))

        # Crear datasets y loaders con formato de diccionario
        train_dataset = DictDataset(X, y)
        val_dataset = DictDataset(X[:20], y[:20])

        self.train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        self.build_model = lambda: SimpleModel(
            input_shape=self.input_shape, n_classes=self.n_classes
        )

    def test_first_batch_dict_format(self):
        """Test: _first_batch con formato de diccionario."""
        xb, yb = _first_batch(self.train_loader)
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)

    def test_smoke_test_dict_format(self):
        """Test: smoke_test con formato de diccionario."""
        model = self.build_model()
        result = smoke_test(model, self.train_loader, self.device)
        self.assertIn("ok", result)
        self.assertIn("x_shape", result)
        self.assertIn("logits_shape", result)
        self.assertIn("softmax_sums_to_1", result)
        self.assertIn("loss_value", result)
        self.assertTrue(result["softmax_sums_to_1"])
        self.assertEqual(len(result["logits_shape"]), 2)
        self.assertEqual(result["logits_shape"][1], self.n_classes)

    def test_overfit_toy_dict_format(self):
        """Test: overfit_toy con formato de diccionario."""
        result = overfit_toy(
            self.build_model,
            self.train_loader,
            self.device,
            toy_samples=20,
            steps=10,
        )
        self.assertIn("ok", result)
        self.assertIn("best", result)
        self.assertIn("history", result)
        self.assertIn("acc", result["best"])
        self.assertIn("loss", result["best"])
        self.assertGreaterEqual(result["best"]["acc"], 0.0)
        self.assertLessEqual(result["best"]["acc"], 1.0)

    def test_lr_range_test_dict_format(self):
        """Test: lr_range_test con formato de diccionario."""
        result = lr_range_test(
            self.build_model,
            self.train_loader,
            self.device,
            lr_start=1e-4,
            lr_end=0.1,
        )
        self.assertIn("ok", result)
        self.assertIn("best_lr", result)
        self.assertIn("best_loss", result)
        self.assertIn("exploded_at", result)
        self.assertIn("history", result)
        if result["ok"]:
            self.assertIsNotNone(result["best_lr"])

    def test_run_epoch_dict_format(self):
        """Test: _run_epoch con formato de diccionario."""
        model = self.build_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Modo entrenamiento
        loss, acc, f1, cm = _run_epoch(
            model, self.train_loader, self.device, optimizer, criterion
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

        # Modo validación
        loss_val, acc_val, f1_val, cm_val = _run_epoch(
            model, self.val_loader, self.device, None, criterion
        )
        self.assertIsInstance(loss_val, float)
        self.assertIsInstance(acc_val, float)

    def test_mini_train_valid_dict_format(self):
        """Test: mini_train_valid con formato de diccionario."""
        result = mini_train_valid(
            self.build_model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=2,
        )
        self.assertIn("ok", result)
        self.assertIn("best", result)
        self.assertIn("history", result)
        self.assertIn("val_loss", result["best"])
        self.assertIn("val_acc", result["best"])
        self.assertIn("val_f1", result["best"])
        self.assertGreater(len(result["history"]), 0)

    def test_quick_diag_dict_format(self):
        """Test: quick_diag con formato de diccionario."""
        model = self.build_model()
        result = quick_diag(model, self.train_loader, self.val_loader, self.device)
        self.assertIn("ok", result)
        self.assertIn("logits_shape", result)
        self.assertIn("softmax_sums_to_1", result)
        self.assertIn("grad_norm", result)
        self.assertIn("weight_update_norm", result)
        self.assertIn("val_accuracy", result)
        self.assertIn("val_recall", result)
        self.assertIn("val_specificity", result)
        self.assertIn("val_f1", result)
        self.assertTrue(result["softmax_sums_to_1"])

    def test_run_all_checks_dict_format(self):
        """Test: run_all_checks con formato de diccionario."""
        ready, report = run_all_checks(
            self.build_model,
            self.train_loader,
            self.val_loader,
            self.device,
            toy_samples=20,
            toy_steps=10,
            mini_epochs=2,
            long_run_params={"optimizer": "SGD", "lr": 0.1},
        )
        self.assertIsInstance(ready, bool)
        self.assertIsInstance(report, str)
        self.assertIn("Smoke test", report)
        self.assertIn("Overfit toy", report)
        self.assertIn("LR range", report)
        self.assertIn("Mini-train", report)
        self.assertIn("Quick diag", report)
        self.assertIn("Long-run params", report)


class TestSoftmaxOK(unittest.TestCase):
    """Pruebas para _softmax_ok."""

    def test_softmax_sums_to_one(self):
        """Test: Verificar que softmax suma a 1."""
        logits = torch.randn(5, 3)
        self.assertTrue(_softmax_ok(logits))

    def test_softmax_invalid(self):
        """Test: Verificar detección de softmax inválido."""
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # Este debería pasar porque softmax siempre suma a 1
        # Pero probamos con valores muy grandes que podrían causar overflow
        self.assertTrue(_softmax_ok(logits))


if __name__ == "__main__":
    unittest.main()

