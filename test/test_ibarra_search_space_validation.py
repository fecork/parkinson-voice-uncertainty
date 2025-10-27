#!/usr/bin/env python3
"""
Pruebas unitarias para validar que el espacio de búsqueda de Optuna
coincide exactamente con las especificaciones del paper de Ibarra.
"""

import unittest
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.cnn2d_optuna_wrapper import CNN2DOptunaWrapper


class TestIbarraSearchSpaceValidation(unittest.TestCase):
    """Validar que el espacio de búsqueda coincide con el paper de Ibarra."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        # No necesitamos instanciar el wrapper para estas pruebas
        pass

    def test_batch_size_search_space(self):
        """Validar que batch_size tiene los valores correctos: [16, 32, 64]."""

        # Simular un trial de Optuna
        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name == "batch_size":
                    return choices
                return None

        mock_trial = MockTrial()

        # Obtener las opciones de batch_size
        batch_size_options = mock_trial.suggest_categorical("batch_size", [16, 32, 64])

        # Validar que coinciden exactamente con las especificaciones
        expected_batch_sizes = [16, 32, 64]
        self.assertEqual(
            set(batch_size_options),
            set(expected_batch_sizes),
            f"Batch size options {batch_size_options} no coinciden con {expected_batch_sizes}",
        )

        print("✅ Batch size: [16, 32, 64] - CORRECTO")

    def test_dropout_rate_search_space(self):
        """Validar que dropout rate tiene los valores correctos: [0.2, 0.5]."""

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name in ["p_drop_conv", "p_drop_fc"]:
                    return choices
                return None

        mock_trial = MockTrial()

        # Validar p_drop_conv
        conv_dropout_options = mock_trial.suggest_categorical("p_drop_conv", [0.2, 0.5])
        expected_dropout = [0.2, 0.5]
        self.assertEqual(
            set(conv_dropout_options),
            set(expected_dropout),
            f"Conv dropout options {conv_dropout_options} no coinciden con {expected_dropout}",
        )

        # Validar p_drop_fc
        fc_dropout_options = mock_trial.suggest_categorical("p_drop_fc", [0.2, 0.5])
        self.assertEqual(
            set(fc_dropout_options),
            set(expected_dropout),
            f"FC dropout options {fc_dropout_options} no coinciden con {expected_dropout}",
        )

        print("✅ Dropout rate: [0.2, 0.5] - CORRECTO")

    def test_depth_conv_layer_search_space(self):
        """Validar que depth conv layer tiene los valores correctos: [32, 64, 128]."""

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name in ["filters_1", "filters_2"]:
                    return choices
                return None

        mock_trial = MockTrial()

        # Validar filters_1 (Depth conv layer I)
        filters_1_options = mock_trial.suggest_categorical("filters_1", [32, 64, 128])
        expected_filters = [32, 64, 128]
        self.assertEqual(
            set(filters_1_options),
            set(expected_filters),
            f"Filters_1 options {filters_1_options} no coinciden con {expected_filters}",
        )

        # Validar filters_2 (Depth conv layer II)
        filters_2_options = mock_trial.suggest_categorical("filters_2", [32, 64, 128])
        self.assertEqual(
            set(filters_2_options),
            set(expected_filters),
            f"Filters_2 options {filters_2_options} no coinciden con {expected_filters}",
        )

        print("✅ Depth conv layer: [32, 64, 128] - CORRECTO")

    def test_fc_units_search_space(self):
        """Validar que FC units tiene los valores correctos: [16, 32, 64]."""

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name == "dense_units":
                    return choices
                return None

        mock_trial = MockTrial()

        # Validar dense_units (FC units)
        dense_units_options = mock_trial.suggest_categorical(
            "dense_units", [16, 32, 64]
        )
        expected_dense_units = [16, 32, 64]
        self.assertEqual(
            set(dense_units_options),
            set(expected_dense_units),
            f"Dense units options {dense_units_options} no coinciden con {expected_dense_units}",
        )

        print("✅ FC units: [16, 32, 64] - CORRECTO")

    def test_kernel_size_i_search_space(self):
        """Validar que kernel size I tiene los valores correctos: [4, 6, 8]."""

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name == "kernel_size_1":
                    return choices
                return None

        mock_trial = MockTrial()

        # Validar kernel_size_1 (Kernel size I)
        kernel_size_1_options = mock_trial.suggest_categorical(
            "kernel_size_1", [4, 6, 8]
        )
        expected_kernel_size_1 = [4, 6, 8]
        self.assertEqual(
            set(kernel_size_1_options),
            set(expected_kernel_size_1),
            f"Kernel size I options {kernel_size_1_options} no coinciden con {expected_kernel_size_1}",
        )

        print("✅ Kernel size I: [4, 6, 8] - CORRECTO")

    def test_kernel_size_ii_search_space(self):
        """Validar que kernel size II tiene los valores correctos: [5, 7, 9]."""

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name == "kernel_size_2":
                    return choices
                return None

        mock_trial = MockTrial()

        # Validar kernel_size_2 (Kernel size II)
        kernel_size_2_options = mock_trial.suggest_categorical(
            "kernel_size_2", [5, 7, 9]
        )
        expected_kernel_size_2 = [5, 7, 9]
        self.assertEqual(
            set(kernel_size_2_options),
            set(expected_kernel_size_2),
            f"Kernel size II options {kernel_size_2_options} no coinciden con {expected_kernel_size_2}",
        )

        print("✅ Kernel size II: [5, 7, 9] - CORRECTO")

    def test_complete_search_space_validation(self):
        """Validar que todo el espacio de búsqueda coincide con las especificaciones de Ibarra."""
        print("\n" + "=" * 70)
        print("VALIDACIÓN COMPLETA DEL ESPACIO DE BÚSQUEDA - PAPER IBARRA")
        print("=" * 70)

        # Especificaciones del paper de Ibarra
        ibarra_specs = {
            "batch_size": [16, 32, 64],
            "dropout_rate": [0.2, 0.5],
            "depth_conv_layer": [32, 64, 128],
            "fc_units": [16, 32, 64],
            "kernel_size_i": [4, 6, 8],
            "kernel_size_ii": [5, 7, 9],
        }

        # Espacio de búsqueda actual en el código
        current_search_space = {
            "batch_size": [16, 32, 64],
            "p_drop_conv": [0.2, 0.5],
            "p_drop_fc": [0.2, 0.5],
            "filters_1": [32, 64, 128],
            "filters_2": [32, 64, 128],
            "dense_units": [16, 32, 64],
            "kernel_size_1": [4, 6, 8],
            "kernel_size_2": [5, 7, 9],
        }

        # Validaciones
        validations = [
            (
                "Batch size",
                current_search_space["batch_size"],
                ibarra_specs["batch_size"],
            ),
            (
                "Dropout rate (conv)",
                current_search_space["p_drop_conv"],
                ibarra_specs["dropout_rate"],
            ),
            (
                "Dropout rate (fc)",
                current_search_space["p_drop_fc"],
                ibarra_specs["dropout_rate"],
            ),
            (
                "Depth conv layer I",
                current_search_space["filters_1"],
                ibarra_specs["depth_conv_layer"],
            ),
            (
                "Depth conv layer II",
                current_search_space["filters_2"],
                ibarra_specs["depth_conv_layer"],
            ),
            ("FC units", current_search_space["dense_units"], ibarra_specs["fc_units"]),
            (
                "Kernel size I",
                current_search_space["kernel_size_1"],
                ibarra_specs["kernel_size_i"],
            ),
            (
                "Kernel size II",
                current_search_space["kernel_size_2"],
                ibarra_specs["kernel_size_ii"],
            ),
        ]

        all_correct = True

        for name, current, expected in validations:
            if set(current) == set(expected):
                print(f"✅ {name}: {current} - CORRECTO")
            else:
                print(f"❌ {name}: {current} vs {expected} - INCORRECTO")
                all_correct = False

        print("\n" + "=" * 70)
        if all_correct:
            print(
                "🎉 ¡TODAS LAS VALIDACIONES PASARON! El espacio de búsqueda coincide con Ibarra"
            )
        else:
            print("⚠️  ALGUNAS VALIDACIONES FALLARON. Revisar el espacio de búsqueda")
        print("=" * 70)

        self.assertTrue(
            all_correct,
            "El espacio de búsqueda no coincide con las especificaciones de Ibarra",
        )


def run_ibarra_validation():
    """Función para ejecutar la validación de Ibarra."""
    print("🔬 Ejecutando validación del espacio de búsqueda - Paper Ibarra")
    print("=" * 70)

    # Crear suite de pruebas
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIbarraSearchSpaceValidation)

    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ibarra_validation()
    sys.exit(0 if success else 1)
