"""
Pruebas Unitarias para GradCAM - Matemática y Algoritmos
========================================================

Este módulo contiene pruebas unitarias para validar la implementación matemática
de GradCAM según el paper original de Selvaraju et al. (2017).

Referencias:
- Selvaraju et al. (2017): "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Implementación: modules/models/cnn2d/model.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import Tuple

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.cnn2d.model import GradCAM, get_last_conv_layer
from modules.models.cnn2d.model import CNN2D


class TestGradCAMMath(unittest.TestCase):
    """Pruebas unitarias para la matemática de GradCAM."""

    def setUp(self):
        """Configuración inicial para cada test."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Crear modelo de prueba
        self.model = CNN2D(n_classes=2).to(self.device)
        self.model.eval()

        # Obtener última capa convolucional
        self.target_layer = get_last_conv_layer(self.model)

        # Crear instancia GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Datos de prueba
        self.batch_size = 2
        self.input_shape = (self.batch_size, 1, 65, 41)
        self.x = torch.randn(self.input_shape, device=self.device, requires_grad=True)

    def test_gradcam_initialization(self):
        """Test: Inicialización correcta de GradCAM."""
        self.assertIsNotNone(self.gradcam.model)
        self.assertIsNotNone(self.gradcam.target_layer)
        self.assertIsNone(self.gradcam.gradients)
        self.assertIsNone(self.gradcam.activations)

    def test_forward_hook_registration(self):
        """Test: Hooks registrados correctamente."""
        # Verificar que los hooks están registrados
        forward_hooks = self.target_layer._forward_hooks
        backward_hooks = self.target_layer._backward_hooks

        self.assertGreater(len(forward_hooks), 0)
        self.assertGreater(len(backward_hooks), 0)

    def test_activation_saving(self):
        """Test: Guardado correcto de activaciones."""
        # Forward pass
        _ = self.model(self.x)

        # Verificar que las activaciones se guardaron
        self.assertIsNotNone(self.gradcam.activations)
        self.assertEqual(self.gradcam.activations.device, self.x.device)

        # Verificar dimensiones
        expected_shape = self.gradcam.activations.shape
        self.assertEqual(len(expected_shape), 4)  # (B, C, H, W)

    def test_gradient_saving(self):
        """Test: Guardado correcto de gradientes."""
        # Forward pass
        logits = self.model(self.x)

        # Backward pass
        target_class = torch.tensor([0, 1], device=self.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Verificar que los gradientes se guardaron
        self.assertIsNotNone(self.gradcam.gradients)
        self.assertEqual(self.gradcam.gradients.device, self.x.device)

        # Verificar dimensiones
        expected_shape = self.gradcam.gradients.shape
        self.assertEqual(len(expected_shape), 4)  # (B, C, H, W)

    def test_gap_calculation(self):
        """Test: Cálculo correcto de Global Average Pooling."""
        # Forward pass
        _ = self.model(self.x)

        # Backward pass
        logits = self.model(self.x)
        target_class = torch.tensor([0, 1], device=self.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Calcular GAP manualmente
        gradients = self.gradcam.gradients
        manual_gap = gradients.mean(dim=[2, 3], keepdim=True)

        # Verificar dimensiones del GAP
        self.assertEqual(manual_gap.shape[0], self.batch_size)  # Batch size
        self.assertEqual(manual_gap.shape[1], gradients.shape[1])  # Número de canales
        self.assertEqual(manual_gap.shape[2], 1)  # Height = 1
        self.assertEqual(manual_gap.shape[3], 1)  # Width = 1

    def test_weighted_combination(self):
        """Test: Combinación ponderada de activaciones."""
        # Forward pass
        _ = self.model(self.x)

        # Backward pass
        logits = self.model(self.x)
        target_class = torch.tensor([0, 1], device=self.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Calcular pesos (GAP de gradientes)
        weights = self.gradcam.gradients.mean(dim=[2, 3], keepdim=True)

        # Combinación ponderada
        activations = self.gradcam.activations
        weighted_combination = (weights * activations).sum(dim=1, keepdim=True)

        # Verificar dimensiones
        self.assertEqual(weighted_combination.shape[0], self.batch_size)
        self.assertEqual(weighted_combination.shape[1], 1)  # Suma sobre canales
        self.assertEqual(weighted_combination.shape[2], activations.shape[2])
        self.assertEqual(weighted_combination.shape[3], activations.shape[3])

    def test_relu_application(self):
        """Test: Aplicación correcta de ReLU."""
        # Crear tensor con valores negativos y positivos
        test_tensor = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], device=self.device)

        # Aplicar ReLU
        relu_result = F.relu(test_tensor)

        # Verificar que los valores negativos se convierten a 0
        expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]], device=self.device)
        self.assertTrue(torch.allclose(relu_result, expected))

    def test_normalization_math(self):
        """Test: Matemática de normalización [0, 1]."""
        # Crear tensor de prueba
        test_tensor = torch.tensor([[1.0, 3.0, 5.0, 7.0]], device=self.device)

        # Normalización manual
        min_val = test_tensor.min(dim=1, keepdim=True)[0]
        max_val = test_tensor.max(dim=1, keepdim=True)[0]
        normalized = (test_tensor - min_val) / (max_val - min_val + 1e-8)

        # Verificar que el rango está en [0, 1]
        self.assertTrue(torch.all(normalized >= 0.0))
        self.assertTrue(torch.all(normalized <= 1.0))

        # Verificar que min = 0 y max = 1
        self.assertAlmostEqual(normalized.min().item(), 0.0, places=6)
        self.assertAlmostEqual(normalized.max().item(), 1.0, places=6)

    def test_bilinear_interpolation(self):
        """Test: Interpolación bilineal correcta."""
        # Crear tensor de entrada
        input_tensor = torch.randn(1, 1, 10, 10, device=self.device)
        target_size = (65, 41)

        # Interpolación bilineal
        interpolated = F.interpolate(
            input_tensor, size=target_size, mode="bilinear", align_corners=False
        )

        # Verificar dimensiones
        self.assertEqual(interpolated.shape[2], target_size[0])
        self.assertEqual(interpolated.shape[3], target_size[1])

    def test_cam_generation_complete(self):
        """Test: Generación completa de CAM."""
        # Generar CAM
        cam = self.gradcam.generate_cam(self.x, target_class=0)

        # Verificar dimensiones
        self.assertEqual(cam.shape[0], self.batch_size)
        self.assertEqual(cam.shape[1], 65)  # Height
        self.assertEqual(cam.shape[2], 41)  # Width

        # Verificar que está normalizado [0, 1]
        self.assertTrue(torch.all(cam >= 0.0))
        self.assertTrue(torch.all(cam <= 1.0))

        # Verificar que no es todo ceros
        self.assertTrue(torch.any(cam > 0.0))

    def test_cam_consistency(self):
        """Test: Consistencia de CAM entre ejecuciones."""
        # Generar CAM múltiples veces
        cam1 = self.gradcam.generate_cam(self.x, target_class=0)
        cam2 = self.gradcam.generate_cam(self.x, target_class=0)

        # Deben ser idénticos (mismo input, mismo target)
        self.assertTrue(torch.allclose(cam1, cam2, atol=1e-6))

    def test_different_target_classes(self):
        """Test: CAM diferente para diferentes clases objetivo."""
        # Generar CAM para clase 0
        cam_class_0 = self.gradcam.generate_cam(self.x, target_class=0)

        # Generar CAM para clase 1
        cam_class_1 = self.gradcam.generate_cam(self.x, target_class=1)

        # Deben ser diferentes (a menos que sea muy similar)
        difference = torch.abs(cam_class_0 - cam_class_1).mean()
        self.assertGreater(difference.item(), 1e-6)

    def test_gradcam_mathematical_properties(self):
        """Test: Propiedades matemáticas de GradCAM."""
        # Generar CAM
        cam = self.gradcam.generate_cam(self.x, target_class=0)

        # 1. No negatividad (ReLU aplicado)
        self.assertTrue(torch.all(cam >= 0.0))

        # 2. Normalización [0, 1]
        self.assertTrue(torch.all(cam <= 1.0))

        # 3. Continuidad (no debe haber saltos abruptos)
        # Calcular diferencia entre píxeles adyacentes
        diff_h = torch.abs(cam[:, 1:, :] - cam[:, :-1, :])
        diff_w = torch.abs(cam[:, :, 1:] - cam[:, :, :-1])

        # Las diferencias deben ser razonables (no infinitas)
        self.assertTrue(torch.all(torch.isfinite(diff_h)))
        self.assertTrue(torch.all(torch.isfinite(diff_w)))

    def test_gradcam_gradient_flow(self):
        """Test: Flujo de gradientes correcto."""
        # Forward pass
        logits = self.model(self.x)

        # Verificar que los gradientes fluyen
        self.assertTrue(self.x.requires_grad)

        # Backward pass
        target_class = torch.tensor([0, 1], device=self.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Verificar que los gradientes se calcularon
        self.assertIsNotNone(self.gradcam.gradients)
        self.assertTrue(torch.any(torch.abs(self.gradcam.gradients) > 1e-8))

    def test_memory_efficiency(self):
        """Test: Eficiencia de memoria."""
        # Limpiar gradientes iniciales
        self.model.zero_grad()

        # Verificar que no hay gradientes inicialmente
        initial_grad_count = len(
            [p for p in self.model.parameters() if p.grad is not None]
        )
        self.assertEqual(initial_grad_count, 0)

        # Forward pass
        logits = self.model(self.x)

        # Backward pass
        target_class = torch.tensor([0, 1], device=self.device)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        logits.backward(gradient=one_hot, retain_graph=True)

        # Verificar que se calcularon gradientes
        final_grad_count = len(
            [p for p in self.model.parameters() if p.grad is not None]
        )
        self.assertGreater(final_grad_count, 0)

        # Verificar que todos los parámetros tienen gradientes
        total_params = len(list(self.model.parameters()))
        self.assertEqual(final_grad_count, total_params)

    def test_batch_processing(self):
        """Test: Procesamiento correcto de batch."""
        # Crear batch más grande
        large_batch_size = 4
        large_x = torch.randn(
            large_batch_size, 1, 65, 41, device=self.device, requires_grad=True
        )

        # Generar CAM para batch grande
        cam = self.gradcam.generate_cam(large_x, target_class=0)

        # Verificar dimensiones
        self.assertEqual(cam.shape[0], large_batch_size)
        self.assertEqual(cam.shape[1], 65)
        self.assertEqual(cam.shape[2], 41)

        # Verificar que cada muestra en el batch es diferente
        # (a menos que los inputs sean idénticos)
        for i in range(1, large_batch_size):
            difference = torch.abs(cam[0] - cam[i]).mean()
            # Debe haber alguna diferencia (a menos que sea muy similar)
            self.assertGreater(difference.item(), 1e-8)

    def test_edge_cases(self):
        """Test: Casos extremos."""
        # 1. Tensor con todos los valores iguales
        uniform_x = torch.ones(1, 1, 65, 41, device=self.device, requires_grad=True)
        cam_uniform = self.gradcam.generate_cam(uniform_x, target_class=0)

        # Debe generar CAM válido
        self.assertTrue(torch.all(torch.isfinite(cam_uniform)))
        self.assertTrue(torch.all(cam_uniform >= 0.0))
        self.assertTrue(torch.all(cam_uniform <= 1.0))

        # 2. Tensor con valores muy pequeños
        small_x = (
            torch.randn(1, 1, 65, 41, device=self.device, requires_grad=True) * 1e-8
        )
        cam_small = self.gradcam.generate_cam(small_x, target_class=0)

        # Debe manejar valores pequeños correctamente
        self.assertTrue(torch.all(torch.isfinite(cam_small)))

        # 3. Tensor con valores muy grandes
        large_x = (
            torch.randn(1, 1, 65, 41, device=self.device, requires_grad=True) * 1e8
        )
        cam_large = self.gradcam.generate_cam(large_x, target_class=0)

        # Debe manejar valores grandes correctamente
        self.assertTrue(torch.all(torch.isfinite(cam_large)))


class TestGradCAMIntegration(unittest.TestCase):
    """Pruebas de integración para GradCAM."""

    def setUp(self):
        """Configuración inicial."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN2D(n_classes=2).to(self.device)
        self.model.eval()

    def test_gradcam_with_real_model(self):
        """Test: GradCAM con modelo real."""
        # Crear instancia GradCAM
        target_layer = get_last_conv_layer(self.model)
        gradcam = GradCAM(self.model, target_layer)

        # Crear datos de prueba
        x = torch.randn(1, 1, 65, 41, device=self.device, requires_grad=True)

        # Generar CAM
        cam = gradcam.generate_cam(x, target_class=0)

        # Verificar que funciona con modelo real
        self.assertIsNotNone(cam)
        self.assertEqual(cam.shape, (1, 65, 41))

    def test_gradcam_with_different_models(self):
        """Test: GradCAM con diferentes modelos."""
        # Crear diferentes modelos
        models = [
            CNN2D(n_classes=2),
            CNN2D(n_classes=2, p_drop_conv=0.5),
        ]

        for model in models:
            model = model.to(self.device)
            model.eval()

            # Crear GradCAM
            target_layer = get_last_conv_layer(model)
            gradcam = GradCAM(model, target_layer)

            # Crear datos de prueba
            x = torch.randn(1, 1, 65, 41, device=self.device, requires_grad=True)

            # Generar CAM
            cam = gradcam.generate_cam(x, target_class=0)

            # Verificar que funciona
            self.assertIsNotNone(cam)
            self.assertEqual(cam.shape, (1, 65, 41))


def run_gradcam_tests():
    """Ejecutar todas las pruebas de GradCAM."""
    print("🧪 Ejecutando pruebas unitarias de GradCAM...")
    print("=" * 60)

    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar pruebas
    suite.addTests(loader.loadTestsFromTestCase(TestGradCAMMath))
    suite.addTests(loader.loadTestsFromTestCase(TestGradCAMIntegration))

    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Mostrar resumen
    print("\n" + "=" * 60)
    print(f"✅ Pruebas ejecutadas: {result.testsRun}")
    print(f"❌ Fallos: {len(result.failures)}")
    print(f"⚠️  Errores: {len(result.errors)}")

    if result.failures:
        print("\n🔍 Fallos:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n🚨 Errores:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_gradcam_tests()
    sys.exit(0 if success else 1)
