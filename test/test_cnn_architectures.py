"""
Pruebas unitarias para verificar las arquitecturas de los modelos CNN2D y CNN2D_DA.

Este módulo verifica que ambos modelos tengan todas las capas correspondientes
en su arquitectura según las especificaciones del proyecto.
"""

import unittest
import torch
import torch.nn as nn
from typing import List, Dict, Any

# Importar los modelos a probar
from modules.models.cnn2d.model import CNN2D, CNN2D_DA
from modules.models.common.layers import (
    FeatureExtractor,
    GradientReversalLayer,
    ClassifierHead,
)


class TestCNNArchitectures(unittest.TestCase):
    """Pruebas para verificar las arquitecturas de CNN2D y CNN2D_DA."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.device = torch.device("cpu")
        self.input_shape = (65, 41)
        self.batch_size = 2

        # Crear modelos
        self.cnn2d = CNN2D(
            n_classes=2, p_drop_conv=0.3, p_drop_fc=0.5, input_shape=self.input_shape
        ).to(self.device)

        self.cnn2d_da = CNN2D_DA(
            n_domains=26, p_drop_conv=0.3, p_drop_fc=0.5, input_shape=self.input_shape
        ).to(self.device)

        # Tensor de prueba
        self.test_input = torch.randn(self.batch_size, 1, *self.input_shape).to(
            self.device
        )

    def test_cnn2d_has_feature_extractor(self):
        """Verificar que CNN2D tiene FeatureExtractor."""
        self.assertIsInstance(
            self.cnn2d.feature_extractor,
            FeatureExtractor,
            "CNN2D debe tener un FeatureExtractor",
        )

    def test_cnn2d_has_pd_head(self):
        """Verificar que CNN2D tiene PD head."""
        self.assertIsInstance(
            self.cnn2d.pd_head,
            ClassifierHead,
            "CNN2D debe tener un ClassifierHead para PD",
        )

    def test_cnn2d_feature_extractor_structure(self):
        """Verificar estructura del FeatureExtractor en CNN2D."""
        feature_extractor = self.cnn2d.feature_extractor

        # Verificar que tiene block1 y block2
        self.assertTrue(
            hasattr(feature_extractor, "block1"), "FeatureExtractor debe tener block1"
        )
        self.assertTrue(
            hasattr(feature_extractor, "block2"), "FeatureExtractor debe tener block2"
        )

        # Verificar estructura de block1
        block1 = feature_extractor.block1
        self.assertIsInstance(block1, nn.Sequential)
        self.assertEqual(
            len(block1), 5
        )  # Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d

        # Verificar tipos de capas en block1
        self.assertIsInstance(block1[0], nn.Conv2d)
        self.assertIsInstance(block1[1], nn.BatchNorm2d)
        self.assertIsInstance(block1[2], nn.ReLU)
        self.assertIsInstance(block1[3], nn.MaxPool2d)
        self.assertIsInstance(block1[4], nn.Dropout2d)

        # Verificar estructura de block2
        block2 = feature_extractor.block2
        self.assertIsInstance(block2, nn.Sequential)
        self.assertEqual(
            len(block2), 5
        )  # Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d

        # Verificar tipos de capas en block2
        self.assertIsInstance(block2[0], nn.Conv2d)
        self.assertIsInstance(block2[1], nn.BatchNorm2d)
        self.assertIsInstance(block2[2], nn.ReLU)
        self.assertIsInstance(block2[3], nn.MaxPool2d)
        self.assertIsInstance(block2[4], nn.Dropout2d)

    def test_cnn2d_pd_head_structure(self):
        """Verificar estructura del PD head en CNN2D."""
        pd_head = self.cnn2d.pd_head

        # Verificar que es un ClassifierHead
        self.assertIsInstance(pd_head, ClassifierHead)

        # Verificar estructura interna del classifier
        classifier = pd_head.classifier
        self.assertIsInstance(classifier, nn.Sequential)
        self.assertEqual(len(classifier), 5)  # Flatten, Linear, ReLU, Dropout, Linear

        # Verificar tipos de capas
        self.assertIsInstance(classifier[0], nn.Flatten)
        self.assertIsInstance(classifier[1], nn.Linear)
        self.assertIsInstance(classifier[2], nn.ReLU)
        self.assertIsInstance(classifier[3], nn.Dropout)
        self.assertIsInstance(classifier[4], nn.Linear)

        # Verificar dimensiones de salida
        self.assertEqual(classifier[4].out_features, 2)  # 2 clases: HC/PD

    def test_cnn2d_da_has_feature_extractor(self):
        """Verificar que CNN2D_DA tiene FeatureExtractor."""
        self.assertIsInstance(
            self.cnn2d_da.feature_extractor,
            FeatureExtractor,
            "CNN2D_DA debe tener un FeatureExtractor",
        )

    def test_cnn2d_da_has_pd_head(self):
        """Verificar que CNN2D_DA tiene PD head."""
        self.assertIsInstance(
            self.cnn2d_da.pd_head,
            ClassifierHead,
            "CNN2D_DA debe tener un ClassifierHead para PD",
        )

    def test_cnn2d_da_has_grl(self):
        """Verificar que CNN2D_DA tiene Gradient Reversal Layer."""
        self.assertIsInstance(
            self.cnn2d_da.grl,
            GradientReversalLayer,
            "CNN2D_DA debe tener un GradientReversalLayer",
        )

    def test_cnn2d_da_has_domain_head(self):
        """Verificar que CNN2D_DA tiene Domain head."""
        self.assertIsInstance(
            self.cnn2d_da.domain_head,
            ClassifierHead,
            "CNN2D_DA debe tener un ClassifierHead para dominio",
        )

    def test_cnn2d_da_domain_head_structure(self):
        """Verificar estructura del Domain head en CNN2D_DA."""
        domain_head = self.cnn2d_da.domain_head

        # Verificar que es un ClassifierHead
        self.assertIsInstance(domain_head, ClassifierHead)

        # Verificar estructura interna del classifier
        classifier = domain_head.classifier
        self.assertIsInstance(classifier, nn.Sequential)
        self.assertEqual(len(classifier), 5)  # Flatten, Linear, ReLU, Dropout, Linear

        # Verificar tipos de capas
        self.assertIsInstance(classifier[0], nn.Flatten)
        self.assertIsInstance(classifier[1], nn.Linear)
        self.assertIsInstance(classifier[2], nn.ReLU)
        self.assertIsInstance(classifier[3], nn.Dropout)
        self.assertIsInstance(classifier[4], nn.Linear)

        # Verificar dimensiones de salida (26 dominios)
        self.assertEqual(classifier[4].out_features, 26)

    def test_cnn2d_forward_pass(self):
        """Verificar que CNN2D hace forward pass correctamente."""
        with torch.no_grad():
            output = self.cnn2d(self.test_input)

            # Verificar dimensiones de salida
            self.assertEqual(output.shape, (self.batch_size, 2))
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_cnn2d_da_forward_pass(self):
        """Verificar que CNN2D_DA hace forward pass correctamente."""
        with torch.no_grad():
            pd_logits, domain_logits = self.cnn2d_da(self.test_input)

            # Verificar dimensiones de salida
            self.assertEqual(pd_logits.shape, (self.batch_size, 2))
            self.assertEqual(domain_logits.shape, (self.batch_size, 26))

            # Verificar que no hay NaN o Inf
            self.assertFalse(torch.isnan(pd_logits).any())
            self.assertFalse(torch.isinf(pd_logits).any())
            self.assertFalse(torch.isnan(domain_logits).any())
            self.assertFalse(torch.isinf(domain_logits).any())

    def test_shared_feature_extractor(self):
        """Verificar que ambos modelos usan el mismo tipo de FeatureExtractor."""
        # Verificar que ambos tienen FeatureExtractor
        self.assertIsInstance(self.cnn2d.feature_extractor, FeatureExtractor)
        self.assertIsInstance(self.cnn2d_da.feature_extractor, FeatureExtractor)

        # Verificar que tienen la misma estructura
        cnn2d_fe = self.cnn2d.feature_extractor
        cnn2d_da_fe = self.cnn2d_da.feature_extractor

        # Verificar que ambos tienen block1 y block2
        self.assertTrue(hasattr(cnn2d_fe, "block1"))
        self.assertTrue(hasattr(cnn2d_fe, "block2"))
        self.assertTrue(hasattr(cnn2d_da_fe, "block1"))
        self.assertTrue(hasattr(cnn2d_da_fe, "block2"))

        # Verificar que tienen la misma feature_dim
        self.assertEqual(cnn2d_fe.feature_dim, cnn2d_da_fe.feature_dim)

    def test_parameter_counts(self):
        """Verificar que los conteos de parámetros son razonables."""
        cnn2d_params = sum(p.numel() for p in self.cnn2d.parameters())
        cnn2d_da_params = sum(p.numel() for p in self.cnn2d_da.parameters())

        # CNN2D_DA debe tener más parámetros que CNN2D (por el domain head)
        self.assertGreater(cnn2d_da_params, cnn2d_params)

        # Verificar que los conteos son razonables (basado en las especificaciones)
        # CNN2D: ~785K parámetros
        # CNN2D_DA: ~1.5M parámetros
        self.assertGreater(cnn2d_params, 700000)  # Al menos 700K
        self.assertLess(cnn2d_params, 900000)  # Menos de 900K

        self.assertGreater(cnn2d_da_params, 1400000)  # Al menos 1.4M
        self.assertLess(cnn2d_da_params, 1700000)  # Menos de 1.7M

    def test_model_consistency(self):
        """Verificar consistencia entre modelos."""
        # Verificar que ambos modelos procesan el mismo input
        with torch.no_grad():
            cnn2d_output = self.cnn2d(self.test_input)
            cnn2d_da_pd, _ = self.cnn2d_da(self.test_input)

            # Las salidas PD deben tener la misma forma
            self.assertEqual(cnn2d_output.shape, cnn2d_da_pd.shape)

            # Verificar que ambos modelos están en el mismo modo (train/eval)
            self.assertEqual(self.cnn2d.training, self.cnn2d_da.training)


class TestArchitectureDetails(unittest.TestCase):
    """Pruebas detalladas de arquitectura específica."""

    def test_feature_extractor_block1_details(self):
        """Verificar detalles específicos del block1 del FeatureExtractor."""
        model = CNN2D()
        block1 = model.feature_extractor.block1

        # Verificar Conv2d
        conv1 = block1[0]
        self.assertEqual(conv1.in_channels, 1)
        self.assertEqual(conv1.out_channels, 32)
        self.assertEqual(conv1.kernel_size, (3, 3))
        self.assertEqual(conv1.padding, (1, 1))

        # Verificar BatchNorm2d
        bn1 = block1[1]
        self.assertEqual(bn1.num_features, 32)

        # Verificar MaxPool2d
        pool1 = block1[3]
        self.assertEqual(pool1.kernel_size, 3)
        self.assertEqual(pool1.stride, 2)
        self.assertEqual(pool1.padding, 1)

        # Verificar Dropout2d
        dropout1 = block1[4]
        self.assertEqual(dropout1.p, 0.3)

    def test_feature_extractor_block2_details(self):
        """Verificar detalles específicos del block2 del FeatureExtractor."""
        model = CNN2D()
        block2 = model.feature_extractor.block2

        # Verificar Conv2d
        conv2 = block2[0]
        self.assertEqual(conv2.in_channels, 32)
        self.assertEqual(conv2.out_channels, 64)
        self.assertEqual(conv2.kernel_size, (3, 3))
        self.assertEqual(conv2.padding, (1, 1))

        # Verificar BatchNorm2d
        bn2 = block2[1]
        self.assertEqual(bn2.num_features, 64)

        # Verificar MaxPool2d
        pool2 = block2[3]
        self.assertEqual(pool2.kernel_size, 3)
        self.assertEqual(pool2.stride, 2)
        self.assertEqual(pool2.padding, 1)

        # Verificar Dropout2d
        dropout2 = block2[4]
        self.assertEqual(dropout2.p, 0.3)

    def test_pd_head_details(self):
        """Verificar detalles específicos del PD head."""
        model = CNN2D()
        pd_head = model.pd_head.classifier

        # Verificar Flatten
        flatten = pd_head[0]
        self.assertEqual(flatten.start_dim, 1)
        self.assertEqual(flatten.end_dim, -1)

        # Verificar primera Linear
        linear1 = pd_head[1]
        self.assertEqual(linear1.out_features, 64)

        # Verificar ReLU
        relu = pd_head[2]
        self.assertIsInstance(relu, nn.ReLU)

        # Verificar Dropout
        dropout = pd_head[3]
        self.assertEqual(dropout.p, 0.5)

        # Verificar segunda Linear (salida)
        linear2 = pd_head[4]
        self.assertEqual(linear2.out_features, 2)

    def test_domain_head_details(self):
        """Verificar detalles específicos del Domain head en CNN2D_DA."""
        model = CNN2D_DA(n_domains=26)
        domain_head = model.domain_head.classifier

        # Verificar Flatten
        flatten = domain_head[0]
        self.assertEqual(flatten.start_dim, 1)
        self.assertEqual(flatten.end_dim, -1)

        # Verificar primera Linear
        linear1 = domain_head[1]
        self.assertEqual(linear1.out_features, 64)

        # Verificar ReLU
        relu = domain_head[2]
        self.assertIsInstance(relu, nn.ReLU)

        # Verificar Dropout
        dropout = domain_head[3]
        self.assertEqual(dropout.p, 0.5)

        # Verificar segunda Linear (salida)
        linear2 = domain_head[4]
        self.assertEqual(linear2.out_features, 26)


class TestFlexibleArchitecture(unittest.TestCase):
    """Pruebas para la nueva arquitectura flexible de CNN2D."""

    def test_flexible_cnn2d_creation(self):
        """Verificar que CNN2D se puede crear con parámetros flexibles."""
        # Crear modelo con parámetros personalizados
        model = CNN2D(
            filters_1=64,
            filters_2=128,
            kernel_size_1=5,
            kernel_size_2=7,
            dense_units=32,
            p_drop_conv=0.4,
            p_drop_fc=0.6
        )
        
        # Verificar que los parámetros se aplicaron
        self.assertEqual(model.filters_1, 64)
        self.assertEqual(model.filters_2, 128)
        self.assertEqual(model.kernel_size_1, 5)
        self.assertEqual(model.kernel_size_2, 7)
        self.assertEqual(model.dense_units, 32)
        self.assertEqual(model.p_drop_conv, 0.4)
        self.assertEqual(model.p_drop_fc, 0.6)

    def test_flexible_feature_extractor(self):
        """Verificar que FeatureExtractor acepta parámetros flexibles."""
        # Crear FeatureExtractor con parámetros personalizados
        fe = FeatureExtractor(
            filters_1=64,
            filters_2=128,
            kernel_size_1=5,
            kernel_size_2=7,
            p_drop_conv=0.4
        )
        
        # Verificar parámetros
        self.assertEqual(fe.filters_1, 64)
        self.assertEqual(fe.filters_2, 128)
        self.assertEqual(fe.kernel_size_1, 5)
        self.assertEqual(fe.kernel_size_2, 7)
        self.assertEqual(fe.p_drop_conv, 0.4)

    def test_flexible_forward_pass(self):
        """Verificar que el modelo flexible hace forward pass correctamente."""
        # Crear modelo con parámetros personalizados
        model = CNN2D(
            filters_1=64,
            filters_2=128,
            kernel_size_1=5,
            kernel_size_2=7,
            dense_units=32
        )
        
        # Crear input de prueba
        test_input = torch.randn(2, 1, 65, 41)
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
            
            # Verificar dimensiones
            self.assertEqual(output.shape, (2, 2))
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_backward_compatibility(self):
        """Verificar que valores default mantienen arquitectura original."""
        # Crear modelo con valores default
        model_default = CNN2D()
        
        # Verificar que usa valores originales
        self.assertEqual(model_default.filters_1, 32)
        self.assertEqual(model_default.filters_2, 64)
        self.assertEqual(model_default.kernel_size_1, 3)
        self.assertEqual(model_default.kernel_size_2, 3)
        self.assertEqual(model_default.dense_units, 64)
        self.assertEqual(model_default.p_drop_conv, 0.3)
        self.assertEqual(model_default.p_drop_fc, 0.5)

    def test_different_kernel_sizes(self):
        """Verificar que diferentes kernel sizes funcionan correctamente."""
        # Probar diferentes combinaciones de kernel sizes
        kernel_combinations = [
            (4, 5), (6, 7), (8, 9), (3, 3), (5, 5)
        ]
        
        for k1, k2 in kernel_combinations:
            with self.subTest(kernel_size_1=k1, kernel_size_2=k2):
                model = CNN2D(
                    kernel_size_1=k1,
                    kernel_size_2=k2
                )
                
                # Verificar que se creó correctamente
                self.assertEqual(model.kernel_size_1, k1)
                self.assertEqual(model.kernel_size_2, k2)
                
                # Verificar forward pass
                test_input = torch.randn(1, 1, 65, 41)
                with torch.no_grad():
                    output = model(test_input)
                    self.assertEqual(output.shape, (1, 2))

    def test_different_filter_sizes(self):
        """Verificar que diferentes tamaños de filtros funcionan correctamente."""
        # Probar diferentes combinaciones de filtros
        filter_combinations = [
            (32, 64), (64, 128), (128, 256), (16, 32)
        ]
        
        for f1, f2 in filter_combinations:
            with self.subTest(filters_1=f1, filters_2=f2):
                model = CNN2D(
                    filters_1=f1,
                    filters_2=f2
                )
                
                # Verificar que se creó correctamente
                self.assertEqual(model.filters_1, f1)
                self.assertEqual(model.filters_2, f2)
                
                # Verificar forward pass
                test_input = torch.randn(1, 1, 65, 41)
                with torch.no_grad():
                    output = model(test_input)
                    self.assertEqual(output.shape, (1, 2))


if __name__ == "__main__":
    # Configurar el runner de pruebas
    unittest.main(verbosity=2)
