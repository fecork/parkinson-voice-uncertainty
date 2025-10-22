"""
Capas Compartidas entre Modelos
================================
Componentes reutilizables: FeatureExtractor, GRL, ClassifierHead.

Estos componentes son usados por:
- CNN2D / CNN2D_DA
- CNN1D_DA
- Time-CNN-BiLSTM-DA
"""

import torch
import torch.nn as nn
from typing import Tuple


# ============================================================
# GRADIENT REVERSAL LAYER
# ============================================================


class GradientReversalFunction(torch.autograd.Function):
    """
    Función para inversión de gradiente (Gradient Reversal Layer).

    Durante forward: pasa x sin cambios
    Durante backward: invierte el gradiente y lo multiplica por lambda

    Reference:
        Ganin & Lempitsky (2015)
        "Unsupervised Domain Adaptation by Backpropagation"
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass - identidad."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - invierte gradiente."""
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) para Domain Adaptation.

    Invierte los gradientes durante backpropagation para hacer
    las features invariantes al dominio.

    Attributes:
        lambda_: Factor de inversión de gradiente (0 a 1)
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Factor inicial de inversión (default: 1.0)
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor (idéntico a input en forward)
        """
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Actualiza el factor lambda."""
        self.lambda_ = lambda_


# ============================================================
# FEATURE EXTRACTOR (2D CNN)
# ============================================================


class FeatureExtractor(nn.Module):
    """
    Extractor de características compartido para Domain Adaptation.

    Arquitectura (según Ibarra et al. 2023):
        - Bloque 1: Conv2d(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
        - Bloque 2: Conv2d(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout

    Input shape: (B, 1, H, W)
    Output shape: (B, 64, H', W')

    Usado por:
        - CNN2D / CNN2D_DA
        - Time-CNN-BiLSTM-DA (time-distributed)
    """

    def __init__(
        self,
        p_drop_conv: float = 0.3,
        input_shape: Tuple[int, int] = (65, 41),
    ):
        """
        Args:
            p_drop_conv: Probabilidad de dropout en capas convolucionales
            input_shape: Dimensiones de entrada (H, W)
        """
        super().__init__()
        self.p_drop_conv = p_drop_conv
        self.input_shape = input_shape

        # Bloque 1: (B, 1, H, W) → (B, 32, H', W')
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3×3 (paper)
            nn.Dropout2d(p=p_drop_conv),
        )

        # Bloque 2: (B, 32, H', W') → (B, 64, H'', W'')
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3×3 (paper)
            nn.Dropout2d(p=p_drop_conv),
        )

        # Calcular dimensión de features dinámicamente
        self.feature_dim = self._calculate_feature_dim(input_shape)

    def _calculate_feature_dim(self, input_shape: Tuple[int, int]) -> int:
        """
        Calcula la dimensión de features después de capas convolucionales.

        Args:
            input_shape: Dimensiones de entrada (H, W)

        Returns:
            Dimensión aplanada de las features
        """
        # Crear tensor dummy y pasarlo por los bloques
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            x = self.block1(dummy_input)
            x = self.block2(x)
            # Retornar tamaño aplanado (C * H * W)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, H, W)

        Returns:
            Feature maps (B, 64, H', W')
        """
        x = self.block1(x)
        x = self.block2(x)
        return x


# ============================================================
# CLASSIFIER HEAD
# ============================================================


class ClassifierHead(nn.Module):
    """
    Cabeza de clasificación fully-connected.

    Arquitectura:
        Linear(feature_dim, hidden_dim) → ReLU → Dropout → Linear(n_classes)

    Usado por:
        - CNN2D / CNN2D_DA (PD y Domain heads)
        - Time-CNN-BiLSTM-DA (PD y Domain heads)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        n_classes: int,
        p_drop_fc: float = 0.5,
    ):
        """
        Args:
            feature_dim: Dimensión de entrada (features aplanadas)
            hidden_dim: Dimensión de capa oculta
            n_classes: Número de clases de salida
            p_drop_fc: Probabilidad de dropout
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop_fc),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature maps (B, C, H, W)

        Returns:
            Logits (B, n_classes)
        """
        return self.classifier(x)

