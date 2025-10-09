"""
CNN 2D Model Module
====================
Modelo CNN 2D compacto para clasificación PD vs HC.
Diseñado para espectrogramas Mel (65×41) con dropout para MC Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# MODELO CNN 2D
# ============================================================


class CNN2D(nn.Module):
    """
    CNN 2D ligera para clasificación binaria PD vs HC.

    Arquitectura:
        - Conv(32, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout
        - Conv(64, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout
        - Flatten → Dense(64) → ReLU → Dropout → Dense(2)

    Input shape: (B, 1, 65, 41)
    Output: (B, 2) logits
    """

    def __init__(
        self, n_classes: int = 2, p_drop_conv: float = 0.3, p_drop_fc: float = 0.5
    ):
        """
        Args:
            n_classes: Número de clases (2 para binario)
            p_drop_conv: Dropout en capas convolucionales
            p_drop_fc: Dropout en capas fully connected
        """
        super().__init__()

        self.n_classes = n_classes
        self.p_drop_conv = p_drop_conv
        self.p_drop_fc = p_drop_fc

        # Feature extractor
        self.features = nn.Sequential(
            # Bloque 1: (B, 1, 65, 41) → (B, 32, 32, 20)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p_drop_conv),
            # Bloque 2: (B, 32, 32, 20) → (B, 64, 16, 10)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p_drop_conv),
        )

        # Calcular tamaño después de convoluciones
        # Input: 65×41 → después de 2 maxpools(2×2): 16×10
        self.feature_dim = 64 * 16 * 10

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop_fc),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            Logits (B, n_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass retornando features para Grad-CAM.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            features: Feature maps de última conv (B, 64, 16, 10)
            logits: Logits (B, n_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return features, logits


# ============================================================
# MC DROPOUT
# ============================================================


def enable_dropout(model: nn.Module):
    """
    Activa dropout durante inferencia para MC Dropout.

    Args:
        model: Modelo PyTorch
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Realiza inferencia con MC Dropout.

    Args:
        model: Modelo PyTorch
        x: Input tensor (B, 1, 65, 41)
        n_samples: Número de forward passes estocásticos
        device: Device para cómputo

    Returns:
        p_mean: Probabilidades promedio (B, n_classes)
        p_std: Desviación estándar (B, n_classes)
        entropy: Entropía predictiva (B,)
    """
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device)

    # Activar dropout
    model.train()
    enable_dropout(model)

    # Realizar múltiples forward passes
    all_probs = []
    for _ in range(n_samples):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    # Stack: (n_samples, B, n_classes)
    all_probs = torch.stack(all_probs, dim=0)

    # Estadísticas
    p_mean = all_probs.mean(dim=0)  # (B, n_classes)
    p_std = all_probs.std(dim=0)  # (B, n_classes)

    # Entropía predictiva
    entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-10), dim=1)  # (B,)

    # Volver a modo eval
    model.eval()

    return p_mean, p_std, entropy


# ============================================================
# GRAD-CAM
# ============================================================


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping para explicabilidad.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Modelo PyTorch
            target_layer: Capa objetivo (típicamente última conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
    ):
        """Hook para guardar activaciones."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ):
        """Hook para guardar gradientes."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self, x: torch.Tensor, target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Genera mapa Grad-CAM.

        Args:
            x: Input tensor (B, 1, 65, 41)
            target_class: Clase objetivo (None = predicción)

        Returns:
            CAM normalizado (B, H_orig, W_orig)
        """
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward pass
        logits = self.model(x)

        # Si no se especifica clase, usar la predicha
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()

        # Crear one-hot de la clase target
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * x.size(0))

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        # Backprop
        logits.backward(gradient=one_hot, retain_graph=True)

        # Calcular pesos: GAP de gradientes
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ReLU (solo activaciones positivas)
        cam = F.relu(cam)

        # Normalizar por batch
        B, _, H, W = cam.shape
        cam = cam.view(B, -1)
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        cam = cam.view(B, 1, H, W)

        # Resize a tamaño original
        cam = F.interpolate(cam, size=(65, 41), mode="bilinear", align_corners=False)

        return cam.squeeze(1)  # (B, 65, 41)


def get_last_conv_layer(model: CNN2D) -> nn.Module:
    """
    Obtiene la última capa convolucional del modelo.

    Args:
        model: Modelo CNN2D

    Returns:
        Última capa Conv2d
    """
    # En nuestro modelo, la última conv es features[5]
    # (después de BN y antes de MaxPool)
    for i, module in enumerate(model.features):
        if isinstance(module, nn.Conv2d):
            last_conv = module

    return last_conv


# ============================================================
# UTILIDADES
# ============================================================


def count_parameters(model: nn.Module) -> int:
    """
    Cuenta parámetros entrenables del modelo.

    Args:
        model: Modelo PyTorch

    Returns:
        Número de parámetros
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """
    Imprime resumen del modelo.

    Args:
        model: Modelo PyTorch
    """
    print("\n" + "=" * 60)
    print("RESUMEN DEL MODELO")
    print("=" * 60)
    print(model)
    print("\n" + "-" * 60)
    print(f"Parámetros totales: {count_parameters(model):,}")
    print("-" * 60 + "\n")


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


if __name__ == "__main__":
    # Crear modelo
    model = CNN2D(n_classes=2, p_drop_conv=0.3, p_drop_fc=0.5)
    print_model_summary(model)

    # Test forward pass
    x = torch.randn(4, 1, 65, 41)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test MC Dropout
    print("\n" + "=" * 60)
    print("TEST MC DROPOUT")
    print("=" * 60)
    p_mean, p_std, entropy = mc_dropout_predict(model, x, n_samples=10)
    print(f"Probabilidades promedio: {p_mean.shape}")
    print(f"Desviación estándar: {p_std.shape}")
    print(f"Entropía: {entropy.shape}")
    print(f"Ejemplo - P(HC): {p_mean[0, 0]:.3f} ± {p_std[0, 0]:.3f}")
    print(f"Ejemplo - P(PD): {p_mean[0, 1]:.3f} ± {p_std[0, 1]:.3f}")
    print(f"Ejemplo - Entropy: {entropy[0]:.3f}")

    # Test Grad-CAM
    print("\n" + "=" * 60)
    print("TEST GRAD-CAM")
    print("=" * 60)
    last_conv = get_last_conv_layer(model)
    gradcam = GradCAM(model, last_conv)
    cam = gradcam.generate_cam(x)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
