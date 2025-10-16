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


# ============================================================
# DOMAIN ADAPTATION ARCHITECTURE
# ============================================================


class GradientReversalFunction(torch.autograd.Function):
    """
    Función para inversión de gradiente (Gradient Reversal Layer).

    Durante forward: pasa x sin cambios
    Durante backward: invierte el gradiente y lo multiplica por lambda

    Reference:
        Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation"
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


class FeatureExtractor(nn.Module):
    """
    Extractor de características compartido para Domain Adaptation.

    Arquitectura (según Ibarra et al. 2023):
        - Bloque 1: Conv2d(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
        - Bloque 2: Conv2d(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout

    Input shape: (B, 1, 65, 41)
    Output shape: (B, 64, H', W')
    """

    def __init__(self, p_drop_conv: float = 0.3):
        """
        Args:
            p_drop_conv: Probabilidad de dropout en capas convolucionales
        """
        super().__init__()
        self.p_drop_conv = p_drop_conv

        # Bloque 1: (B, 1, 65, 41) → (B, 32, 33, 21)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # MaxPool 3×3 (paper)
            nn.Dropout2d(p=p_drop_conv),
        )

        # Bloque 2: (B, 32, 33, 21) → (B, 64, 17, 11)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # MaxPool 3×3 (paper)
            nn.Dropout2d(p=p_drop_conv),
        )

        # Calcular dimensión de features
        # Input: 65×41
        # Después MaxPool1(3×3, s=2, p=1): 33×21
        # Después MaxPool2(3×3, s=2, p=1): 17×11
        self.feature_dim = 64 * 17 * 11

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            Feature maps (B, 64, H', W')
        """
        x = self.block1(x)
        x = self.block2(x)
        return x


class ClassifierHead(nn.Module):
    """
    Cabeza de clasificación fully-connected.

    Arquitectura:
        Linear(feature_dim, hidden_dim) → ReLU → Dropout → Linear(n_classes)
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


class CNN2D_DA(nn.Module):
    """
    CNN 2D con Domain Adaptation para clasificación de Parkinson.

    Arquitectura dual-head con Gradient Reversal Layer (GRL):
        - Feature Extractor (compartido)
        - PD Head: Clasifica Parkinson vs Healthy (2 clases)
        - Domain Head: Clasifica dominio con GRL (n_domains clases)

    Reference:
        Ibarra et al. (2023) "Towards a Corpus (and Language)-Independent
        Screening of Parkinson's Disease from Voice and Speech through
        Domain Adaptation"

    Input shape: (B, 1, 65, 41)
    Output: (logits_pd, logits_domain)
    """

    def __init__(
        self,
        n_domains: int = 26,
        p_drop_conv: float = 0.3,
        p_drop_fc: float = 0.5,
    ):
        """
        Args:
            n_domains: Número de dominios (típicamente 26 = 13 archivos × 2 clases)
            p_drop_conv: Dropout en capas convolucionales
            p_drop_fc: Dropout en capas fully connected
        """
        super().__init__()

        self.n_domains = n_domains
        self.p_drop_conv = p_drop_conv
        self.p_drop_fc = p_drop_fc

        # Extractor de características compartido
        self.feature_extractor = FeatureExtractor(p_drop_conv=p_drop_conv)
        feature_dim = self.feature_extractor.feature_dim

        # Cabeza PD (tarea principal)
        self.pd_head = ClassifierHead(
            feature_dim=feature_dim,
            hidden_dim=64,
            n_classes=2,  # Binario: HC vs PD
            p_drop_fc=p_drop_fc,
        )

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=1.0)

        # Cabeza de dominio (tarea auxiliar)
        self.domain_head = ClassifierHead(
            feature_dim=feature_dim,
            hidden_dim=64,
            n_classes=n_domains,
            p_drop_fc=p_drop_fc,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass con dual-head.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            logits_pd: Logits para clasificación PD (B, 2)
            logits_domain: Logits para clasificación de dominio (B, n_domains)
        """
        # Extracción de características
        features = self.feature_extractor(x)

        # Cabeza PD (sin GRL)
        logits_pd = self.pd_head(features)

        # Cabeza de dominio (con GRL)
        features_reversed = self.grl(features)
        logits_domain = self.domain_head(features_reversed)

        return logits_pd, logits_domain

    def set_lambda(self, lambda_: float):
        """
        Actualiza el factor lambda de la GRL.

        Args:
            lambda_: Nuevo valor de lambda (0 a 1)
        """
        self.grl.set_lambda(lambda_)


def get_last_conv_layer_da(model: CNN2D_DA) -> nn.Module:
    """
    Obtiene la última capa convolucional del modelo DA.

    Args:
        model: Modelo CNN2D_DA

    Returns:
        Última capa Conv2d del feature extractor
    """
    # La última conv está en block2 del feature extractor
    for module in model.feature_extractor.block2:
        if isinstance(module, nn.Conv2d):
            return module

    # Fallback: buscar en block1
    for module in model.feature_extractor.block1:
        if isinstance(module, nn.Conv2d):
            last_conv = module

    return last_conv


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: CNN2D BASELINE")
    print("=" * 70)

    # Crear modelo baseline
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

    print("\n" + "=" * 70)
    print("TEST 2: CNN2D_DA (DOMAIN ADAPTATION)")
    print("=" * 70)

    # Crear modelo DA
    model_da = CNN2D_DA(n_domains=26, p_drop_conv=0.3, p_drop_fc=0.5)
    print_model_summary(model_da)

    # Test forward pass
    logits_pd, logits_domain = model_da(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output PD shape: {logits_pd.shape} (2 clases: HC/PD)")
    print(f"Output Domain shape: {logits_domain.shape} (26 dominios)")

    # Test lambda scheduling
    print("\n" + "=" * 60)
    print("TEST GRADIENT REVERSAL LAYER")
    print("=" * 60)
    print(f"Lambda inicial: {model_da.grl.lambda_}")
    model_da.set_lambda(0.5)
    print(f"Lambda actualizado: {model_da.grl.lambda_}")

    # Test forward con nuevo lambda
    logits_pd2, logits_domain2 = model_da(x)
    print("Forward pass con lambda=0.5 OK")

    print("\n✅ Todos los tests pasaron correctamente")
