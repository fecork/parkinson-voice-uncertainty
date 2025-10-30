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

# Importar componentes compartidos
from ..common.layers import (
    FeatureExtractor,
    GradientReversalFunction,
    GradientReversalLayer,
    ClassifierHead,
)


# ============================================================
# MODELO CNN 2D
# ============================================================


class CNN2D(nn.Module):
    """
    CNN 2D para clasificación binaria PD vs HC (sin Domain Adaptation).

    Arquitectura (igual backbone que CNN2D_DA):
        - Bloque 1: Conv2d(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
        - Bloque 2: Conv2d(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
        - PD Head: Flatten → Dense(64) → ReLU → Dropout → Dense(2)

    Input shape: (B, 1, 65, 41)
    Output: (B, 2) logits

    Note:
        Usa el mismo FeatureExtractor que CNN2D_DA para permitir
        comparación justa entre arquitecturas con y sin DA.
    """

    def __init__(
        self,
        n_classes: int = 2,
        p_drop_conv: float = 0.3,
        p_drop_fc: float = 0.5,
        input_shape: Tuple[int, int] = (65, 41),
        filters_1: int = 32,
        filters_2: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        dense_units: int = 64,
    ):
        """
        Args:
            n_classes: Número de clases (2 para binario)
            p_drop_conv: Dropout en capas convolucionales
            p_drop_fc: Dropout en capas fully connected
            input_shape: Dimensiones de entrada (H, W)
            filters_1: Número de filtros en primera capa convolucional
            filters_2: Número de filtros en segunda capa convolucional
            kernel_size_1: Tamaño del kernel en primera capa convolucional
            kernel_size_2: Tamaño del kernel en segunda capa convolucional
            dense_units: Número de unidades en capa densa oculta
        """
        super().__init__()

        self.n_classes = n_classes
        self.p_drop_conv = p_drop_conv
        self.p_drop_fc = p_drop_fc
        self.input_shape = input_shape
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.dense_units = dense_units

        # Feature extractor (compartido con CNN2D_DA)
        self.feature_extractor = FeatureExtractor(
            p_drop_conv=p_drop_conv,
            input_shape=input_shape,
            filters_1=filters_1,
            filters_2=filters_2,
            kernel_size_1=kernel_size_1,
            kernel_size_2=kernel_size_2,
        )

        # PD Head (clasificación Parkinson)
        self.pd_head = ClassifierHead(
            feature_dim=self.feature_extractor.feature_dim,
            hidden_dim=dense_units,
            n_classes=n_classes,
            p_drop_fc=p_drop_fc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            Logits (B, n_classes)
        """
        features = self.feature_extractor(x)
        logits = self.pd_head(features)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass retornando features para Grad-CAM.

        Args:
            x: Input tensor (B, 1, 65, 41)

        Returns:
            features: Feature maps de última conv (B, 64, H', W')
            logits: Logits (B, n_classes)
        """
        features = self.feature_extractor(x)
        logits = self.pd_head(features)
        return features, logits

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Crea una instancia usando los mismos args/defectos del __init__ (sin duplicarlos)."""
        sig = inspect.signature(cls.__init__)
        valid = {k for k in sig.parameters if k != "self"}
        kwargs = {k: v for k, v in config.items() if k in valid}
        return cls(**kwargs)  # el __init__ completa defaults

    def get_config(self) -> Dict[str, Any]:
        """Config efectiva de ESTA instancia (sirve para recrearla 1:1)."""
        return {
            "n_classes": self.n_classes,
            "p_drop_conv": self.p_drop_conv,
            "p_drop_fc": self.p_drop_fc,
            "input_shape": self.input_shape,
            "filters_1": self.filters_1,
            "filters_2": self.filters_2,
            "kernel_size_1": self.kernel_size_1,
            "kernel_size_2": self.kernel_size_2,
            "dense_units": self.dense_units,
        }

    def new_like(self, **overrides):
        """
        Crea una NUEVA instancia con la misma config que esta,
        permitiendo overrides puntuales.
        """
        cfg = self.get_config()
        cfg.update(overrides)
        return type(self).from_config(cfg)

    def to_builder(self, **overrides):
        """
        Devuelve un CALLABLE (sin args) que crea nuevas instancias idénticas.
        Útil para pasar a run_all_checks(build_model=...).
        """
        cfg = self.get_config()
        cfg.update(overrides)

        def _factory():
            return type(self).from_config(cfg)

        return _factory


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
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
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
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

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
# UTILIDADES
# ============================================================


# Funciones utilitarias movidas a modules.models.common.layers


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


# ============================================================
# DOMAIN ADAPTATION ARCHITECTURE
# ============================================================
# NOTA: FeatureExtractor, GRL y ClassifierHead ahora están en
# modules/models/common/layers.py para reutilización


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
        input_shape: Tuple[int, int] = (65, 41),
    ):
        """
        Args:
            n_domains: Número de dominios (26 = 13 archivos × 2 clases)
            p_drop_conv: Dropout en capas convolucionales
            p_drop_fc: Dropout en capas fully connected
            input_shape: Dimensiones de entrada (H, W)
        """
        super().__init__()

        self.n_domains = n_domains
        self.p_drop_conv = p_drop_conv
        self.p_drop_fc = p_drop_fc
        self.input_shape = input_shape

        # Extractor de características compartido
        self.feature_extractor = FeatureExtractor(
            p_drop_conv=p_drop_conv, input_shape=input_shape
        )
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

    print("\n[OK] Todos los tests pasaron correctamente")
