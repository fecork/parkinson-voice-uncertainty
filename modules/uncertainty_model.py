"""
Modelo CNN con estimación de incertidumbre epistémica y aleatoria.

Arquitectura:
- Backbone: CNN con bloques Conv2D → BN → ReLU → MaxPool → Dropout
- Dos cabezas:
  * Cabeza A: predicción (logits)
  * Cabeza B: ruido de datos (s_logit para σ²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Dropout activo en modo eval (para MC Dropout)."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class MCDropout2d(nn.Module):
    """Dropout2d activo en modo eval (para MC Dropout)."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout2d(x, p=self.p, training=True)


class UncertaintyCNN(nn.Module):
    """
    CNN con dos cabezas para estimación de incertidumbre.

    Args:
        n_classes: Número de clases
        p_drop_conv: Probabilidad de dropout en capas convolucionales
        p_drop_fc: Probabilidad de dropout en capas fully connected
        input_shape: Tupla (H, W) del tamaño de entrada
        s_min: Valor mínimo para clamp de s_logit
        s_max: Valor máximo para clamp de s_logit
    """

    def __init__(
        self,
        n_classes=2,
        p_drop_conv=0.25,
        p_drop_fc=0.25,
        input_shape=(65, 41),
        s_min=-10.0,
        s_max=3.0,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.s_min = s_min
        self.s_max = s_max

        # Feature extractor (backbone)
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            MCDropout2d(p=p_drop_conv),
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            MCDropout2d(p=p_drop_conv),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Calcular el tamaño del feature vector
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            feat_size = self.features(dummy_input).shape[1]

        # Cabeza A: Predicción (logits)
        self.head_logits = nn.Sequential(
            nn.Linear(feat_size, 64),
            nn.ReLU(inplace=True),
            MCDropout(p=p_drop_fc),
            nn.Linear(64, n_classes),
        )

        # Cabeza B: Ruido de datos (s_logit para σ²)
        self.head_noise = nn.Sequential(
            nn.Linear(feat_size, 64),
            nn.ReLU(inplace=True),
            MCDropout(p=p_drop_fc),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            logits: Logits de predicción [B, C]
            s_logit: Log-varianza (clampeada) [B, C]
        """
        # Extract features
        features = self.features(x)

        # Cabeza A: logits
        logits = self.head_logits(features)

        # Cabeza B: s_logit (log-varianza)
        s_logit = self.head_noise(features)
        s_logit = torch.clamp(s_logit, min=self.s_min, max=self.s_max)

        return logits, s_logit

    def predict_with_uncertainty(self, x, n_samples=30):
        """
        Predicción con estimación de incertidumbre usando MC Dropout.

        Args:
            x: Input tensor [B, 1, H, W]
            n_samples: Número de pasadas MC

        Returns:
            dict con:
                - pred: Predicción final [B]
                - probs_mean: Probabilidades promedio [B, C]
                - entropy_total: Entropía predictiva [B]
                - epistemic: Incertidumbre epistémica (BALD) [B]
                - aleatoric: Incertidumbre aleatoria promedio [B]
        """
        self.eval()  # Pero MCDropout sigue activo

        all_probs = []
        all_sigma2 = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, s_logit = self(x)
                probs = F.softmax(logits, dim=1)
                sigma2 = torch.exp(s_logit)

                all_probs.append(probs)
                all_sigma2.append(sigma2)

        # Stack: [n_samples, B, C]
        all_probs = torch.stack(all_probs)
        all_sigma2 = torch.stack(all_sigma2)

        # Probabilidades promedio
        probs_mean = all_probs.mean(dim=0)  # [B, C]

        # Predicción final
        pred = probs_mean.argmax(dim=1)  # [B]

        # Entropía total (predictiva)
        eps = 1e-12
        entropy_total = -(probs_mean * torch.log(probs_mean + eps)).sum(dim=1)  # [B]

        # Entropía de cada muestra MC
        entropy_samples = -(all_probs * torch.log(all_probs + eps)).sum(
            dim=2
        )  # [n_samples, B]

        # BALD (epistémica pura)
        epistemic = entropy_total - entropy_samples.mean(dim=0)  # [B]

        # Aleatoria (promedio de σ² agregado por muestra)
        aleatoric = all_sigma2.mean(dim=(0, 2))  # [B]

        return {
            "pred": pred,
            "probs_mean": probs_mean,
            "confidence": probs_mean.max(dim=1)[0],
            "entropy_total": entropy_total,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }


def print_uncertainty_model_summary(model):
    """Imprime resumen del modelo con incertidumbre."""
    print("\n" + "=" * 60)
    print("RESUMEN DEL MODELO CON INCERTIDUMBRE")
    print("=" * 60)
    print(model)
    print("\n" + "-" * 60)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print("-" * 60 + "\n")
