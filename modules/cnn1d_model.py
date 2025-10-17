"""
CNN 1D Model Module with Domain Adaptation
============================================
Modelo CNN 1D con atención temporal y Domain Adaptation para
clasificación PD vs HC. Diseñado para espectrogramas Mel (65×41)
procesados como secuencias temporales.

Reference:
    Ibarra et al. (2023) "Towards a Corpus (and Language)-Independent
    Screening of Parkinson's Disease from Voice and Speech through
    Domain Adaptation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# GRADIENT REVERSAL LAYER (reutilizado de cnn_model.py)
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
# CNN 1D CON DOMAIN ADAPTATION
# ============================================================


class CNN1D_DA(nn.Module):
    """
    CNN 1D con Domain Adaptation y atención temporal.

    Arquitectura según Ibarra et al. (2023):
        - 3 bloques Conv1D con kernels [5, 11, 21]
        - Atención temporal sobre última conv (split + softmax)
        - Dual-head: PD detector y Domain detector con GRL

    Input shape: (B, F=65, T=41)
    Output: (logits_pd, logits_domain) o (logits_pd, logits_domain, embeddings)
    """

    def __init__(
        self,
        in_ch: int = 65,
        c1: int = 64,
        c2: int = 128,
        c3: int = 128,
        p_drop: float = 0.3,
        num_pd: int = 2,
        num_domains: int = 4,
    ):
        """
        Args:
            in_ch: Canales de entrada (freq bins, default 65)
            c1: Canales bloque 1 (default 64)
            c2: Canales bloque 2 (default 128)
            c3: Canales bloque 3 (default 128)
            p_drop: Probabilidad de dropout (default 0.3)
            num_pd: Clases PD (2: HC/PD)
            num_domains: Número de dominios/corpus
        """
        super().__init__()

        self.in_ch = in_ch
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.p_drop = p_drop
        self.num_pd = num_pd
        self.num_domains = num_domains

        # Block 1: Conv1D(k=5) → BN → ReLU → Dropout → MaxPool(k=6)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool1d(kernel_size=6),
        )

        # Block 2: Conv1D(k=11) → BN → ReLU → Dropout → MaxPool(k=6)
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool1d(kernel_size=6),
        )

        # Block 3: Conv1D(k=21) → BN → ReLU → Dropout (SIN MaxPool)
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=21, padding=10, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # Atención temporal (split en mitades)
        self.half = c3 // 2  # 64 para c3=128

        # PD Head: Linear(emb, 64) → ReLU → Dropout → Linear(64, 2)
        self.pd_head = nn.Sequential(
            nn.Linear(self.half, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, num_pd),
        )

        # Domain Head: misma estructura → n_domains
        self.dom_head = nn.Sequential(
            nn.Linear(self.half, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, num_domains),
        )

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=0.0)

    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass con atención temporal y dual-head.

        Args:
            x: Input tensor (B, F=65, T=41)
            return_embeddings: Si retornar embeddings para t-SNE

        Returns:
            logits_pd: Logits para clasificación PD (B, 2)
            logits_domain: Logits para clasificación de dominio (B, n_domains)
            embeddings: (opcional) Embeddings atendidos (B, 64)
        """
        # Feature extraction con convs 1D
        y = self.block1(x)  # [B, 64, T/6]
        y = self.block2(y)  # [B, 128, T/36]
        y = self.block3(y)  # [B, 128, T']

        # Atención temporal (paper: split + softmax temporal)
        A, V = y[:, :self.half, :], y[:, self.half:, :]
        alpha = F.softmax(A, dim=-1)  # [B, 64, T'] - softmax temporal
        z = (alpha * V).sum(dim=-1)  # [B, 64] - embedding atendido

        # Dual heads
        pd_logits = self.pd_head(z)
        dom_reversed = self.grl(z)
        dom_logits = self.dom_head(dom_reversed)

        if return_embeddings:
            return pd_logits, dom_logits, z
        return pd_logits, dom_logits, None

    def set_lambda(self, lambda_: float):
        """
        Actualiza el factor lambda de la GRL.

        Args:
            lambda_: Nuevo valor de lambda (0 a 1)
        """
        self.grl.set_lambda(lambda_)


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
    print("\n" + "=" * 70)
    print("ARQUITECTURA DEL MODELO CNN1D_DA")
    print("=" * 70)
    print(model)
    print("\n" + "-" * 70)
    print(f"Parámetros totales: {count_parameters(model):,}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print("-" * 70 + "\n")


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: CNN1D_DA")
    print("=" * 70)

    # Crear modelo
    model = CNN1D_DA(
        in_ch=65, c1=64, c2=128, c3=128, p_drop=0.3, num_pd=2, num_domains=26
    )
    print_model_summary(model)

    # Test forward pass
    x = torch.randn(4, 65, 41)  # [B, F, T]
    print(f"\nInput shape: {x.shape}")

    # Test sin embeddings
    logits_pd, logits_domain, _ = model(x, return_embeddings=False)
    print(f"Output PD shape: {logits_pd.shape} (2 clases: HC/PD)")
    print(f"Output Domain shape: {logits_domain.shape} (26 dominios)")

    # Test con embeddings
    logits_pd2, logits_domain2, embeddings = model(x, return_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape} (B, 64)")

    # Test lambda scheduling
    print("\n" + "=" * 70)
    print("TEST GRADIENT REVERSAL LAYER")
    print("=" * 70)
    print(f"Lambda inicial: {model.grl.lambda_}")
    model.set_lambda(0.5)
    print(f"Lambda actualizado: {model.grl.lambda_}")

    # Forward con nuevo lambda
    logits_pd3, logits_domain3, _ = model(x)
    print("Forward pass con lambda=0.5 OK")

    print("\n[OK] Todos los tests pasaron correctamente")
