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


class MCDropout2d(nn.Dropout2d):
    """Dropout2d activo en eval (para MC Dropout)."""

    def forward(self, x):
        return F.dropout2d(x, self.p, training=True)


class MCDropout(nn.Dropout):
    """Dropout activo en eval (para MC Dropout)."""

    def forward(self, x):
        return F.dropout(x, self.p, training=True)


class UncertaintyCNN(nn.Module):
    """
    CNN con dos cabezas para estimación de incertidumbre.
    Backbone: 2× [Conv2d -> BN -> ReLU -> MaxPool(3,2,1) -> MCDropout]
    Cabeza A: fc_logits (predicción)
    Cabeza B: fc_slog (incertidumbre aleatoria, log σ² por clase)
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
        self.s_min = s_min
        self.s_max = s_max

        # Backbone: 2 bloques Conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            MCDropout2d(p_drop_conv),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            MCDropout2d(p_drop_conv),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop_fc = MCDropout(p_drop_fc)

        # Cabezas A y B
        self.fc_logits = nn.Linear(64, n_classes)
        self.fc_slog = nn.Linear(64, n_classes)

    def forward(self, x):
        """Forward pass retorna logits y s_logit."""
        h = self.block2(self.block1(x))
        h = self.gap(h).flatten(1)
        h = self.drop_fc(h)
        logits = self.fc_logits(h)
        s_logit = torch.clamp(self.fc_slog(h), self.s_min, self.s_max)
        return logits, s_logit

    def predict_with_uncertainty(self, x, n_samples=30):
        """
        Predicción con incertidumbre usando MC Dropout + ruido gaussiano.

        Implementa decomposición de Kendall & Gal (2017):
        - H_total = H[p̄]
        - Aleatoric = E_t[H[p_t]]  (con ruido gaussiano en logits)
        - Epistemic = H_total - Aleatoric  (BALD)

        Args:
            x: Input tensor [B, 1, H, W]
            n_samples: Número de pasadas MC (T_test)

        Returns:
            dict con pred, probs_mean, confidence, entropy_total,
                     epistemic, aleatoric, sigma2_mean
        """
        self.eval()  # MCDropout sigue activo
        eps = 1e-12

        all_probs = []
        all_entropies = []
        all_sigma2_mean = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, s_logit = self(x)  # [B, C]

                # Inyectar ruido gaussiano en logits (aleatoric)
                sigma = torch.exp(0.5 * s_logit)  # σ = exp(0.5 * log σ²)
                eps_noise = torch.randn_like(logits)
                logits_t = logits + sigma * eps_noise  # x̂_t = logits + σ⊙ε

                # Probabilidades con ruido
                probs_t = F.softmax(logits_t, dim=1)  # [B, C]

                # Entropía condicional H[p_t]
                H_t = -(probs_t * torch.log(probs_t + eps)).sum(dim=1)  # [B]

                all_probs.append(probs_t)
                all_entropies.append(H_t)

                # Estadística auxiliar: σ² promedio
                all_sigma2_mean.append(torch.exp(s_logit).mean(dim=1))  # [B]

        # Stack: [n_samples, B, *]
        all_probs = torch.stack(all_probs)  # [T, B, C]
        all_entropies = torch.stack(all_entropies)  # [T, B]
        all_sigma2_mean = torch.stack(all_sigma2_mean)  # [T, B]

        # Probabilidades promedio
        p_mean = all_probs.mean(dim=0)  # [B, C]

        # Predicción final
        pred = p_mean.argmax(dim=1)  # [B]
        confidence = p_mean.max(dim=1)[0]  # [B]

        # Decomposición de Kendall & Gal
        H_total = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)  # H[p̄]
        H_cond = all_entropies.mean(dim=0)  # E_t[H[p_t]] = ALEATORIC
        epistemic = H_total - H_cond  # BALD = EPISTEMIC

        # Estadística auxiliar (para monitoreo de σ²)
        sigma2_mean = all_sigma2_mean.mean(dim=0)  # [B]

        return {
            "pred": pred,
            "probs_mean": p_mean,
            "confidence": confidence,
            "entropy_total": H_total,
            "epistemic": epistemic,  # I[y,w|x,D]
            "aleatoric": H_cond,  # E_w[H[y|x,w]] ← CORREGIDO
            "sigma2_mean": sigma2_mean,  # Auxiliar: mean σ²
        }


def print_uncertainty_model_summary(model, sample_input_shape=(2, 1, 65, 41)):
    """
    Imprime resumen compacto del modelo con incertidumbre.

    Args:
        model: Modelo UncertaintyCNN
        sample_input_shape: Shape de ejemplo (B, C, H, W)
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧩 Parámetros entrenables: {n_params / 1e6:.3f} M ({n_params:,})")

    # Test forward
    device = next(model.parameters()).device
    x = torch.randn(*sample_input_shape).to(device)
    with torch.no_grad():
        logits, s_logit = model(x)
    print(
        f"✅ Shapes → logits: {tuple(logits.shape)} | s_logit: {tuple(s_logit.shape)}\n"
    )
