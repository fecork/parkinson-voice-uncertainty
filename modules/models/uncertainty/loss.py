"""
Pérdida heteroscedástica para clasificación con estimación de incertidumbre aleatoria.
"""

import torch
import torch.nn.functional as F


def heteroscedastic_classification_loss(logits, s_logit, targets, n_noise_samples=5):
    """
    Pérdida heteroscedástica para clasificación.

    Maximiza el log-likelihood esperado bajo ruido gaussiano en logits.

    Args:
        logits: Predicciones [B, C]
        s_logit: Log-varianza [B, C]
        targets: Labels [B]
        n_noise_samples: Número de muestras de ruido T_noise

    Returns:
        loss: Pérdida escalar
    """
    batch_size, n_classes = logits.shape

    # Calcular σ = exp(0.5 * s_logit)
    sigma = torch.exp(0.5 * s_logit)  # [B, C]

    # Almacenar log-probabilidades de cada muestra
    log_probs_samples = []

    for _ in range(n_noise_samples):
        # Muestrear ruido ε ~ N(0, 1)
        epsilon = torch.randn_like(logits)  # [B, C]

        # Logits con ruido: x̂ = logits + σ ⊙ ε
        noisy_logits = logits + sigma * epsilon  # [B, C]

        # Log-probabilidades
        log_probs = F.log_softmax(noisy_logits, dim=1)  # [B, C]

        # Extraer log_prob de la clase correcta
        log_prob_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        log_probs_samples.append(log_prob_target)

    # Stack: [n_noise_samples, B]
    log_probs_samples = torch.stack(log_probs_samples)

    # Log-mean-exp estable
    # loss = - mean_batch( log( mean_t exp(logp_y_t) ) )
    # Para estabilidad: m = max_t logp_y_t
    # log(mean_t exp(logp_y_t)) = m + log(mean_t exp(logp_y_t - m))

    m = log_probs_samples.max(dim=0)[0]  # [B]

    log_mean_exp = m + torch.log(
        torch.mean(torch.exp(log_probs_samples - m.unsqueeze(0)), dim=0) + 1e-12
    )  # [B]

    # Pérdida negativa
    loss = -log_mean_exp.mean()

    return loss


def compute_nll(probs_mean, targets):
    """
    Calcula el Negative Log-Likelihood.

    Args:
        probs_mean: Probabilidades promedio [B, C]
        targets: Labels [B]

    Returns:
        nll: NLL promedio
    """
    eps = 1e-12
    log_probs = torch.log(probs_mean + eps)
    nll = F.nll_loss(log_probs, targets)
    return nll.item()


def compute_brier_score(probs_mean, targets, n_classes):
    """
    Calcula el Brier Score.

    Args:
        probs_mean: Probabilidades promedio [B, C]
        targets: Labels [B]
        n_classes: Número de clases

    Returns:
        brier: Brier score promedio
    """
    # One-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=n_classes).float()

    # Brier score = mean((p - y)^2)
    brier = ((probs_mean - targets_one_hot) ** 2).sum(dim=1).mean()

    return brier.item()


def compute_ece(probs_mean, targets, n_bins=15):
    """
    Calcula el Expected Calibration Error (ECE).

    Args:
        probs_mean: Probabilidades promedio [B, C]
        targets: Labels [B]
        n_bins: Número de bins

    Returns:
        ece: ECE
    """
    confidences, predictions = probs_mean.max(dim=1)
    accuracies = predictions.eq(targets)

    # Crear bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs_mean.device)

    ece = 0.0
    for i in range(n_bins):
        # Samples en este bin
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )

        if in_bin.sum() > 0:
            bin_accuracy = accuracies[in_bin].float().mean()
            bin_confidence = confidences[in_bin].mean()

            ece += (in_bin.sum().float() / len(probs_mean)) * torch.abs(
                bin_accuracy - bin_confidence
            )

    return ece.item()
