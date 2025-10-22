"""
Time-CNN-BiLSTM Visualization Module
=====================================
Funciones de visualización para Time-CNN-BiLSTM con Domain Adaptation.
Reutiliza código existente de CNN2D y CNN1D.
"""

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

# Reutilizar funciones de CNN2D
from ..cnn2d.visualization import (
    plot_training_history,
)

# Reutilizar t-SNE de CNN1D (si existe)
try:
    from ..cnn1d.visualization import plot_tsne_embeddings
except ImportError:
    plot_tsne_embeddings = None


# ============================================================
# TRAINING CURVES PARA DA
# ============================================================


def plot_training_curves_da(
    history: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
    title: str = "Time-CNN-BiLSTM-DA Training",
):
    """
    Grafica curvas de entrenamiento para modelo DA (dual-task).

    Args:
        history: Dict con 'train' y 'val', cada uno una lista de dicts con métricas
        save_path: Path para guardar figura
        title: Título de la figura
    """
    train_history = history["train"]
    val_history = history["val"]

    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, y=0.995)

    # Loss PD
    ax = axes[0, 0]
    ax.plot(epochs, [m["loss_pd"] for m in train_history], label="Train", marker="o", markersize=3)
    ax.plot(epochs, [m["loss_pd"] for m in val_history], label="Val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PD Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss Domain
    ax = axes[0, 1]
    ax.plot(epochs, [m["loss_domain"] for m in train_history], label="Train", marker="o", markersize=3)
    ax.plot(epochs, [m["loss_domain"] for m in val_history], label="Val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Domain Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss Total
    ax = axes[0, 2]
    ax.plot(epochs, [m["loss_total"] for m in train_history], label="Train", marker="o", markersize=3)
    ax.plot(epochs, [m["loss_total"] for m in val_history], label="Val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy PD
    ax = axes[1, 0]
    ax.plot(epochs, [m["accuracy_pd"] for m in train_history], label="Train", marker="o", markersize=3)
    ax.plot(epochs, [m["accuracy_pd"] for m in val_history], label="Val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("PD Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # F1 PD (solo val)
    ax = axes[1, 1]
    ax.plot(epochs, [m.get("f1_pd", 0) for m in val_history], label="Val F1", marker="s", markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("PD F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Lambda GRL
    ax = axes[1, 2]
    ax.plot(epochs, [m.get("lambda_grl", 0) for m in train_history], label="λ GRL", marker="o", markersize=3, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lambda")
    ax.set_title("GRL Lambda Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura guardada: {save_path}")

    plt.show()


# ============================================================
# K-FOLD RESULTS
# ============================================================


def plot_kfold_results(
    fold_results: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Visualiza resultados de K-fold CV.

    Args:
        fold_results: Lista de resultados por fold
        save_path: Path para guardar figura
    """
    n_folds = len(fold_results)

    # Extraer métricas
    folds = [r["fold"] for r in fold_results]
    val_losses = [r["best_val_loss_pd"] for r in fold_results]
    val_accs = [r["best_metrics"]["accuracy_pd"] for r in fold_results]
    val_f1s = [r["best_metrics"]["f1_pd"] for r in fold_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"K-Fold CV Results (n={n_folds})", fontsize=14)

    # Loss
    ax = axes[0]
    ax.bar(folds, val_losses, color="steelblue", alpha=0.7)
    ax.axhline(np.mean(val_losses), color="red", linestyle="--", label=f"Mean: {np.mean(val_losses):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Val Loss PD")
    ax.set_title("Validation Loss per Fold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Accuracy
    ax = axes[1]
    ax.bar(folds, val_accs, color="green", alpha=0.7)
    ax.axhline(np.mean(val_accs), color="red", linestyle="--", label=f"Mean: {np.mean(val_accs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Validation Accuracy per Fold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # F1
    ax = axes[2]
    ax.bar(folds, val_f1s, color="orange", alpha=0.7)
    ax.axhline(np.mean(val_f1s), color="red", linestyle="--", label=f"Mean: {np.mean(val_f1s):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Val F1 Score")
    ax.set_title("Validation F1 Score per Fold")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura guardada: {save_path}")

    plt.show()


# ============================================================
# LSTM ATTENTION WEIGHTS (OPCIONAL)
# ============================================================


def plot_lstm_attention_weights(
    model,
    dataloader,
    device,
    n_samples: int = 5,
    save_path: Optional[str] = None,
):
    """
    Visualiza la atención temporal aprendida por el modelo.

    Nota: Esta es una aproximación visual. El modelo BiLSTM no tiene
    mecanismo de atención explícito, pero podemos visualizar la importancia
    de cada frame observando las activaciones LSTM.

    Args:
        model: Modelo TimeCNNBiLSTM_DA
        dataloader: DataLoader con datos
        device: Device
        n_samples: Número de muestras a visualizar
        save_path: Path para guardar figura
    """
    model.eval()

    samples_collected = 0
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))

    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= n_samples:
                break

            X = batch["X"].to(device)
            lengths = batch["length"]
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths)
            lengths = lengths.to(device)

            y_true = batch["y_task"].cpu().numpy()

            # Forward para obtener activaciones
            # Nota: Necesitaríamos modificar el modelo para retornar activaciones LSTM
            # Por ahora, solo visualizamos la presencia de frames

            for i in range(min(X.size(0), n_samples - samples_collected)):
                ax = axes[samples_collected]
                length = lengths[i].item()
                n_frames = X.size(1)

                # Visualizar frames válidos vs padding
                frame_weights = np.zeros(n_frames)
                frame_weights[:length] = 1.0

                ax.bar(range(n_frames), frame_weights, color="steelblue", alpha=0.7)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Valid Frame")
                ax.set_title(
                    f"Sample {samples_collected + 1} | "
                    f"Label: {'PD' if y_true[i] == 1 else 'HC'} | "
                    f"Valid Frames: {length}/{n_frames}"
                )
                ax.set_ylim([0, 1.2])
                ax.grid(True, alpha=0.3, axis="y")

                samples_collected += 1

                if samples_collected >= n_samples:
                    break

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura guardada: {save_path}")

    plt.show()


# ============================================================
# EXPORT FUNCIONES REUTILIZADAS
# ============================================================

# Re-exportar funciones útiles de otros módulos
__all__ = [
    "plot_training_curves_da",
    "plot_kfold_results",
    "plot_lstm_attention_weights",
    "plot_training_history",  # de CNN2D
]

# Agregar plot_tsne_embeddings solo si está disponible
if plot_tsne_embeddings is not None:
    __all__.append("plot_tsne_embeddings")
