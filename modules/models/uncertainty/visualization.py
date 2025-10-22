"""
Funciones de visualización para modelos con incertidumbre.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_uncertainty_histograms(metrics, save_path=None, show=True):
    """
    Histogramas de incertidumbres separando aciertos vs errores.

    Args:
        metrics: Dict con resultados de evaluate_with_uncertainty
        save_path: Path para guardar la figura
        show: Si mostrar la figura
    """
    preds = metrics["predictions"]
    targets = metrics["targets"]
    entropy = metrics["entropy"]
    epistemic = metrics["epistemic"]
    aleatoric = metrics["aleatoric"]

    correct_mask = preds == targets

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Entropía
    axes[0].hist(
        entropy[correct_mask],
        bins=30,
        alpha=0.6,
        label="Correcto",
        color="green",
        density=True,
    )
    axes[0].hist(
        entropy[~correct_mask],
        bins=30,
        alpha=0.6,
        label="Incorrecto",
        color="red",
        density=True,
    )
    axes[0].set_xlabel("Entropía Predictiva")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Entropía Total (H(p̄))")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Epistémica
    axes[1].hist(
        epistemic[correct_mask],
        bins=30,
        alpha=0.6,
        label="Correcto",
        color="green",
        density=True,
    )
    axes[1].hist(
        epistemic[~correct_mask],
        bins=30,
        alpha=0.6,
        label="Incorrecto",
        color="red",
        density=True,
    )
    axes[1].set_xlabel("Incertidumbre Epistémica (BALD)")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Epistémica (Modelo)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Aleatoria
    axes[2].hist(
        aleatoric[correct_mask],
        bins=30,
        alpha=0.6,
        label="Correcto",
        color="green",
        density=True,
    )
    axes[2].hist(
        aleatoric[~correct_mask],
        bins=30,
        alpha=0.6,
        label="Incorrecto",
        color="red",
        density=True,
    )
    axes[2].set_xlabel("Incertidumbre Aleatoria (σ²)")
    axes[2].set_ylabel("Densidad")
    axes[2].set_title("Aleatoria (Datos)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Histogramas guardados en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_reliability_diagram(metrics, n_bins=10, save_path=None, show=True):
    """
    Diagrama de calibración (reliability plot).

    Args:
        metrics: Dict con resultados de evaluate_with_uncertainty
        n_bins: Número de bins para el diagrama
        save_path: Path para guardar la figura
        show: Si mostrar la figura
    """
    preds = metrics["predictions"]
    targets = metrics["targets"]
    confidence = metrics["confidence"]

    # Crear bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = []
    confidences_bin = []
    counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.sum() / len(confidence)

        if prop_in_bin > 0:
            accuracy_in_bin = (preds[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            accuracies.append(accuracy_in_bin)
            confidences_bin.append(avg_confidence_in_bin)
            counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences_bin.append(0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Diagonal perfecta
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")

    # Barras
    width = 1.0 / n_bins
    ax.bar(
        confidences_bin,
        accuracies,
        width=width,
        alpha=0.6,
        edgecolor="black",
        label="Calibración real",
    )

    ax.set_xlabel("Confianza", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram (ECE: {metrics['ece']:.4f})", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Diagrama de calibración guardado en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_uncertainty_scatter(metrics, save_path=None, show=True):
    """
    Scatter plot de incertidumbre epistémica vs aleatoria.

    Args:
        metrics: Dict con resultados de evaluate_with_uncertainty
        save_path: Path para guardar la figura
        show: Si mostrar la figura
    """
    preds = metrics["predictions"]
    targets = metrics["targets"]
    epistemic = metrics["epistemic"]
    aleatoric = metrics["aleatoric"]

    correct_mask = preds == targets

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        epistemic[correct_mask],
        aleatoric[correct_mask],
        alpha=0.5,
        c="green",
        label="Correcto",
        s=20,
    )
    ax.scatter(
        epistemic[~correct_mask],
        aleatoric[~correct_mask],
        alpha=0.7,
        c="red",
        label="Incorrecto",
        s=30,
        marker="x",
    )

    ax.set_xlabel("Incertidumbre Epistémica (BALD)", fontsize=12)
    ax.set_ylabel("Incertidumbre Aleatoria (σ²)", fontsize=12)
    ax.set_title("Incertidumbre Epistémica vs Aleatoria", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Scatter plot guardado en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_history_uncertainty(history, save_path=None, show=True):
    """
    Plot del historial de entrenamiento.

    Args:
        history: Dict con listas de train_loss, val_loss, train_acc, val_acc
        save_path: Path para guardar
        show: Si mostrar
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Pérdida Heteroscedástica")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Historial guardado en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig
