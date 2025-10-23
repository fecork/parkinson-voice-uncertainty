"""
CNN 1D Visualization Module
=============================
Funciones de visualización específicas para CNN1D con Domain Adaptation.
Incluye t-SNE para embeddings y curvas de entrenamiento.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# TRAINING PROGRESS
# ============================================================


def plot_1d_training_progress(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Visualiza progreso de entrenamiento multi-task para CNN1D_DA.

    Args:
        history: Dict con historiales de train/val
        save_path: Ruta para guardar figura
        show: Si mostrar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Progreso de Entrenamiento - CNN1D con Domain Adaptation",
        fontsize=16
    )

    epochs = range(1, len(history["train_loss_pd"]) + 1)

    # 1. Loss PD (Train vs Val)
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss_pd"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss_pd"], label="Val", linewidth=2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss PD (Tarea Principal)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Loss Domain (Train vs Val)
    ax = axes[0, 1]
    ax.plot(epochs, history["train_loss_domain"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss_domain"], label="Val", linewidth=2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Domain (Tarea Adversarial)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. F1-Score PD en Validation
    ax = axes[1, 0]
    ax.plot(
        epochs, history["val_f1_pd"], label="Val F1",
        linewidth=2, color="green"
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score PD (Validación)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Lambda GRL
    ax = axes[1, 1]
    ax.plot(
        epochs, history["lambda"], label="Lambda",
        linewidth=2, color="purple"
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("λ")
    ax.set_title("Factor Lambda (GRL)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Gráfica guardada en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# t-SNE EMBEDDINGS
# ============================================================


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Visualiza embeddings con t-SNE.

    Verifica que Domain Adaptation mezcla dominios:
    - Clusters deben formarse por clase (HC/PD), no por corpus
    - Colores por clase, markers por dominio

    Args:
        embeddings: [N, emb_dim] embeddings del modelo
        labels: [N] clases (0=HC, 1=PD)
        domains: [N] IDs de dominio/corpus
        save_path: Ruta para guardar figura
        show: Si mostrar la figura
        perplexity: Perplexity para t-SNE
        random_state: Seed para reproducibilidad
    """
    print(f"\n🔄 Computando t-SNE (perplexity={perplexity})...")

    # Compute t-SNE
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state
    )
    emb_2d = tsne.fit_transform(embeddings)

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # Colores y markers
    colors = {0: "blue", 1: "red"}  # HC=blue, PD=red
    class_names = {0: "Healthy", 1: "Parkinson"}
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">"]

    # Plot por clase y dominio
    unique_domains = np.unique(domains)
    for cls in [0, 1]:
        for dom in unique_domains:
            mask = (labels == cls) & (domains == dom)
            if mask.sum() > 0:
                ax.scatter(
                    emb_2d[mask, 0],
                    emb_2d[mask, 1],
                    c=colors[cls],
                    marker=markers[int(dom) % len(markers)],
                    alpha=0.6,
                    s=50,
                    label=f"{class_names[cls]} - Dom{int(dom)}",
                    edgecolors="k",
                    linewidths=0.5,
                )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(
        "t-SNE de Embeddings - Verificación Domain Adaptation\n"
        "(Clusters por clase = DA funciona; clusters por dominio = DA falla)",
        fontsize=14,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 t-SNE guardado en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# CONFUSION MATRIX (reutiliza de cnn_utils o define simple)
# ============================================================


def plot_simple_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusión",
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot simple de confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: Nombres de clases
        title: Título
        save_path: Ruta para guardar
        show: Si mostrar
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Valores en celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Matriz guardada en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
