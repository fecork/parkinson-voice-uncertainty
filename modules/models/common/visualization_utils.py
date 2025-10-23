"""
Utilidades de Visualización Comunes
===================================
Herramientas compartidas para visualización de todas las arquitecturas:
- Gráficas de entrenamiento
- Matrices de confusión
- t-SNE embeddings
- Métricas de rendimiento
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================


def setup_plot_style():
    """Configura estilo común para todas las gráficas."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# ============================================================
# GRÁFICAS DE ENTRENAMIENTO
# ============================================================


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Progreso de Entrenamiento",
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualiza curvas de entrenamiento estándar.
    
    Args:
        history: Dict con historiales de train/val
        title: Título de la gráfica
        save_path: Ruta para guardar
        show: Si mostrar la figura
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    setup_plot_style()
    
    # Determinar número de subplots
    n_metrics = len([k for k in history.keys() if not k.startswith('val_')])
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16)
    
    epochs = range(1, len(list(history.values())[0]) + 1)
    
    # Plot cada métrica
    metric_idx = 0
    for metric_name, values in history.items():
        if metric_name.startswith('val_'):
            continue
            
        ax = axes[metric_idx]
        
        # Plot train
        ax.plot(epochs, values, label=f'Train {metric_name}', linewidth=2)
        
        # Plot val si existe
        val_metric = f'val_{metric_name}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], label=f'Val {metric_name}', linewidth=2)
        
        ax.set_xlabel('Época')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        metric_idx += 1
    
    # Ocultar subplots vacíos
    for i in range(metric_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Gráfica guardada en: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# ============================================================
# MATRICES DE CONFUSIÓN
# ============================================================


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusión",
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Visualiza matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        class_names: Nombres de las clases
        title: Título de la gráfica
        save_path: Ruta para guardar
        show: Si mostrar la figura
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
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
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
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
    
    return fig


# ============================================================
# t-SNE EMBEDDINGS
# ============================================================


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domains: Optional[np.ndarray] = None,
    title: str = "t-SNE de Embeddings",
    save_path: Optional[Path] = None,
    show: bool = True,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 9)
) -> plt.Figure:
    """
    Visualiza embeddings con t-SNE.
    
    Args:
        embeddings: [N, emb_dim] embeddings del modelo
        labels: [N] clases (0=HC, 1=PD)
        domains: [N] IDs de dominio/corpus (opcional)
        title: Título de la gráfica
        save_path: Ruta para guardar
        show: Si mostrar la figura
        perplexity: Perplexity para t-SNE
        random_state: Seed para reproducibilidad
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    setup_plot_style()
    
    print(f"\n🔄 Computando t-SNE (perplexity={perplexity})...")
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state
    )
    emb_2d = tsne.fit_transform(embeddings)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores y markers
    colors = {0: "blue", 1: "red"}  # HC=blue, PD=red
    class_names = {0: "Healthy", 1: "Parkinson"}
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "<", ">"]
    
    # Plot por clase y dominio
    if domains is not None:
        unique_domains = np.unique(domains)
        for cls in [0, 1]:
            for dom in unique_domains:
                mask = (labels == cls) & (domains == dom)
                if mask.sum() > 0:
                    ax.scatter(
                        emb_2d[mask, 0], emb_2d[mask, 1],
                        c=colors[cls], marker=markers[int(dom) % len(markers)],
                        alpha=0.6, s=50,
                        label=f"{class_names[cls]} - Dom{int(dom)}",
                        edgecolors="k", linewidths=0.5,
                    )
    else:
        # Solo por clase
        for cls in [0, 1]:
            mask = (labels == cls)
            if mask.sum() > 0:
                ax.scatter(
                    emb_2d[mask, 0], emb_2d[mask, 1],
                    c=colors[cls], alpha=0.6, s=50,
                    label=class_names[cls],
                    edgecolors="k", linewidths=0.5,
                )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14)
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
    
    return fig


# ============================================================
# MÉTRICAS DE RENDIMIENTO
# ============================================================


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Comparación de Métricas",
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualiza comparación de métricas entre modelos.
    
    Args:
        metrics_dict: Dict con métricas por modelo
        title: Título de la gráfica
        save_path: Ruta para guardar
        show: Si mostrar la figura
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Preparar datos
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valores')
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Comparación guardada en: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
