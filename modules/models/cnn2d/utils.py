"""
CNN Utilities Module
====================
Funciones auxiliares para Domain Adaptation, visualización y utilidades generales.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Import from core module to avoid code duplication
from modules.core.utils import create_10fold_splits_by_speaker

__all__ = [
    "create_domain_mapping",
    "assign_domain_labels",
    "calculate_class_weights",
    "split_dataset_by_speaker",
    "create_10fold_splits_by_speaker",
    "create_dataloaders_from_existing",
    "compute_class_weights_from_dataset",
    "compute_class_weights_auto",
    "print_model_architecture",
    "visualize_model_graph",
    "plot_training_history_da",
    "plot_confusion_matrix",
]


# ============================================================
# DOMAIN LABEL MANAGEMENT
# ============================================================


def create_domain_mapping(metadata_list: List[dict]) -> Dict[str, int]:
    """
    Crea mapeo consistente de identificadores únicos a domain_id.

    Args:
        metadata_list: Lista de metadatos con 'subject_id' y 'filename'

    Returns:
        Dict mapeando identificador único → domain_id [0, n_domains-1]
    """
    unique_identifiers = set()

    for meta in metadata_list:
        # Usar subject_id o filename como identificador único
        identifier = meta.get("subject_id", meta.get("filename", "unknown"))
        unique_identifiers.add(identifier)

    # Ordenar para consistencia
    sorted_identifiers = sorted(unique_identifiers)

    # Crear mapeo
    domain_mapping = {ident: idx for idx, ident in enumerate(sorted_identifiers)}

    return domain_mapping


def create_domain_labels_from_metadata(
    metadata_list: List[dict], domain_mapping: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Convierte metadata a labels de dominio usando mapeo.

        Args:
        metadata_list: Lista de metadatos
        domain_mapping: Mapeo opcional (si None, se crea automáticamente)

        Returns:
        Tensor de domain labels (N,)
    """
    if domain_mapping is None:
        domain_mapping = create_domain_mapping(metadata_list)

    domain_labels = []
    for meta in metadata_list:
        identifier = meta.get("subject_id", meta.get("filename", "unknown"))
        domain_id = domain_mapping.get(identifier, 0)
        domain_labels.append(domain_id)

    return torch.tensor(domain_labels, dtype=torch.long)


def print_domain_statistics(
    domain_labels: torch.Tensor, task_labels: torch.Tensor
) -> None:
    """
    Imprime estadísticas de distribución de dominios.

    Args:
        domain_labels: Labels de dominio (N,)
        task_labels: Labels de tarea PD (N,)
    """
    n_domains = len(torch.unique(domain_labels))
    # domain_counts = Counter(domain_labels.numpy())  # Para análisis futuro

    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DE DOMINIOS")
    print("=" * 60)
    print(f"Total dominios únicos: {n_domains}")
    print(f"Total muestras: {len(domain_labels)}")
    avg_samples = len(domain_labels) / n_domains
    print(f"Promedio muestras/dominio: {avg_samples:.1f}")

    # Distribución por clase
    n_pd = (task_labels == 1).sum().item()
    n_hc = (task_labels == 0).sum().item()
    print("\nDistribución por clase:")
    print(f"  Parkinson (1): {n_pd} muestras")
    print(f"  Healthy (0):   {n_hc} muestras")
    balance_pd = n_pd / (n_pd + n_hc) * 100
    balance_hc = n_hc / (n_pd + n_hc) * 100
    print(f"  Balance: {balance_pd:.1f}% PD / {balance_hc:.1f}% HC")
    print("=" * 60 + "\n")


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calcula pesos de clase para balanceo en loss.

    Args:
        labels: Labels de clase (N,)

    Returns:
        Tensor de pesos (n_classes,)
    """
    class_counts = torch.bincount(labels)
    total_samples = len(labels)

    # Peso inversamente proporcional a frecuencia
    weights = total_samples / (len(class_counts) * class_counts.float())

    return weights


def compute_class_weights_auto(
    labels: torch.Tensor, threshold: float = 0.4
) -> Optional[torch.Tensor]:
    """
    Detecta desbalance automáticamente y calcula pesos si es necesario.

    Args:
        labels: Labels de clase (N,)
        threshold: Umbral para detectar desbalance (default: 0.4)
                   Si clase minoritaria < threshold * total, aplicar pesos

    Returns:
        Tensor de pesos si hay desbalance, None si está balanceado
    """
    class_counts = torch.bincount(labels)
    total_samples = len(labels)

    # Calcular proporción de clase minoritaria
    min_proportion = class_counts.min().item() / total_samples

    if min_proportion < threshold:
        # Hay desbalance, calcular pesos
        weights = calculate_class_weights(labels)
        print(
            f"   ⚠️  Desbalance detectado (min class: {min_proportion:.1%}). Aplicando pesos."
        )
        return weights
    else:
        print(f"   ✓ Dataset balanceado (min class: {min_proportion:.1%}). Sin pesos.")
        return None


# ============================================================
# SPEAKER-INDEPENDENT SPLITS
# ============================================================


def split_by_speaker(
    metadata_list: List[dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split speaker-independent: garantiza que un hablante está solo en un conjunto.

    Args:
        metadata_list: Lista de metadatos con 'subject_id'
        train_ratio: Proporción para training
        val_ratio: Proporción para validation
        test_ratio: Proporción para test
        seed: Semilla para reproducibilidad

    Returns:
        Dict con índices: {'train': [...], 'val': [...], 'test': [...]}
    """
    # Agrupar índices por subject_id
    subject_to_indices = {}
    for idx, meta in enumerate(metadata_list):
        subject_id = meta.get("subject_id", meta.get("filename", f"unknown_{idx}"))
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)

    subjects = list(subject_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(subjects)

    # Split de hablantes
    n_train = int(len(subjects) * train_ratio)
    n_val = int(len(subjects) * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    # Obtener índices de muestras
    train_indices = [idx for subj in train_subjects for idx in subject_to_indices[subj]]
    val_indices = [idx for subj in val_subjects for idx in subject_to_indices[subj]]
    test_indices = [idx for subj in test_subjects for idx in subject_to_indices[subj]]

    print(f"\n📊 Split speaker-independent:")
    print(f"   Train: {len(train_subjects)} hablantes, {len(train_indices)} muestras")
    print(f"   Val:   {len(val_subjects)} hablantes, {len(val_indices)} muestras")
    print(f"   Test:  {len(test_subjects)} hablantes, {len(test_indices)} muestras")

    return {"train": train_indices, "val": val_indices, "test": test_indices}


# ============================================================
# DATALOADER CREATION
# ============================================================


def create_dataloaders_from_existing(
    base_dataset,
    split_indices: Dict[str, List[int]],
    batch_size: int = 32,
    spec_augment_params: Optional[Dict] = None,
    num_workers: int = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Crea DataLoaders desde dataset existente con splits.

    Args:
        base_dataset: Dataset base (ConcatDataset o similar)
        split_indices: Dict con 'train', 'val', 'test' indices
        batch_size: Tamaño de batch
        spec_augment_params: Parámetros para SpecAugment (opcional)
        num_workers: Workers para DataLoader

    Returns:
        Dict con DataLoaders: {'train': ..., 'val': ..., 'test': ...}
    """
    from torch.utils.data import DataLoader, Subset

    # Crear subsets
    train_subset = Subset(base_dataset, split_indices["train"])
    val_subset = Subset(base_dataset, split_indices["val"])

    # Crear loaders
    loaders = {
        "train": DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    # Test loader si existe
    if "test" in split_indices:
        test_subset = Subset(base_dataset, split_indices["test"])
        loaders["test"] = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    print(f"\n✅ DataLoaders creados:")
    print(f"   Train batches: {len(loaders['train'])}")
    print(f"   Val batches:   {len(loaders['val'])}")
    if "test" in loaders:
        print(f"   Test batches:  {len(loaders['test'])}")

    return loaders


def compute_class_weights_from_dataset(dataset, indices: List[int]) -> torch.Tensor:
    """
    Calcula pesos de clase desde dataset y índices.

    Args:
        dataset: Dataset PyTorch
        indices: Lista de índices a considerar

    Returns:
        Tensor de pesos de clase
    """
    labels = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, dict):
            label = sample["label"]
        else:
            label = sample[1]  # Asumiendo (X, y) o (X, y, domain)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return calculate_class_weights(labels_tensor)


# ============================================================
# MODEL ARCHITECTURE VISUALIZATION
# ============================================================


def print_model_architecture(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 65, 41)
) -> None:
    """
    Imprime arquitectura del modelo con detalles por capa.

        Args:
        model: Modelo PyTorch
        input_shape: Shape de entrada (default: (B, 1, 65, 41))
    """
    try:
        from torchinfo import summary

        print("\n" + "=" * 70)
        print("🏗️  ARQUITECTURA DEL MODELO")
        print("=" * 70)

        summary(
            model,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"],
            verbose=1,
        )
    except ImportError:
        # Fallback sin torchinfo
        print("\n" + "=" * 70)
        print("🏗️  ARQUITECTURA DEL MODELO")
        print("=" * 70)
        print(model)
        print("\n" + "-" * 70)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parámetros: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("-" * 70 + "\n")


def visualize_model_graph(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 65, 41),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea diagrama visual de la arquitectura del modelo.

    Args:
        model: Modelo PyTorch
        input_shape: Shape de entrada
        save_path: Ruta para guardar imagen (opcional)
    """
    try:
        from torchviz import make_dot

        # Crear input dummy
        x = torch.randn(input_shape)

        # Forward pass
        if hasattr(model, "forward") and "logits_pd" in str(model.forward.__code__):
            # Modelo DA con dos salidas
            logits_pd, logits_domain = model(x)
            output = {"pd": logits_pd, "domain": logits_domain}
        else:
            output = model(x)

        # Crear grafo
        dot = make_dot(output, params=dict(model.named_parameters()))

        if save_path:
            dot.render(save_path, format="png")
            print(f"📊 Grafo guardado en: {save_path}.png")
        else:
            print("📊 Grafo creado (usa save_path para guardarlo)")

        return dot
    except ImportError:
        print("⚠️  torchviz no disponible. Instala con: pip install torchviz graphviz")


# ============================================================
# TRAINING VISUALIZATION
# ============================================================


def plot_training_history_da(
    history: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grafica historial de entrenamiento con Domain Adaptation.

    Args:
        history: Dict con métricas de entrenamiento
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Historial de Entrenamiento - Domain Adaptation", fontsize=16)

    epochs = range(1, len(history["train_loss_pd"]) + 1)

    # Plot 1: Loss PD
    axes[0, 0].plot(epochs, history["train_loss_pd"], label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss_pd"], label="Val", linewidth=2)
    axes[0, 0].set_title("Loss PD (Tarea Principal)")
    axes[0, 0].set_xlabel("Época")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Loss Domain
    axes[0, 1].plot(
        epochs, history["train_loss_domain"], label="Train", linewidth=2, alpha=0.7
    )
    axes[0, 1].plot(
        epochs, history["val_loss_domain"], label="Val", linewidth=2, alpha=0.7
    )
    axes[0, 1].set_title("Loss Domain (Tarea Auxiliar)")
    axes[0, 1].set_xlabel("Época")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Loss Total
    axes[0, 2].plot(epochs, history["train_loss_total"], label="Train", linewidth=2)
    axes[0, 2].plot(epochs, history["val_loss_total"], label="Val", linewidth=2)
    axes[0, 2].set_title("Loss Total (PD + α·Domain)")
    axes[0, 2].set_xlabel("Época")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Plot 4: Accuracy PD
    axes[1, 0].plot(epochs, history["train_acc_pd"], label="Train", linewidth=2)
    axes[1, 0].plot(epochs, history["val_acc_pd"], label="Val", linewidth=2)
    axes[1, 0].set_title("Accuracy PD")
    axes[1, 0].set_xlabel("Época")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 5: F1 Score PD
    axes[1, 1].plot(epochs, history["train_f1_pd"], label="Train", linewidth=2)
    axes[1, 1].plot(epochs, history["val_f1_pd"], label="Val", linewidth=2)
    axes[1, 1].set_title("F1 Score PD")
    axes[1, 1].set_xlabel("Época")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Plot 6: Lambda Schedule
    axes[1, 2].plot(epochs, history["lambda_values"], linewidth=2, color="purple")
    axes[1, 2].set_title("Lambda Schedule (GRL)")
    axes[1, 2].set_xlabel("Época")
    axes[1, 2].set_ylabel("λ")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Gráfica guardada en: {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Matriz de Confusión",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza matriz de confusión.

    Args:
        cm: Matriz de confusión (n_classes, n_classes)
        class_names: Nombres de clases
        title: Título de la gráfica
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """
    if class_names is None:
        class_names = ["HC", "PD"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalizar por filas (recall por clase)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proporción"},
        ax=ax,
    )

    # Anotar conteos absolutos
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"n={cm[i, j]}",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Clase Real", fontsize=12)
    ax.set_xlabel("Clase Predicha", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Matriz guardada en: {save_path}")

    if show:
        plt.show()

    return fig


def plot_lambda_schedule(
    n_epochs: int = 100,
    gamma: float = 10.0,
    power: float = 0.75,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza el scheduler de lambda para GRL.

    Args:
        n_epochs: Número de épocas
        gamma: Parámetro gamma
        power: Exponente
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """

    # Función local para calcular lambda schedule
    def _compute_lambda(epoch, max_epoch, gamma_val, power_val):
        p = epoch / max(max_epoch, 1)
        lambda_p = 2.0 / (1.0 + np.exp(-gamma_val * p)) ** power_val - 1.0
        return max(0.0, min(1.0, lambda_p))

    epochs = np.arange(n_epochs)
    lambdas = [_compute_lambda(e, n_epochs, gamma, power) for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, lambdas, linewidth=2, color="purple")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="λ = 0.5")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="λ = 1.0")

    ax.set_title(
        f"Lambda Schedule (γ={gamma}, power={power})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("λ (Factor GRL)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()

    # Anotar puntos clave
    lambda_25 = lambdas[n_epochs // 4]
    lambda_50 = lambdas[n_epochs // 2]
    lambda_75 = lambdas[3 * n_epochs // 4]

    ax.scatter([n_epochs // 4], [lambda_25], color="red", s=100, zorder=5)
    ax.scatter([n_epochs // 2], [lambda_50], color="red", s=100, zorder=5)
    ax.scatter([3 * n_epochs // 4], [lambda_75], color="red", s=100, zorder=5)

    ax.text(
        n_epochs // 4, lambda_25 + 0.05, f"{lambda_25:.3f}", ha="center", fontsize=10
    )
    ax.text(
        n_epochs // 2, lambda_50 + 0.05, f"{lambda_50:.3f}", ha="center", fontsize=10
    )
    ax.text(
        3 * n_epochs // 4,
        lambda_75 + 0.05,
        f"{lambda_75:.3f}",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Schedule guardado en: {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================
# COMPARATIVE VISUALIZATION
# ============================================================


def plot_model_comparison(
    results_baseline: Dict,
    results_da: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compara resultados entre modelo baseline y DA.

    Args:
        results_baseline: Resultados del modelo sin DA
        results_da: Resultados del modelo con DA
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparación: Baseline vs Domain Adaptation", fontsize=16)

    # Extraer métricas finales
    baseline_acc = results_baseline["history"]["val_acc"][-1]
    baseline_f1 = results_baseline["history"]["val_f1"][-1]

    da_acc = results_da["history"]["val_acc_pd"][-1]
    da_f1 = results_da["history"]["val_f1_pd"][-1]

    # Plot 1: Barras de comparación
    metrics = ["Accuracy", "F1 Score"]
    baseline_vals = [baseline_acc, baseline_f1]
    da_vals = [da_acc, da_f1]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width / 2, baseline_vals, width, label="Baseline", alpha=0.8)
    axes[0].bar(x + width / 2, da_vals, width, label="Domain Adaptation", alpha=0.8)

    axes[0].set_ylabel("Score")
    axes[0].set_title("Métricas de Validación Final")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([0, 1.0])

    # Anotar valores
    for i, (b, d) in enumerate(zip(baseline_vals, da_vals)):
        axes[0].text(i - width / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=10)
        axes[0].text(i + width / 2, d + 0.02, f"{d:.3f}", ha="center", fontsize=10)

    # Plot 2: Curvas de aprendizaje
    epochs_baseline = range(1, len(results_baseline["history"]["val_loss"]) + 1)
    epochs_da = range(1, len(results_da["history"]["val_loss_pd"]) + 1)

    axes[1].plot(
        epochs_baseline,
        results_baseline["history"]["val_loss"],
        label="Baseline",
        linewidth=2,
    )
    axes[1].plot(
        epochs_da, results_da["history"]["val_loss_pd"], label="DA", linewidth=2
    )

    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Curvas de Aprendizaje")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Comparación guardada en: {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================
# QUICK TEST
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: CNN UTILS MODULE")
    print("=" * 70)

    # Test 1: Domain mapping
    print("\n1. Test domain mapping:")
    metadata = [
        {"subject_id": "1580", "filename": "1580-a_h.egg"},
        {"subject_id": "1580", "filename": "1580-a_l.egg"},
        {"subject_id": "1121", "filename": "1121-u_h.egg"},
    ]

    domain_map = create_domain_mapping(metadata)
    print(f"   Domain mapping: {domain_map}")

    domain_labels = create_domain_labels_from_metadata(metadata, domain_map)
    print(f"   Domain labels: {domain_labels}")

    # Test 2: Class weights
    print("\n2. Test class weights:")
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])  # Desbalanceado
    weights = calculate_class_weights(labels)
    print(f"   Labels: {labels}")
    print(f"   Weights: {weights}")

    # Test 3: Lambda schedule visualization
    print("\n3. Test lambda schedule:")
    fig = plot_lambda_schedule(n_epochs=100, show=False)
    print("   ✓ Lambda schedule plot creado")
    plt.close(fig)

    print("\n✅ Todos los tests pasaron correctamente")
