"""
CNN Visualization Module
=========================
Visualizaciones para análisis: Grad-CAM, incertidumbre, métricas.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from .cnn_model import GradCAM, get_last_conv_layer


# ============================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ============================================================
# VISUALIZACIÓN DE GRAD-CAM
# ============================================================


def visualize_gradcam(
    model: nn.Module,
    spec: torch.Tensor,
    label: int,
    prediction: int,
    cam: torch.Tensor,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Visualiza espectrograma con superposición de Grad-CAM.

    Args:
        model: Modelo PyTorch
        spec: Espectrograma (1, 65, 41)
        label: Etiqueta verdadera
        prediction: Predicción del modelo
        cam: Mapa CAM (65, 41)
        title: Título personalizado
        save_path: Ruta para guardar figura
    """
    class_names = ["HC", "PD"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Espectrograma original
    spec_np = spec.squeeze().cpu().numpy()

    axes[0].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Espectrograma Original")
    axes[0].set_xlabel("Frames temporales")
    axes[0].set_ylabel("Mel bins")

    # Grad-CAM
    cam_np = cam.cpu().numpy() if torch.is_tensor(cam) else cam

    axes[1].imshow(cam_np, aspect="auto", origin="lower", cmap="jet", alpha=0.8)
    axes[1].set_title("Grad-CAM")
    axes[1].set_xlabel("Frames temporales")
    axes[1].set_ylabel("Mel bins")

    # Superposición
    axes[2].imshow(spec_np, aspect="auto", origin="lower", cmap="gray")
    axes[2].imshow(cam_np, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title("Superposición")
    axes[2].set_xlabel("Frames temporales")
    axes[2].set_ylabel("Mel bins")

    # Título general
    if title is None:
        correct = "✓" if label == prediction else "✗"
        title = (
            f"{correct} Real: {class_names[label]} | Pred: {class_names[prediction]}"
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Guardado: {save_path}")

    plt.show()


def visualize_multiple_gradcam(
    model: nn.Module,
    specs: torch.Tensor,
    labels: List[int],
    predictions: List[int],
    indices: List[int],
    dataset,
    n_cols: int = 3,
    save_path: Optional[Path] = None,
):
    """
    Visualiza múltiples casos con Grad-CAM.

    Args:
        model: Modelo PyTorch
        specs: Batch de espectrogramas
        labels: Etiquetas verdaderas
        predictions: Predicciones
        indices: Índices de los casos
        dataset: Dataset para obtener información adicional
        n_cols: Número de columnas
        save_path: Ruta para guardar
    """
    n_cases = len(indices)
    n_rows = (n_cases + n_cols - 1) // n_cols

    class_names = ["HC", "PD"]

    # Crear Grad-CAM
    last_conv = get_last_conv_layer(model)
    gradcam = GradCAM(model, last_conv)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cases == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Obtener datos
        spec = specs[idx : idx + 1]  # (1, 1, 65, 41)
        label = labels[idx]
        pred = predictions[idx]

        # Generar CAM
        cam = gradcam.generate_cam(spec, target_class=pred)
        cam_np = cam.squeeze().cpu().numpy()

        # Espectrograma
        spec_np = spec.squeeze().cpu().numpy()

        # Visualizar superposición
        ax.imshow(spec_np, aspect="auto", origin="lower", cmap="gray", alpha=0.7)
        im = ax.imshow(cam_np, aspect="auto", origin="lower", cmap="jet", alpha=0.5)

        # Título
        correct = "✓" if label == pred else "✗"
        ax.set_title(
            f"{correct} Real: {class_names[label]} | Pred: {class_names[pred]}",
            fontsize=10,
        )
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Ocultar ejes sobrantes
    for i in range(n_cases, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Guardado: {save_path}")

    plt.show()


def visualize_interesting_cases_with_gradcam(
    model: nn.Module,
    loader,
    mc_results: Dict,
    interesting_cases: Dict,
    device: torch.device,
    save_dir: Optional[Path] = None,
):
    """
    Visualiza casos interesantes con Grad-CAM.

    Args:
        model: Modelo PyTorch
        loader: DataLoader
        mc_results: Resultados de MC Dropout
        interesting_cases: Output de find_interesting_cases
        device: Device
        save_dir: Directorio para guardar
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Obtener todos los specs
    all_specs = []
    for batch in loader:
        all_specs.append(batch["spectrogram"])
    all_specs = torch.cat(all_specs, dim=0)

    class_names = ["HC", "PD"]
    last_conv = get_last_conv_layer(model)
    gradcam = GradCAM(model, last_conv)

    # Visualizar cada categoría
    for category, indices in interesting_cases.items():
        if not indices:
            continue

        print(f"\n📊 Visualizando: {category}")

        n_show = min(3, len(indices))
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices[:n_show]):
            spec = all_specs[idx : idx + 1].to(device)
            label = mc_results["labels"][idx]
            pred = mc_results["predictions"][idx]
            entropy_val = mc_results["entropy"][idx]

            # Generar CAM
            cam = gradcam.generate_cam(spec, target_class=pred)
            cam_np = cam.squeeze().cpu().numpy()
            spec_np = spec.squeeze().cpu().numpy()

            # Original
            axes[i, 0].imshow(spec_np, aspect="auto", origin="lower", cmap="viridis")
            axes[i, 0].set_title("Original")
            axes[i, 0].set_ylabel("Mel bins")

            # CAM
            axes[i, 1].imshow(cam_np, aspect="auto", origin="lower", cmap="jet")
            axes[i, 1].set_title("Grad-CAM")

            # Superposición
            axes[i, 2].imshow(spec_np, aspect="auto", origin="lower", cmap="gray")
            im = axes[i, 2].imshow(
                cam_np, aspect="auto", origin="lower", cmap="jet", alpha=0.5
            )
            axes[i, 2].set_title(
                f"Real: {class_names[label]} | Pred: {class_names[pred]}\n"
                f"Entropy: {entropy_val:.3f}"
            )

            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

            if i == n_show - 1:
                for ax in axes[i]:
                    ax.set_xlabel("Frames")

        plt.suptitle(
            f"Casos: {category.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"gradcam_{category}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 Guardado: {save_path}")

        plt.show()


# ============================================================
# VISUALIZACIÓN DE INCERTIDUMBRE
# ============================================================


def plot_uncertainty_distribution(mc_results: Dict, save_path: Optional[Path] = None):
    """
    Visualiza distribución de incertidumbre.

    Args:
        mc_results: Resultados de MC Dropout
        save_path: Ruta para guardar
    """
    entropy = mc_results["entropy"]
    variance = mc_results["variance"]
    labels = mc_results["labels"]
    predictions = mc_results["predictions"]

    correct = predictions == labels
    class_names = ["HC", "PD"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribución de entropía
    axes[0, 0].hist(entropy[labels == 0], bins=30, alpha=0.6, label="HC", density=True)
    axes[0, 0].hist(entropy[labels == 1], bins=30, alpha=0.6, label="PD", density=True)
    axes[0, 0].set_xlabel("Entropía")
    axes[0, 0].set_ylabel("Densidad")
    axes[0, 0].set_title("Distribución de Entropía por Clase")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Entropía: correcto vs incorrecto
    axes[0, 1].boxplot(
        [entropy[correct], entropy[~correct]], labels=["Correcto", "Incorrecto"]
    )
    axes[0, 1].set_ylabel("Entropía")
    axes[0, 1].set_title("Entropía: Predicciones Correctas vs Incorrectas")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Varianza por clase
    axes[1, 0].hist(variance[labels == 0], bins=30, alpha=0.6, label="HC", density=True)
    axes[1, 0].hist(variance[labels == 1], bins=30, alpha=0.6, label="PD", density=True)
    axes[1, 0].set_xlabel("Varianza")
    axes[1, 0].set_ylabel("Densidad")
    axes[1, 0].set_title("Distribución de Varianza por Clase")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Scatter: Entropía vs Confianza
    confidence = mc_results["probabilities_mean"].max(axis=1)

    colors = ["green" if c else "red" for c in correct]
    axes[1, 1].scatter(confidence, entropy, c=colors, alpha=0.5, s=10)
    axes[1, 1].set_xlabel("Confianza (max prob)")
    axes[1, 1].set_ylabel("Entropía")
    axes[1, 1].set_title("Confianza vs Incertidumbre")
    axes[1, 1].grid(True, alpha=0.3)

    # Leyenda personalizada
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.5, label="Correcto"),
        Patch(facecolor="red", alpha=0.5, label="Incorrecto"),
    ]
    axes[1, 1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Guardado: {save_path}")

    plt.show()


def plot_aggregated_results(file_results: Dict, save_path: Optional[Path] = None):
    """
    Visualiza resultados agregados por archivo.

    Args:
        file_results: Resultados agregados
        save_path: Ruta para guardar
    """
    predictions = file_results["file_predictions"]
    labels = file_results["file_labels"]
    uncertainty = file_results["file_uncertainty"]
    probs = file_results["file_probabilities"]

    correct = predictions == labels
    class_names = ["HC", "PD"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Incertidumbre por archivo
    axes[0, 0].bar(
        range(len(uncertainty)), uncertainty, color=["g" if c else "r" for c in correct]
    )
    axes[0, 0].set_xlabel("Archivo ID")
    axes[0, 0].set_ylabel("Incertidumbre")
    axes[0, 0].set_title("Incertidumbre por Archivo (Verde=Correcto, Rojo=Error)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. Distribución de incertidumbre
    axes[0, 1].hist(
        uncertainty[correct], bins=15, alpha=0.6, label="Correcto", density=True
    )
    axes[0, 1].hist(
        uncertainty[~correct], bins=15, alpha=0.6, label="Incorrecto", density=True
    )
    axes[0, 1].set_xlabel("Incertidumbre")
    axes[0, 1].set_ylabel("Densidad")
    axes[0, 1].set_title("Distribución de Incertidumbre")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confianza por clase
    confidence = probs.max(axis=1)

    axes[1, 0].boxplot(
        [confidence[labels == 0], confidence[labels == 1]], labels=class_names
    )
    axes[1, 0].set_ylabel("Confianza")
    axes[1, 0].set_title("Confianza por Clase Real")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Matriz de confusión con incertidumbre
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1, 1],
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axes[1, 1].set_xlabel("Predicción")
    axes[1, 1].set_ylabel("Real")
    axes[1, 1].set_title("Matriz de Confusión (Nivel Archivo)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Guardado: {save_path}")

    plt.show()


# ============================================================
# VISUALIZACIÓN DE ENTRENAMIENTO
# ============================================================


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """
    Visualiza historial de entrenamiento.

    Args:
        history: Dict con métricas de entrenamiento
        save_path: Ruta para guardar
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Evolución de Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Evolución de Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1-Score
    axes[2].plot(epochs, history["train_f1"], "b-", label="Train", linewidth=2)
    axes[2].plot(epochs, history["val_f1"], "r-", label="Val", linewidth=2)
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_title("Evolución de F1-Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Guardado: {save_path}")

    plt.show()


# ============================================================
# REPORTE VISUAL COMPLETO
# ============================================================


def generate_visual_report(
    model: nn.Module,
    loader,
    mc_results: Dict,
    file_results: Dict,
    history: Dict,
    save_dir: Path,
):
    """
    Genera reporte visual completo.

    Args:
        model: Modelo entrenado
        loader: DataLoader de test
        mc_results: Resultados MC Dropout
        file_results: Resultados agregados
        history: Historial de entrenamiento
        save_dir: Directorio para guardar
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERANDO REPORTE VISUAL COMPLETO")
    print("=" * 70)

    # 1. Historial de entrenamiento
    print("\n📈 Gráfica de entrenamiento...")
    plot_training_history(history, save_path=save_dir / "training_history.png")

    # 2. Distribución de incertidumbre
    print("\n📊 Distribución de incertidumbre...")
    plot_uncertainty_distribution(
        mc_results, save_path=save_dir / "uncertainty_distribution.png"
    )

    # 3. Resultados agregados
    print("\n📁 Resultados agregados por archivo...")
    plot_aggregated_results(file_results, save_path=save_dir / "aggregated_results.png")

    # 4. Casos interesantes con Grad-CAM
    print("\n🔍 Casos interesantes con Grad-CAM...")
    from .cnn_inference import find_interesting_cases

    interesting_cases = find_interesting_cases(mc_results, n_per_category=5)

    device = next(model.parameters()).device
    visualize_interesting_cases_with_gradcam(
        model,
        loader,
        mc_results,
        interesting_cases,
        device,
        save_dir=save_dir / "gradcam",
    )

    print(f"\n✅ Reporte visual completo guardado en: {save_dir}")
    print("=" * 70 + "\n")
