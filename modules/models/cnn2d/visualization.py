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

from .model import GradCAM, get_last_conv_layer


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


# ============================================================
# DOMAIN ADAPTATION VISUALIZATIONS
# ============================================================


def plot_da_training_progress(
    history: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza progreso de entrenamiento con Domain Adaptation
    con gráficas en tiempo real.

    Args:
        history: Dict con métricas de entrenamiento
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Progreso de Entrenamiento - Domain Adaptation", fontsize=16)

    epochs = range(1, len(history["train_loss_pd"]) + 1)

    # Plot 1: Pérdidas combinadas
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss_pd"], "b-", label="Train PD", linewidth=2)
    ax1.plot(epochs, history["val_loss_pd"], "b--", label="Val PD", linewidth=2)
    ax1.plot(
        epochs,
        history["train_loss_domain"],
        "r-",
        label="Train Domain",
        linewidth=1.5,
        alpha=0.6,
    )
    ax1.plot(
        epochs,
        history["val_loss_domain"],
        "r--",
        label="Val Domain",
        linewidth=1.5,
        alpha=0.6,
    )
    ax1.set_title("Pérdidas: PD + Domain")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # Plot 2: Accuracy y F1
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["val_acc_pd"], "g-", label="Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_f1_pd"], "m-", label="F1 Score", linewidth=2)
    ax2.set_title("Métricas de Validación (PD)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Score")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Plot 3: Lambda progression
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["lambda_values"], "purple", linewidth=2.5)
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.fill_between(epochs, 0, history["lambda_values"], alpha=0.2, color="purple")
    ax3.set_title("Scheduler de Lambda (GRL)")
    ax3.set_xlabel("Época")
    ax3.set_ylabel("λ")
    ax3.grid(alpha=0.3)

    # Plot 4: Razón de pérdidas
    ax4 = axes[1, 1]
    ratio = [
        d / (p + 1e-8)
        for p, d in zip(history["val_loss_pd"], history["val_loss_domain"])
    ]
    ax4.plot(epochs, ratio, "orange", linewidth=2)
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ratio = 1")
    ax4.set_title("Ratio: Loss_Domain / Loss_PD")
    ax4.set_xlabel("Época")
    ax4.set_ylabel("Ratio")
    ax4.legend(loc="best")
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Gráfica guardada en: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_domain_confusion(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_domains: int,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza matriz de confusión para predicción de dominios.

    Args:
        model: Modelo con DA
        loader: DataLoader con datos
        device: Device para cómputo
        n_domains: Número de dominios
        save_path: Ruta para guardar imagen
        show: Si True, muestra la gráfica

    Returns:
        Figura de matplotlib
    """
    model.eval()

    all_preds_domain = []
    all_labels_domain = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                specs, _, labels_domain = batch
            else:
                specs = batch["spectrogram"]
                labels_domain = batch.get("domain", torch.zeros(specs.size(0)))

            specs = specs.to(device)
            _, logits_domain = model(specs)
            preds_domain = logits_domain.argmax(dim=1)

            all_preds_domain.extend(preds_domain.cpu().numpy())
            all_labels_domain.extend(labels_domain.numpy())

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels_domain, all_preds_domain)

    # Visualizar (reducida si hay muchos dominios)
    fig, ax = plt.subplots(figsize=(max(10, n_domains * 0.5), max(8, n_domains * 0.4)))

    if n_domains > 15:
        # Solo mostrar proporción, no anotar valores
        sns.heatmap(
            cm, cmap="Blues", cbar=True, square=True, linewidths=0.5, ax=ax, annot=False
        )
    else:
        sns.heatmap(
            cm,
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot=True,
            fmt="d",
        )

    ax.set_title("Matriz de Confusión - Clasificación de Dominios", fontsize=14)
    ax.set_xlabel("Dominio Predicho", fontsize=12)
    ax.set_ylabel("Dominio Real", fontsize=12)

    # Calcular accuracy de dominio
    accuracy_domain = (np.array(all_labels_domain) == np.array(all_preds_domain)).mean()
    ax.text(
        0.5,
        -0.1,
        f"Accuracy de Dominio: {accuracy_domain:.3f}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Matriz guardada en: {save_path}")

    if show:
        plt.show()

    return fig


def create_da_summary_report(
    model: nn.Module,
    history: Dict,
    test_results: Dict,
    save_dir: Path,
    model_name: str = "CNN2D_DA",
) -> None:
    """
    Crea reporte visual completo para modelo con DA.

    Args:
        model: Modelo entrenado con DA
        history: Historial de entrenamiento
        test_results: Resultados de evaluación en test
        save_dir: Directorio para guardar reportes
        model_name: Nombre del modelo
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"📊 GENERANDO REPORTE VISUAL - {model_name}")
    print("=" * 70)

    # 1. Historial de entrenamiento
    print("\n1. Historial de entrenamiento...")
    plot_da_training_progress(
        history, save_path=save_dir / "training_history.png", show=False
    )

    # 2. Matriz de confusión PD
    print("2. Matriz de confusión (PD)...")
    if "confusion_matrix" in test_results:
        from .cnn_utils import plot_confusion_matrix

        plot_confusion_matrix(
            test_results["confusion_matrix"],
            class_names=["HC", "PD"],
            title="Matriz de Confusión - Clasificación PD",
            save_path=save_dir / "confusion_matrix_pd.png",
            show=False,
        )

    # 3. Métricas finales
    print("3. Resumen de métricas...")
    create_metrics_summary_figure(
        history, test_results, save_path=save_dir / "metrics_summary.png"
    )

    print(f"\n✅ Reporte completo guardado en: {save_dir}")
    print("=" * 70 + "\n")


def create_metrics_summary_figure(
    history: Dict,
    test_results: Dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Crea figura resumen con métricas principales.

    Args:
        history: Historial de entrenamiento
        test_results: Resultados de test
        save_path: Ruta para guardar imagen

    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Título general
    fig.suptitle("Resumen de Métricas - Domain Adaptation", fontsize=16)

    # 1. Curvas de aprendizaje
    ax1 = fig.add_subplot(gs[0, :])
    epochs = range(1, len(history["val_loss_pd"]) + 1)
    ax1.plot(epochs, history["train_loss_pd"], "b-", label="Train Loss PD", alpha=0.7)
    ax1.plot(epochs, history["val_loss_pd"], "b-", linewidth=2, label="Val Loss PD")
    ax1.plot(epochs, history["train_f1_pd"], "g--", label="Train F1 PD", alpha=0.7)
    ax1.plot(epochs, history["val_f1_pd"], "g-", linewidth=2, label="Val F1 PD")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Valor")
    ax1.set_title("Curvas de Aprendizaje")
    ax1.legend(loc="best", ncol=2)
    ax1.grid(alpha=0.3)

    # 2. Métricas finales
    ax2 = fig.add_subplot(gs[1, 0])
    if "classification_report" in test_results:
        report = test_results["classification_report"]
        metrics_names = ["Accuracy", "F1 HC", "F1 PD", "F1 Macro"]
        metrics_values = [
            report["accuracy"],
            report["HC"]["f1-score"],
            report["PD"]["f1-score"],
            report["macro avg"]["f1-score"],
        ]

        bars = ax2.bar(
            metrics_names, metrics_values, color=["blue", "green", "red", "purple"]
        )
        ax2.set_ylim([0, 1.0])
        ax2.set_title("Métricas en Test Set")
        ax2.set_ylabel("Score")
        ax2.grid(axis="y", alpha=0.3)

        # Anotar valores
        for bar, val in zip(bars, metrics_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 3. Información del modelo
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    info_text = [
        "INFORMACIÓN DEL MODELO",
        "=" * 30,
        f"Mejor época: {np.argmin(history['val_loss_pd']) + 1}",
        f"Mejor Val Loss PD: {min(history['val_loss_pd']):.4f}",
        f"Mejor Val F1 PD: {max(history['val_f1_pd']):.4f}",
        "",
        "RESULTADOS EN TEST:",
        "=" * 30,
    ]

    if "classification_report" in test_results:
        report = test_results["classification_report"]
        info_text.extend(
            [
                f"Accuracy: {report['accuracy']:.4f}",
                f"F1 Macro: {report['macro avg']['f1-score']:.4f}",
                "",
                "Por clase:",
                f"  HC - P: {report['HC']['precision']:.3f} R: {report['HC']['recall']:.3f} F1: {report['HC']['f1-score']:.3f}",
                f"  PD - P: {report['PD']['precision']:.3f} R: {report['PD']['recall']:.3f} F1: {report['PD']['f1-score']:.3f}",
            ]
        )

    info_str = "\n".join(info_text)
    ax3.text(
        0.1,
        0.9,
        info_str,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Resumen guardado en: {save_path}")

    plt.show()

    return fig
