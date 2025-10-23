"""
Visualization Module
====================
Funciones para visualizar señales de audio, espectrogramas y resultados.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import Audio, display


# ============================================================
# AUDIO & SPECTROGRAM VISUALIZATION
# ============================================================


def visualize_audio_and_spectrograms(
    dataset: List[Dict],
    num_samples: int = 3,
    sr: int = 44100,
    figsize_per_sample: Tuple[int, int] = (5, 12),
    show: bool = True,
    play_audio: bool = True,
) -> Tuple[plt.Figure, List[Audio]]:
    """
    Visualiza señales de audio originales y sus espectrogramas procesados.

    Args:
        dataset: Lista de muestras procesadas
        num_samples: Número de muestras a visualizar
        sr: Sample rate para reproducción de audio
        figsize_per_sample: Tamaño de figura por muestra
        show: Si True, muestra las figuras inmediatamente
        play_audio: Si True, reproduce los audios

    Returns:
        fig: Figura de matplotlib
        audios: Lista de objetos Audio para reproducción
    """
    if not dataset:
        print("❌ No hay datos para visualizar")
        return None, []

    samples = random.sample(dataset, min(num_samples, len(dataset)))

    # Crear figura
    fig, axes = plt.subplots(
        3,
        num_samples,
        figsize=(figsize_per_sample[0] * num_samples, figsize_per_sample[1]),
    )

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    # Lista para almacenar objetos Audio
    audio_objects = []

    for i, sample in enumerate(samples):
        # Extraer metadatos y datos
        meta = sample["metadata"]
        segment = sample["segment"]
        spectrogram = sample["spectrogram"]

        # 1. Audio original (segmento de 400ms)
        time_axis = np.linspace(0, len(segment) / sr, len(segment))
        axes[0, i].plot(time_axis, segment, linewidth=1.5, color="steelblue")
        axes[0, i].set_title(
            f"Audio Original {i + 1}\n{meta.vowel_type} - {meta.condition}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, i].set_xlabel("Tiempo (s)", fontsize=10)
        axes[0, i].set_ylabel("Amplitud", fontsize=10)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(-1, 1)

        # 2. Espectrograma Mel (procesado)
        im = axes[1, i].imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="bilinear",
        )
        axes[1, i].set_title(
            f"Espectrograma Mel {i + 1}\n{spectrogram.shape} (65×41)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, i].set_xlabel("Frames (10ms)", fontsize=10)
        axes[1, i].set_ylabel("Bandas Mel", fontsize=10)
        axes[1, i].grid(True, alpha=0.3)
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        # 3. Comparación: Audio vs Espectrograma
        axes[2, i].plot(time_axis, segment, linewidth=1.5, color="steelblue", alpha=0.7)
        axes[2, i].set_title(
            f"Comparación {i + 1}\nAudio: {len(segment) / sr:.2f}s → Espectrograma: {spectrogram.shape[1]} frames",
            fontsize=12,
            fontweight="bold",
        )
        axes[2, i].set_xlabel("Tiempo (s)", fontsize=10)
        axes[2, i].set_ylabel("Amplitud", fontsize=10)
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].set_ylim(-1, 1)

        # Añadir líneas verticales para frames del espectrograma
        frame_duration = 0.01  # 10ms por frame
        for frame in range(0, spectrogram.shape[1], 5):
            time_frame = frame * frame_duration
            if time_frame <= len(segment) / sr:
                axes[2, i].axvline(
                    x=time_frame, color="red", alpha=0.3, linestyle="--", linewidth=0.5
                )

        # Crear objeto Audio
        audio_obj = Audio(segment, rate=sr)
        audio_objects.append(
            {
                "audio": audio_obj,
                "filename": meta.filename,
                "vowel": meta.vowel_type,
                "condition": meta.condition,
            }
        )

    plt.tight_layout()

    if show:
        plt.show()

    # Reproducir audio si se solicita
    if play_audio:
        print("\n🔊 Reproduciendo audios:")
        for i, audio_info in enumerate(audio_objects):
            print(
                f"  {i + 1}. {audio_info['filename']} - {audio_info['vowel']} - {audio_info['condition']}"
            )
            display(audio_info["audio"])

    # Mostrar información del procesamiento (último sample)
    last_sample = samples[-1]
    last_meta = last_sample["metadata"]
    print(f"\n📊 INFORMACIÓN DEL PROCESAMIENTO:")
    print(
        f"  - Audio original: {len(last_sample['segment']) / sr:.2f}s (400ms por segmento)"
    )
    print(
        f"  - Espectrograma: {last_sample['spectrogram'].shape[0]}×{last_sample['spectrogram'].shape[1]} (65 bandas × 41 frames)"
    )
    print(f"  - Frames temporales: {last_sample['spectrogram'].shape[1]} (cada 10ms)")
    print(
        f"  - Ventana FFT: {'40ms' if last_meta.vowel_type == 'a' else '25ms'} para vocal {last_meta.vowel_type}"
    )
    print(f"  - Normalización: z-score aplicada")

    return fig, audio_objects


def plot_spectrogram_comparison(
    spectrograms: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 4),
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Figure:
    """
    Compara múltiples espectrogramas lado a lado.

    Args:
        spectrograms: Lista de espectrogramas a comparar
        titles: Títulos para cada espectrograma
        figsize: Tamaño de la figura
        cmap: Colormap a usar
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    n = len(spectrograms)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, (spec, title) in enumerate(zip(spectrograms, titles)):
        im = axes[i].imshow(spec, aspect="auto", origin="lower", cmap=cmap)
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Frames")
        axes[i].set_ylabel("Mel Bands")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_waveform(
    audio: np.ndarray,
    sr: int = 44100,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza forma de onda de audio.

    Args:
        audio: Señal de audio
        sr: Sample rate
        title: Título del plot
        figsize: Tamaño de la figura
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio, linewidth=0.8, color="steelblue")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Amplitud", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mel_spectrogram(
    spectrogram: np.ndarray,
    sr: int = 44100,
    hop_length: int = 441,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza un espectrograma Mel.

    Args:
        spectrogram: Espectrograma Mel
        sr: Sample rate
        hop_length: Hop length usado en el espectrograma
        title: Título del plot
        figsize: Tamaño de la figura
        cmap: Colormap
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Frames", fontsize=11)
    ax.set_ylabel("Mel Bands", fontsize=11)

    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================
# DATASET STATISTICS VISUALIZATION
# ============================================================


def plot_label_distribution(
    y_task: np.ndarray,
    y_domain: Optional[np.ndarray] = None,
    task_labels: List[str] = ["Control", "Parkinson"],
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza distribución de etiquetas del dataset.

    Args:
        y_task: Etiquetas de tarea
        y_domain: Etiquetas de dominio (opcional)
        task_labels: Nombres de las clases
        figsize: Tamaño de la figura
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    import torch
    from collections import Counter

    # Convertir a numpy si es tensor
    if hasattr(y_task, "numpy"):
        y_task = y_task.numpy()

    n_plots = 2 if y_domain is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot distribución de tareas
    task_counts = Counter(y_task)
    labels = [
        task_labels[i] if i < len(task_labels) else f"Class {i}"
        for i in sorted(task_counts.keys())
    ]
    counts = [task_counts[i] for i in sorted(task_counts.keys())]

    axes[0].bar(labels, counts, color=["#3498db", "#e74c3c"])
    axes[0].set_title("Distribución de Clases", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Número de Muestras")
    axes[0].grid(True, alpha=0.3, axis="y")

    for i, (label, count) in enumerate(zip(labels, counts)):
        axes[0].text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

    # Plot distribución de dominios si está disponible
    if y_domain is not None:
        if hasattr(y_domain, "numpy"):
            y_domain = y_domain.numpy()

        domain_counts = Counter(y_domain)
        domain_labels = [f"Domain {i}" for i in sorted(domain_counts.keys())]
        domain_values = [domain_counts[i] for i in sorted(domain_counts.keys())]

        axes[1].bar(range(len(domain_labels)), domain_values, color="#9b59b6")
        axes[1].set_title("Distribución de Dominios", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Dominio")
        axes[1].set_ylabel("Número de Muestras")
        axes[1].set_xticks(range(len(domain_labels)))
        axes[1].set_xticklabels(domain_labels, rotation=45)
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_sample_spectrograms_grid(
    spectrograms: np.ndarray,
    num_samples: int = 9,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza una cuadrícula de espectrogramas.

    Args:
        spectrograms: Tensor de espectrogramas (N, C, H, W) o (N, H, W)
        num_samples: Número de muestras a mostrar
        titles: Títulos opcionales para cada subplot
        figsize: Tamaño de la figura
        cmap: Colormap
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    import torch

    # Convertir a numpy si es necesario
    if isinstance(spectrograms, torch.Tensor):
        spectrograms = spectrograms.numpy()

    # Remover dimensión de canal si existe
    if spectrograms.ndim == 4:
        spectrograms = spectrograms[:, 0, :, :]  # (N, H, W)

    num_samples = min(num_samples, len(spectrograms))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        im = axes[i].imshow(spectrograms[i], aspect="auto", origin="lower", cmap=cmap)
        title = titles[i] if titles and i < len(titles) else f"Sample {i + 1}"
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel("Frames", fontsize=9)
        axes[i].set_ylabel("Mel Bands", fontsize=9)

    # Ocultar ejes vacíos
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================
# AUGMENTATION COMPARISON
# ============================================================


def compare_original_vs_augmented(
    original: np.ndarray,
    augmented: np.ndarray,
    original_title: str = "Original",
    augmented_title: str = "Augmented",
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Figure:
    """
    Compara espectrograma original vs aumentado.

    Args:
        original: Espectrograma original
        augmented: Espectrograma aumentado
        original_title: Título para el original
        augmented_title: Título para el aumentado
        figsize: Tamaño de la figura
        cmap: Colormap
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original
    im1 = axes[0].imshow(original, aspect="auto", origin="lower", cmap=cmap)
    axes[0].set_title(original_title, fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Mel Bands")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Augmented
    im2 = axes[1].imshow(augmented, aspect="auto", origin="lower", cmap=cmap)
    axes[1].set_title(augmented_title, fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mel Bands")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def compare_audio_waveforms(
    audios: Dict[str, np.ndarray],
    sr: int = 44100,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    play_audio: bool = True,
) -> Tuple[plt.Figure, List[Audio]]:
    """
    Compara múltiples formas de onda de audio.

    Args:
        audios: Diccionario {nombre: señal_audio}
        sr: Sample rate
        figsize: Tamaño de la figura
        show: Si True, muestra la figura
        play_audio: Si True, reproduce los audios

    Returns:
        Figura de matplotlib y lista de objetos Audio
    """
    n = len(audios)
    fig, axes = plt.subplots(n, 1, figsize=figsize)

    if n == 1:
        axes = [axes]

    audio_objects = []

    for i, (name, audio) in enumerate(audios.items()):
        time = np.linspace(0, len(audio) / sr, len(audio))
        axes[i].plot(time, audio, linewidth=0.8)
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Tiempo (s)")
        axes[i].set_ylabel("Amplitud")
        axes[i].grid(True, alpha=0.3)

        audio_objects.append({"name": name, "audio": Audio(audio, rate=sr)})

    plt.tight_layout()

    if show:
        plt.show()

    if play_audio:
        print("\n🔊 Reproduciendo audios:")
        for audio_info in audio_objects:
            print(f"\n{audio_info['name']}:")
            display(audio_info["audio"])

    return fig, audio_objects


# ============================================================
# TRAINING VISUALIZATION
# ============================================================


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ["loss", "accuracy"],
    figsize: Tuple[int, int] = (14, 5),
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza historial de entrenamiento.

    Args:
        history: Diccionario con métricas por época
        metrics: Lista de métricas a plotear
        figsize: Tamaño de la figura
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        if train_key in history:
            axes[i].plot(history[train_key], label="Train", linewidth=2)
        if val_key in history:
            axes[i].plot(history[val_key], label="Validation", linewidth=2)

        axes[i].set_title(
            f"{metric.capitalize()} History", fontsize=12, fontweight="bold"
        )
        axes[i].set_xlabel("Época")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ["Control", "Parkinson"],
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza matriz de confusión.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        labels: Nombres de las clases
        figsize: Tamaño de la figura
        cmap: Colormap
        show: Si True, muestra la figura

    Returns:
        Figura de matplotlib
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def create_audio_player(audio: np.ndarray, sr: int = 44100) -> Audio:
    """
    Crea un objeto Audio para reproducción en notebook.

    Args:
        audio: Señal de audio
        sr: Sample rate

    Returns:
        Objeto Audio de IPython
    """
    return Audio(audio, rate=sr)


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    """
    Guarda una figura en archivo.

    Args:
        fig: Figura de matplotlib
        filepath: Ruta donde guardar
        dpi: Resolución
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"💾 Figura guardada en: {filepath}")
