"""
Utility Functions Module
=========================
Funciones auxiliares comunes para el proyecto.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict


# ============================================================
# ENVIRONMENT DETECTION
# ============================================================


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab

        return True
    except ImportError:
        return False


def setup_colab_environment(mount_drive: bool = True):
    """
    Setup Google Colab environment.

    Args:
        mount_drive: Whether to mount Google Drive
    """
    if not is_colab():
        print("⚠️ No estás en Google Colab")
        return

    print("🚀 Configurando entorno de Google Colab...")

    # Install required packages
    packages = [
        "librosa",
        "soundfile",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {package}...")
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )

    # Mount Google Drive
    if mount_drive:
        print("📂 Montando Google Drive...")
        from google.colab import drive

        drive.mount("/content/drive")
        print("✅ Google Drive montado en /content/drive")


def get_data_path() -> str:
    """
    Get data path based on environment (Colab or local).

    Returns:
        Path to data directory
    """
    if is_colab():
        return "/content/drive/MyDrive/parkinson-voice-uncertainty/vowels"
    else:
        return "./vowels"


def get_modules_path() -> str:
    """
    Get modules path based on environment.

    Returns:
        Path to modules directory
    """
    if is_colab():
        return "/content/drive/MyDrive/parkinson-voice-uncertainty/modules"
    else:
        return "./modules"


def add_modules_to_path():
    """Add modules directory to Python path for imports."""
    modules_path = str(Path(get_modules_path()).parent)
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
        print(f"✅ Módulos agregados al path: {modules_path}")


# ============================================================
# FILE OPERATIONS
# ============================================================


def list_audio_files(data_path: str, extension: str = "*.egg") -> List[Path]:
    """
    List audio files in data directory.

    Args:
        data_path: Path to data directory
        extension: File extension pattern

    Returns:
        List of Path objects
    """
    path = Path(data_path)
    if not path.exists():
        print(f"❌ Directorio no encontrado: {data_path}")
        return []

    files = list(path.glob(extension))
    print(f"📁 Encontrados {len(files)} archivos {extension}")
    return files


def ensure_directory(path: str):
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY UTILITIES
# ============================================================


def print_section_header(title: str, width: int = 60):
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Width of the separator line
    """
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_config(config: dict, title: str = "Configuration"):
    """
    Print configuration dictionary in a formatted way.

    Args:
        config: Configuration dictionary
        title: Title for the configuration
    """
    print(f"\n⚙️ {title}:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    • {sub_key}: {sub_value}")
        else:
            print(f"  • {key}: {value}")


def print_dataset_stats(X, y_task, y_domain, metadata: Optional[List] = None):
    """
    Print dataset statistics.

    Args:
        X: Feature tensor
        y_task: Task labels
        y_domain: Domain labels
        metadata: Optional metadata list
    """
    import torch
    from collections import Counter

    print("\n📊 Dataset Statistics:")
    print(f"  Shape: {X.shape}")
    print(f"  Samples: {len(X)}")

    if hasattr(y_task, "numpy"):
        task_dist = Counter(y_task.numpy())
        domain_dist = Counter(y_domain.numpy())
    else:
        task_dist = Counter(y_task)
        domain_dist = Counter(y_domain)

    print(f"\n  Task Distribution:")
    for label, count in task_dist.items():
        print(f"    • Class {label}: {count} ({count / len(y_task) * 100:.1f}%)")

    print(f"\n  Domain Distribution:")
    for label, count in domain_dist.items():
        print(f"    • Domain {label}: {count} ({count / len(y_domain) * 100:.1f}%)")

    if metadata:
        print(f"\n  Metadata: {len(metadata)} entries available")


# ============================================================
# EXPERIMENT TRACKING
# ============================================================


def save_experiment_config(config: dict, save_path: str):
    """
    Save experiment configuration to file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    import json

    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuración guardada en: {save_path}")


def load_experiment_config(load_path: str) -> dict:
    """
    Load experiment configuration from file.

    Args:
        load_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    import json

    with open(load_path, "r") as f:
        config = json.load(f)
    print(f"📂 Configuración cargada desde: {load_path}")
    return config


# ============================================================
# K-FOLD CROSS-VALIDATION
# ============================================================


def create_10fold_splits_by_speaker(
    metadata_list: List[dict], n_folds: int = 10, seed: int = 42
) -> List[Dict[str, List[int]]]:
    """
    Crea 10 folds estratificados independientes por hablante.

    Asegura que:
    - Todos los segmentos de un hablante están en el mismo fold
    - Cada fold está estratificado por etiqueta PD (balanceado HC/PD)
    - Sin fugas de hablante entre train/val

    Args:
        metadata_list: Lista de metadatos con 'subject_id' y 'label'
        n_folds: Número de folds (default: 10)
        seed: Semilla para reproducibilidad

    Returns:
        Lista de dicts con splits: [{'train': [...], 'val': [...]}, ...]
    """
    from sklearn.model_selection import StratifiedKFold

    # Agrupar por subject_id
    subject_to_indices = {}
    subject_to_label = {}

    for idx, meta in enumerate(metadata_list):
        subject_id = meta.get("subject_id", meta.get("filename", f"unknown_{idx}"))
        label = meta.get("label", 0)

        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
            subject_to_label[subject_id] = label

        subject_to_indices[subject_id].append(idx)

    # Preparar arrays para StratifiedKFold
    subjects = list(subject_to_indices.keys())
    labels = [subject_to_label[subj] for subj in subjects]

    subjects = np.array(subjects)
    labels = np.array(labels)

    # Crear folds estratificados sobre hablantes
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_splits = []

    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(
        skf.split(subjects, labels)
    ):
        train_subjects = subjects[train_subject_idx]
        val_subjects = subjects[val_subject_idx]

        # Obtener índices de muestras
        train_indices = [
            idx for subj in train_subjects for idx in subject_to_indices[subj]
        ]
        val_indices = [idx for subj in val_subjects for idx in subject_to_indices[subj]]

        fold_splits.append({"train": train_indices, "val": val_indices})

    print(f"\n📊 10-Fold CV speaker-independent creado:")
    print(f"   Total hablantes: {len(subjects)}")
    print(f"   Total muestras: {len(metadata_list)}")
    print(f"   Folds: {n_folds}")

    # Estadísticas de primer fold
    fold_1 = fold_splits[0]
    print(f"\n   Fold 1 (ejemplo):")
    print(f"      Train: {len(fold_1['train'])} muestras")
    print(f"      Val:   {len(fold_1['val'])} muestras")

    return fold_splits
