"""
CNN 1D Module
=============
Módulos para modelo CNN 1D con atención temporal y Domain Adaptation.
"""

from .model import CNN1D_DA, print_model_summary
from .training import (
    train_model_da,
    evaluate_da,
    evaluate_patient_level,
    save_metrics,
)
from .visualization import (
    plot_1d_training_progress,
    plot_tsne_embeddings,
    plot_simple_confusion_matrix,
)

__all__ = [
    "CNN1D_DA",
    "print_model_summary",
    "train_model_da",
    "evaluate_da",
    "evaluate_patient_level",
    "save_metrics",
    "plot_1d_training_progress",
    "plot_tsne_embeddings",
    "plot_simple_confusion_matrix",
]
