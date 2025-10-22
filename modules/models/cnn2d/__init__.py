"""
CNN 2D Module
=============
Módulos para modelo CNN 2D (con y sin Domain Adaptation).
"""

from .model import CNN2D, CNN2D_DA, GradientReversalLayer, print_model_summary
from .training import (
    train_model,
    detailed_evaluation,
    print_evaluation_report,
    save_training_results,
    train_model_da,
    train_model_da_kfold,
    evaluate_da,
)
from .inference import (
    mc_dropout_inference,
    aggregate_by_file,
    aggregate_by_patient,
    analyze_uncertainty,
    print_inference_report,
)
from .visualization import generate_visual_report
from .utils import (
    split_by_speaker,
    create_dataloaders_from_existing,
    compute_class_weights_from_dataset,
    compute_class_weights_auto,
    plot_confusion_matrix,
)

__all__ = [
    "CNN2D",
    "CNN2D_DA",
    "GradientReversalLayer",
    "print_model_summary",
    "train_model",
    "detailed_evaluation",
    "print_evaluation_report",
    "save_training_results",
    "train_model_da",
    "train_model_da_kfold",
    "evaluate_da",
    "mc_dropout_inference",
    "aggregate_by_file",
    "aggregate_by_patient",
    "analyze_uncertainty",
    "print_inference_report",
    "generate_visual_report",
    "split_by_speaker",
    "create_dataloaders_from_existing",
    "compute_class_weights_from_dataset",
    "compute_class_weights_auto",
    "plot_confusion_matrix",
]
