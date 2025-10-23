"""
Uncertainty Module
==================
Módulos para modelado de incertidumbre (epistémica y aleatoria).
"""

from .model import UncertaintyCNN, print_uncertainty_model_summary
from .loss import heteroscedastic_classification_loss
from .training import (
    train_uncertainty_model,
    evaluate_with_uncertainty,
    print_uncertainty_results,
)
from .visualization import (
    plot_uncertainty_histograms,
    plot_reliability_diagram,
    plot_uncertainty_scatter,
    plot_training_history_uncertainty,
)

__all__ = [
    "UncertaintyCNN",
    "print_uncertainty_model_summary",
    "heteroscedastic_classification_loss",
    "train_uncertainty_model",
    "evaluate_with_uncertainty",
    "print_uncertainty_results",
    "plot_uncertainty_histograms",
    "plot_reliability_diagram",
    "plot_uncertainty_scatter",
    "plot_training_history_uncertainty",
]
