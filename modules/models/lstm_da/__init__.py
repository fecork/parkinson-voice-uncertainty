"""
Time-CNN-BiLSTM with Domain Adaptation Package
===============================================
Modelo Time-CNN-BiLSTM-DA para clasificación de Parkinson.
"""

from .model import TimeCNNBiLSTM_DA
from ..common.training_utils import count_parameters, print_model_summary
from .training import (
    train_one_epoch_da,
    validate_epoch_da,
    train_model_da_kfold,
    grl_lambda,
)
from .visualization import (
    plot_training_curves_da,
    plot_kfold_results,
    plot_lstm_attention_weights,
    plot_training_history,
)

__all__ = [
    "TimeCNNBiLSTM_DA",
    "count_parameters",
    "print_model_summary",
    "train_one_epoch_da",
    "validate_epoch_da",
    "train_model_da_kfold",
    "grl_lambda",
    "plot_training_curves_da",
    "plot_kfold_results",
    "plot_lstm_attention_weights",
    "plot_training_history",
]

