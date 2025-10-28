#!/usr/bin/env python3
"""
Configuración centralizada para experimentos
===========================================

Configuraciones centralizadas para todos los experimentos del proyecto.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuración base para experimentos."""

    # Configuración del optimizador (SGD como en el paper de Ibarra)
    optimizer_type: str = "SGD"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True

    # Configuración del scheduler
    scheduler_type: str = "StepLR"
    step_size: int = 10
    gamma: float = 0.1

    # Configuración de entrenamiento
    n_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 32
    num_workers: int = 0

    # Configuración de datos
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify: bool = True

    # Configuración de class weights
    use_class_weights: bool = True
    class_weights_method: str = "inverse_frequency"

    # Configuración de filtrado de vocal
    filter_vocal_a: bool = True
    target_vocal: str = "a"

    # Configuración de K-fold
    n_splits: int = 10
    shuffle: bool = True
    stratify_by_speaker: bool = True


# Configuraciones predefinidas
CNN2D_BASELINE_CONFIG = ExperimentConfig(
    experiment_name="cnn2d_baseline",
    n_epochs=100,
    early_stopping_patience=10,
    batch_size=32,
)

CNN2D_OPTUNA_CONFIG = ExperimentConfig(
    experiment_name="cnn2d_optuna",
    n_epochs=100,
    early_stopping_patience=10,
    batch_size=32,
)

# Configuración de Optuna
OPTUNA_CONFIG = {
    "enabled": True,
    "experiment_name": "cnn2d_optuna_optimization",
    "n_trials": 30,
    "n_epochs_per_trial": 10,
    "metric": "f1",
    "direction": "maximize",
    "pruning_enabled": True,
    "pruning_patience": 3,
    "pruning_min_trials": 2,
}

# Configuración de Weights & Biases
WANDB_CONFIG = {
    "project_name": "parkinson-voice-uncertainty",
    "enabled": True,
    "api_key": "b452ba0c4bbe61d8c58e966aa86a9037ae19594e",
}


def get_experiment_config(experiment_type: str = "cnn2d_baseline") -> Dict[str, Any]:
    """
    Obtener configuración para un tipo de experimento.

    Args:
        experiment_type: Tipo de experimento

    Returns:
        Diccionario con configuración
    """
    configs = {
        "cnn2d_baseline": CNN2D_BASELINE_CONFIG,
        "cnn2d_optuna": CNN2D_OPTUNA_CONFIG,
    }

    if experiment_type not in configs:
        raise ValueError(f"Tipo de experimento no válido: {experiment_type}")

    config = configs[experiment_type]

    return {
        "experiment_name": config.experiment_name,
        "optimizer": {
            "type": config.optimizer_type,
            "learning_rate": config.learning_rate,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "nesterov": config.nesterov,
        },
        "scheduler": {
            "type": config.scheduler_type,
            "step_size": config.step_size,
            "gamma": config.gamma,
        },
        "training": {
            "n_epochs": config.n_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
        },
        "data": {
            "test_size": config.test_size,
            "val_size": config.val_size,
            "random_state": config.random_state,
            "stratify": config.stratify,
        },
        "class_weights": {
            "enabled": config.use_class_weights,
            "method": config.class_weights_method,
        },
        "vocal_filter": {
            "enabled": config.filter_vocal_a,
            "target_vocal": config.target_vocal,
        },
        "kfold": {
            "n_splits": config.n_splits,
            "shuffle": config.shuffle,
            "stratify_by_speaker": config.stratify_by_speaker,
        },
    }
