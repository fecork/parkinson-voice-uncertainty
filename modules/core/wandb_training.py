#!/usr/bin/env python3
"""
Funciones de entrenamiento con Weights & Biases
==============================================

Funciones especializadas para entrenar modelos con monitoreo en tiempo real.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from modules.core.training_monitor import TrainingMonitor

# Importar la versión genérica
from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic


def train_with_wandb_monitoring(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    monitor: TrainingMonitor,
    device: torch.device,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    save_dir: Optional[Path] = None,
    model_name: str = "best_model_wandb.pth",
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Entrenar modelo con monitoreo en tiempo real usando wandb (GENÉRICO).
    
    Esta función ahora es genérica y funciona con cualquier arquitectura:
    - CNN1D, CNN2D, LSTM, etc.
    - Detecta automáticamente la arquitectura basada en el modelo
    - Usa funciones específicas si están disponibles, sino genéricas

    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader para entrenamiento
        val_loader: DataLoader para validación
        optimizer: Optimizador PyTorch
        criterion: Función de pérdida
        scheduler: Scheduler de learning rate (opcional)
        monitor: Monitor de entrenamiento con wandb
        device: Dispositivo (CPU/GPU)
        epochs: Número máximo de épocas
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar el modelo
        model_name: Nombre del archivo del modelo
        verbose: Si imprimir progreso
        **kwargs: Parámetros específicos de arquitectura (alpha, lambda_, etc.)

    Returns:
        Diccionario con resultados del entrenamiento
    """
    # Usar la versión genérica
    return train_with_wandb_monitoring_generic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        monitor=monitor,
        device=device,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        model_name=model_name,
        verbose=verbose,
        **kwargs
    )


def create_training_config(
    experiment_name: str = "cnn2d_training",
    use_wandb: bool = True,
    plot_every: int = 5,
    save_plots: bool = True,
    model_architecture: str = "CNN2D",
    dataset: str = "Parkinson Voice",
    optimization: str = "Optuna",
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crear configuración estándar para entrenamiento con wandb.

    Args:
        experiment_name: Nombre del experimento
        use_wandb: Si usar Weights & Biases
        plot_every: Cada cuántas épocas plotear
        save_plots: Si guardar gráficos
        model_architecture: Arquitectura del modelo
        dataset: Nombre del dataset
        optimization: Tipo de optimización
        best_params: Mejores parámetros encontrados

    Returns:
        Diccionario con configuración
    """
    config = {
        "experiment_name": experiment_name,
        "use_wandb": use_wandb,
        "plot_every": plot_every,
        "save_plots": save_plots,
        "model_architecture": model_architecture,
        "dataset": dataset,
        "optimization": optimization,
    }

    if best_params:
        config.update(best_params)

    return config


def setup_wandb_training(
    config: Dict[str, Any],
    wandb_config: Dict[str, Any],
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 65, 41),
) -> TrainingMonitor:
    """
    Configurar monitoreo de entrenamiento con wandb.

    Args:
        config: Configuración del experimento
        wandb_config: Configuración de wandb
        model: Modelo a monitorear
        input_shape: Forma de entrada del modelo

    Returns:
        Monitor de entrenamiento configurado
    """
    from modules.core.training_monitor import create_training_monitor

    # Crear monitor
    monitor = create_training_monitor(
        config=config,
        experiment_name=config["experiment_name"],
        use_wandb=config["use_wandb"],
        wandb_key=wandb_config["api_key"],
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
    )

    # Registrar modelo en wandb
    if config["use_wandb"]:
        monitor.log_model(model, input_shape=input_shape)

    return monitor
