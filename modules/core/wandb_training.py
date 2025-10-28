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
) -> Dict[str, Any]:
    """
    Entrenar modelo con monitoreo en tiempo real usando wandb.

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

    Returns:
        Diccionario con resultados del entrenamiento
    """
    if verbose:
        print("=" * 70)
        print("INICIANDO ENTRENAMIENTO CON MONITOREO WANDB")
        print("=" * 70)

    # Importar funciones de entrenamiento
    from modules.models.cnn2d.training import train_one_epoch, evaluate

    model.train()
    best_val_f1 = 0.0
    patience_counter = 0
    training_history = {
        "train_loss": [],
        "train_f1": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        # Entrenar una época
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluar
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Guardar en historial
        training_history["train_loss"].append(train_metrics["loss"])
        training_history["train_f1"].append(train_metrics["f1"])
        training_history["train_accuracy"].append(train_metrics["accuracy"])
        training_history["train_precision"].append(train_metrics["precision"])
        training_history["train_recall"].append(train_metrics["recall"])
        training_history["val_loss"].append(val_metrics["loss"])
        training_history["val_f1"].append(val_metrics["f1"])
        training_history["val_accuracy"].append(val_metrics["accuracy"])
        training_history["val_precision"].append(val_metrics["precision"])
        training_history["val_recall"].append(val_metrics["recall"])
        training_history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Loggear métricas a wandb y local
        monitor.log(
            epoch=epoch + 1,
            train_loss=train_metrics["loss"],
            train_f1=train_metrics["f1"],
            train_accuracy=train_metrics["accuracy"],
            train_precision=train_metrics["precision"],
            train_recall=train_metrics["recall"],
            val_loss=val_metrics["loss"],
            val_f1=val_metrics["f1"],
            val_accuracy=val_metrics["accuracy"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        # Plotear localmente cada N épocas
        if monitor.should_plot(epoch + 1):
            monitor.plot_local()

        # Early stopping basado en val_f1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            # Guardar mejor modelo
            if save_dir:
                torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_counter += 1

        # Aplicar scheduler
        if scheduler:
            scheduler.step()

        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch + 1}")
                print(f"    Mejor val_f1: {best_val_f1:.4f}")
            break

        # Imprimir progreso
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"Época {epoch + 1:3d}/{epochs} | "
                f"Train F1: {train_metrics['f1']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    # Finalizar monitoreo
    monitor.finish()
    monitor.print_summary()

    return {
        "model": model,
        "best_val_f1": best_val_f1,
        "final_epoch": epoch + 1,
        "history": training_history,
        "early_stopped": patience_counter >= early_stopping_patience,
    }


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
