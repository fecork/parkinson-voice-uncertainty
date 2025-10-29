#!/usr/bin/env python3
"""
Entrenamiento genérico con Weights & Biases
==========================================

Funciones de entrenamiento genéricas que funcionan con cualquier arquitectura
(CNN1D, CNN2D, LSTM, etc.) con monitoreo de WanDB.
"""

import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from modules.core.training_monitor import TrainingMonitor
from modules.core.generic_training import (
    train_one_epoch_generic, 
    evaluate_generic,
    get_architecture_specific_functions
)


def train_with_wandb_monitoring_generic(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    monitor: TrainingMonitor,
    device: torch.device,
    architecture: str = "generic",
    epochs: int = 100,
    early_stopping_patience: int = 10,
    save_dir: Optional[Path] = None,
    model_name: str = "best_model_wandb.pth",
    verbose: bool = True,
    forward_fn: Optional[Callable] = None,
    **kwargs  # Para parámetros específicos de arquitectura (alpha, lambda_, etc.)
) -> Dict[str, Any]:
    """
    Entrenar modelo con monitoreo en tiempo real usando wandb (GENÉRICO).
    
    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader para entrenamiento
        val_loader: DataLoader para validación
        optimizer: Optimizador PyTorch
        criterion: Función de pérdida
        scheduler: Scheduler de learning rate (opcional)
        monitor: Monitor de entrenamiento con wandb
        device: Dispositivo (CPU/GPU)
        architecture: Arquitectura del modelo ("cnn1d", "cnn2d", "lstm", "generic")
        epochs: Número máximo de épocas
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar el modelo
        model_name: Nombre del archivo del modelo
        verbose: Si imprimir progreso
        forward_fn: Función personalizada para forward pass
        **kwargs: Parámetros específicos de arquitectura (alpha, lambda_, etc.)
    
    Returns:
        Diccionario con resultados del entrenamiento
    """
    if verbose:
        print("=" * 70)
        print(f"INICIANDO ENTRENAMIENTO GENÉRICO CON MONITOREO WANDB")
        print(f"Arquitectura: {architecture.upper()}")
        print("=" * 70)
    
    # Obtener funciones de entrenamiento específicas o genéricas
    train_model_fn = None
    if architecture.lower() != "generic":
        try:
            train_one_epoch_fn, evaluate_fn, train_model_fn = get_architecture_specific_functions(architecture)
            if verbose:
                if train_model_fn is not None:
                    print(f"✅ Usando train_model() completo para {architecture.upper()}")
                else:
                    print(f"✅ Usando funciones básicas para {architecture.upper()}")
        except ImportError as e:
            if verbose:
                print(f"⚠️  No se encontraron funciones específicas para {architecture}: {e}")
                print("   Usando funciones genéricas como fallback")
            train_one_epoch_fn, evaluate_fn = train_one_epoch_generic, evaluate_generic
            train_model_fn = None
    else:
        train_one_epoch_fn, evaluate_fn = train_one_epoch_generic, evaluate_generic
        train_model_fn = None
        if verbose:
            print("✅ Usando funciones genéricas")
    
    # Si tenemos train_model() completo, usarlo directamente
    if train_model_fn is not None:
        if verbose:
            print(f"🚀 Ejecutando entrenamiento completo con {architecture.upper()}")
        
        # Usar train_model() completo de la arquitectura específica
        if architecture.lower() in ["cnn2d_da", "lstm_da"]:
            # Para Domain Adaptation
            training_results = train_model_fn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion_pd=criterion,
                criterion_domain=kwargs.get("criterion_domain", criterion),
                device=device,
                n_epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                save_dir=save_dir,
                verbose=verbose,
                **kwargs
            )
        else:
            # Para arquitecturas normales
            training_results = train_model_fn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                n_epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                save_dir=save_dir,
                verbose=verbose,
                scheduler=scheduler,
                monitor_metric="f1"
            )
        
        # Convertir resultados al formato esperado por WanDB
        history = training_results["history"]
        
        # Loggear métricas a wandb
        for epoch in range(len(history["train_loss"])):
            monitor.log(
                epoch=epoch + 1,
                train_loss=history["train_loss"][epoch],
                train_f1=history["train_f1"][epoch],
                train_accuracy=history["train_acc"][epoch],
                val_loss=history["val_loss"][epoch],
                val_f1=history["val_f1"][epoch],
                val_accuracy=history["val_acc"][epoch],
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            
            # Plotear localmente cada N épocas
            if monitor.should_plot(epoch + 1):
                monitor.plot_local()
        
        # Finalizar monitoreo
        monitor.finish()
        monitor.print_summary()
        
        return {
            "model": training_results["model"],
            "best_val_f1": training_results.get("best_val_metric", 0.0),
            "final_epoch": len(history["train_loss"]),
            "history": {
                "train_loss": history["train_loss"],
                "train_f1": history["train_f1"],
                "train_accuracy": history["train_acc"],
                "train_precision": [0.0] * len(history["train_loss"]),  # No disponible en train_model
                "train_recall": [0.0] * len(history["train_loss"]),     # No disponible en train_model
                "val_loss": history["val_loss"],
                "val_f1": history["val_f1"],
                "val_accuracy": history["val_acc"],
                "val_precision": [0.0] * len(history["val_loss"]),      # No disponible en train_model
                "val_recall": [0.0] * len(history["val_loss"]),         # No disponible en train_model
                "learning_rate": [optimizer.param_groups[0]["lr"]] * len(history["train_loss"]),
            },
            "early_stopped": len(history["train_loss"]) < epochs,
        }
    
    # Si no tenemos train_model(), usar el loop manual
    if verbose:
        print(f"🔄 Ejecutando entrenamiento manual con funciones básicas")
    
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
        if architecture.lower() in ["cnn2d_da", "lstm_da"]:
            # Para arquitecturas con Domain Adaptation
            train_metrics = train_one_epoch_fn(
                model, train_loader, optimizer, criterion, 
                device, **kwargs
            )
        else:
            # Para arquitecturas normales
            if forward_fn is not None:
                train_metrics = train_one_epoch_fn(
                    model, train_loader, optimizer, criterion, device, forward_fn
                )
            else:
                train_metrics = train_one_epoch_fn(
                    model, train_loader, optimizer, criterion, device
                )
        
        # Evaluar
        if architecture.lower() in ["cnn2d_da", "lstm_da"]:
            # Para arquitecturas con Domain Adaptation
            val_metrics = evaluate_fn(
                model, val_loader, criterion, device, **kwargs
            )
        else:
            # Para arquitecturas normales
            if forward_fn is not None:
                val_metrics = evaluate_fn(
                    model, val_loader, criterion, device, forward_fn
                )
            else:
                val_metrics = evaluate_fn(
                    model, val_loader, criterion, device
                )
        
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


# Alias para compatibilidad hacia atrás
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
    Alias para compatibilidad hacia atrás.
    Detecta automáticamente la arquitectura basada en el tipo de modelo.
    """
    # Detectar arquitectura basada en el nombre del modelo
    model_name_str = model.__class__.__name__.lower()
    
    if "cnn2d" in model_name_str:
        architecture = "cnn2d"
    elif "cnn1d" in model_name_str:
        architecture = "cnn1d"
    elif "lstm" in model_name_str:
        architecture = "lstm"
    else:
        architecture = "generic"
    
    return train_with_wandb_monitoring_generic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        monitor=monitor,
        device=device,
        architecture=architecture,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        model_name=model_name,
        verbose=verbose,
        **kwargs
    )
