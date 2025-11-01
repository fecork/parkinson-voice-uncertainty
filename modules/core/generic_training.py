#!/usr/bin/env python3
"""
Funciones de entrenamiento genéricas para cualquier arquitectura
================================================================

Funciones de entrenamiento que funcionan con cualquier modelo PyTorch,
independientemente de la arquitectura (CNN1D, CNN2D, LSTM, etc.).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def train_one_epoch_generic(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Entrena una época de forma genérica para cualquier modelo.
    
    Args:
        model: Modelo PyTorch a entrenar
        loader: DataLoader de entrenamiento
        optimizer: Optimizador
        criterion: Función de pérdida
        device: Dispositivo (CPU/GPU)
        forward_fn: Función personalizada para forward pass (opcional)
                   Si no se proporciona, usa model(batch) directamente
    
    Returns:
        Dict con métricas: loss, accuracy, recall (sensitivity), specificity, f1
    """
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        # Extraer datos del batch
        if isinstance(batch, dict):
            # Formato dict (recomendado)
            if "spectrogram" in batch:
                inputs = batch["spectrogram"].to(device)
                labels = batch["label"].to(device)
            elif "X" in batch:
                inputs = batch["X"].to(device)
                labels = batch["y"].to(device)
            else:
                # Buscar la primera clave que no sea label/y
                input_key = [k for k in batch.keys() if k not in ["label", "y"]][0]
                inputs = batch[input_key].to(device)
                labels = batch.get("label", batch.get("y")).to(device)
        else:
            # Formato tuple (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        # Forward pass
        if forward_fn is not None:
            logits = forward_fn(model, inputs)
        else:
            logits = model(inputs)
        
        # Calcular pérdida
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples
    
    # Calcular métricas según paper Ibarra 2023: ACC, SEN, SPE, F1
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0),
    }
    
    return metrics


@torch.no_grad()
def evaluate_generic(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evalúa el modelo de forma genérica.
    
    Args:
        model: Modelo PyTorch a evaluar
        loader: DataLoader de evaluación
        criterion: Función de pérdida
        device: Dispositivo (CPU/GPU)
        forward_fn: Función personalizada para forward pass (opcional)
    
    Returns:
        Dict con métricas: loss, accuracy, recall (sensitivity), specificity, f1
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        # Extraer datos del batch
        if isinstance(batch, dict):
            # Formato dict (recomendado)
            if "spectrogram" in batch:
                inputs = batch["spectrogram"].to(device)
                labels = batch["label"].to(device)
            elif "X" in batch:
                inputs = batch["X"].to(device)
                labels = batch["y"].to(device)
            else:
                # Buscar la primera clave que no sea label/y
                input_key = [k for k in batch.keys() if k not in ["label", "y"]][0]
                inputs = batch[input_key].to(device)
                labels = batch.get("label", batch.get("y")).to(device)
        else:
            # Formato tuple (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        # Forward pass
        if forward_fn is not None:
            logits = forward_fn(model, inputs)
        else:
            logits = model(inputs)
        
        # Calcular pérdida
        loss = criterion(logits, labels)
        
        # Métricas
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples
    
    # Calcular métricas según paper Ibarra 2023: ACC, SEN, SPE, F1
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0),
    }
    
    return metrics


def get_architecture_specific_functions(architecture: str) -> tuple:
    """
    Obtiene las funciones de entrenamiento específicas para una arquitectura.
    
    Args:
        architecture: Nombre de la arquitectura ("cnn1d", "cnn2d", "lstm", etc.)
    
    Returns:
        Tuple con (train_one_epoch_fn, evaluate_fn, train_model_fn)
    """
    if architecture.lower() == "cnn2d":
        from modules.models.cnn2d.training import train_one_epoch, evaluate, train_model
        return train_one_epoch, evaluate, train_model
    elif architecture.lower() == "cnn1d":
        from modules.models.cnn1d.training import train_one_epoch, evaluate, train_model
        return train_one_epoch, evaluate, train_model
    elif architecture.lower() == "lstm":
        from modules.models.lstm.training import train_one_epoch, evaluate, train_model
        return train_one_epoch, evaluate, train_model
    elif architecture.lower() == "lstm_da":
        from modules.models.lstm_da.training import train_one_epoch_da, evaluate_da, train_model_da
        return train_one_epoch_da, evaluate_da, train_model_da
    elif architecture.lower() == "cnn2d_da":
        from modules.models.cnn2d.training import train_one_epoch_da, evaluate_da, train_model_da
        return train_one_epoch_da, evaluate_da, train_model_da
    else:
        # Usar funciones genéricas como fallback
        return train_one_epoch_generic, evaluate_generic, None
