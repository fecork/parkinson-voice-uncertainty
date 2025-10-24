"""
Utilidades de Entrenamiento Comunes
==================================
Herramientas compartidas para entrenamiento de todas las arquitecturas:
- EarlyStopping
- Funciones de evaluación
- Métricas comunes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# EARLY STOPPING
# ============================================================


class EarlyStopping:
    """
    Early stopping para detener entrenamiento cuando no mejora.
    Utilizado por todas las arquitecturas: CNN2D, CNN1D, LSTM.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Épocas a esperar antes de detener
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' para minimizar, 'max' para maximizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == "min":
            self.compare_fn = lambda score, best: score < (best - min_delta)
        else:
            self.compare_fn = lambda score, best: score > (best + min_delta)

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Actualiza early stopping.

        Args:
            score: Métrica actual
            epoch: Época actual

        Returns:
            True si se debe detener
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.compare_fn(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


# ============================================================
# MÉTRICAS DE EVALUACIÓN
# ============================================================


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas de clasificación estándar.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_proba: Probabilidades (opcional)
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Métricas por clase
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics


def compute_class_weights_auto(labels: torch.Tensor, threshold: float = 0.4) -> Optional[torch.Tensor]:
    """
    Calcula pesos de clase automáticamente si hay desbalance.
    
    Args:
        labels: Tensor con etiquetas
        threshold: Umbral mínimo de balance (0.4 = 40%)
        
    Returns:
        Pesos de clase o None si está balanceado
    """
    unique, counts = torch.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_samples = len(labels)
    
    # Calcular balance mínimo
    min_balance = counts.min().float() / n_samples
    
    if min_balance >= threshold:
        return None  # Está balanceado
    
    # Calcular pesos inversamente proporcionales
    weights = n_samples / (n_classes * counts.float())
    return weights


# ============================================================
# UTILIDADES DE MODELO
# ============================================================


def count_parameters(model: nn.Module) -> int:
    """
    Cuenta parámetros entrenables del modelo.

    Args:
        model: Modelo PyTorch

    Returns:
        Número de parámetros
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str = "MODELO"):
    """
    Imprime resumen del modelo.

    Args:
        model: Modelo PyTorch
        model_name: Nombre del modelo para el título
    """
    print("\n" + "=" * 70)
    print(f"ARQUITECTURA DEL {model_name}")
    print("=" * 70)
    print(model)
    print("\n" + "-" * 70)
    print(f"Parámetros totales: {count_parameters(model):,}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print("-" * 70 + "\n")


# ============================================================
# UTILIDADES DE ENTRENAMIENTO
# ============================================================


def save_training_results(
    results: Dict,
    save_path: str,
    model_name: str = "model",
    include_metrics: bool = True
) -> None:
    """
    Guarda resultados de entrenamiento en archivo JSON.
    
    Args:
        results: Diccionario con resultados
        save_path: Ruta donde guardar
        model_name: Nombre del modelo
        include_metrics: Si incluir métricas detalladas
    """
    import json
    from pathlib import Path
    
    # Convertir tensors a listas para JSON
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    # Agregar metadatos
    serializable_results['model_name'] = model_name
    serializable_results['timestamp'] = str(Path().cwd())
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Resultados guardados en: {save_path}")


def load_training_results(load_path: str) -> Dict:
    """
    Carga resultados de entrenamiento desde archivo JSON.
    
    Args:
        load_path: Ruta del archivo
        
    Returns:
        Diccionario con resultados
    """
    import json
    
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    return results


