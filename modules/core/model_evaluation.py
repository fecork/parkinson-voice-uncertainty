"""
Model Evaluation Core Module
============================
Módulo central para evaluación y comparación de modelos.

Este módulo proporciona funcionalidades para:
- Evaluación estándar de modelos
- Comparación entre diferentes arquitecturas
- Métricas de rendimiento
- Visualización de resultados
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json


class ModelEvaluator:
    """
    Evaluador estándar para modelos de deep learning.

    Proporciona evaluación completa con métricas estándar
    y visualizaciones para cualquier modelo.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model: Modelo PyTorch a evaluar
            device: Dispositivo para cómputo (auto-detecta si no se especifica)
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluar modelo en un conjunto de datos.

        Args:
            X: Datos de entrada (N, C, H, W)
            y: Labels (N,)
            batch_size: Tamaño de batch para evaluación

        Returns:
            Diccionario con métricas de evaluación
        """
        self.model.eval()

        # Crear DataLoader
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                specs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                logits = self.model(specs)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calcular métricas
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, [p[1] for p in all_probs]),
        }

        return metrics

    def get_confusion_matrix(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 32
    ) -> np.ndarray:
        """Obtener matriz de confusión."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                specs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                logits = self.model(specs)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return confusion_matrix(all_labels, all_preds)

    def get_classification_report(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 32
    ) -> str:
        """Obtener reporte de clasificación detallado."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                specs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                logits = self.model(specs)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return classification_report(all_labels, all_preds)


def compare_models(
    models: Dict[str, nn.Module],
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Comparar múltiples modelos en el mismo conjunto de test.

    Args:
        models: Diccionario con nombre_modelo -> modelo
        X_test: Datos de test
        y_test: Labels de test
        batch_size: Tamaño de batch

    Returns:
        DataFrame con métricas de todos los modelos
    """
    results = []

    for name, model in models.items():
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test, batch_size)
        metrics["model"] = name
        results.append(metrics)

    return pd.DataFrame(results)


def save_model_results(
    model: nn.Module,
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Dict[str, Any] = None,
):
    """
    Guardar modelo y resultados de evaluación.

    Args:
        model: Modelo PyTorch
        metrics: Métricas de evaluación
        save_path: Ruta base para guardar archivos
        additional_info: Información adicional a guardar
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "additional_info": additional_info or {},
        },
        str(save_path) + ".pth",
    )

    # Guardar métricas en JSON
    results = {"metrics": metrics, "additional_info": additional_info or {}}

    with open(str(save_path) + "_results.json", "w") as f:
        json.dump(results, f, indent=2)


def plot_model_comparison(
    results_df: pd.DataFrame, metric: str = "f1", title: str = "Comparación de Modelos"
):
    """
    Crear gráfica de comparación de modelos.

    Args:
        results_df: DataFrame con resultados de modelos
        metric: Métrica a graficar
        title: Título de la gráfica
    """
    plt.figure(figsize=(10, 6))

    if "model" in results_df.columns:
        plt.bar(results_df["model"], results_df[metric], alpha=0.7)
        plt.xlabel("Modelo")
    else:
        plt.bar(range(len(results_df)), results_df[metric], alpha=0.7)
        plt.xlabel("Índice del Modelo")

    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_heatmap(
    conf_matrix: np.ndarray,
    class_names: List[str] = None,
    title: str = "Matriz de Confusión",
):
    """
    Crear heatmap de matriz de confusión.

    Args:
        conf_matrix: Matriz de confusión
        class_names: Nombres de las clases
        title: Título de la gráfica
    """
    plt.figure(figsize=(8, 6))

    if class_names is None:
        class_names = [f"Clase {i}" for i in range(len(conf_matrix))]

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.tight_layout()
    plt.show()


def create_evaluation_report(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crear reporte completo de evaluación.

    Args:
        model: Modelo a evaluar
        X_test: Datos de test
        y_test: Labels de test
        class_names: Nombres de las clases
        save_path: Ruta para guardar reporte (opcional)

    Returns:
        Diccionario con reporte completo
    """
    evaluator = ModelEvaluator(model)

    # Métricas básicas
    metrics = evaluator.evaluate(X_test, y_test)

    # Matriz de confusión
    conf_matrix = evaluator.get_confusion_matrix(X_test, y_test)

    # Reporte de clasificación
    class_report = evaluator.get_classification_report(X_test, y_test)

    # Crear reporte
    report = {
        "metrics": metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "class_names": class_names or ["Clase 0", "Clase 1"],
        "n_samples": len(y_test),
        "n_classes": len(np.unique(y_test)),
    }

    # Guardar si se especifica ruta
    if save_path:
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
