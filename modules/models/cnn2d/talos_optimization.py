"""
Talos Hyperparameter Optimization for CNN2D (Legacy)
====================================================
Módulo legacy para optimización de hiperparámetros usando Talos con CNN2D.

NOTA: Este módulo está deprecado. Usar modules/core/talos_optimization.py
y modules/models/cnn2d/talos_wrapper.py en su lugar.

Funciones:
- get_search_params(): Espacio de búsqueda de hiperparámetros
- talos_train_function(): Función de entrenamiento para Talos
- evaluate_best_model(): Re-entrenamiento del mejor modelo
- create_talos_model(): Wrapper para crear modelo con parámetros específicos
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing import Dict, Tuple, Any
import json
from pathlib import Path

# Importar módulos propios
from .model import CNN2D
from .training import train_one_epoch, evaluate
from ..common.training_utils import EarlyStopping, compute_class_weights_auto

# Importar nuevo sistema
from ...core.talos_optimization import TalosOptimizer
from .talos_wrapper import CNN2DTalosWrapper, create_cnn2d_optimizer, optimize_cnn2d


def get_search_params() -> Dict[str, list]:
    """
    Define el espacio de búsqueda de hiperparámetros para Talos.

    Returns:
        Diccionario con parámetros y sus valores posibles
    """
    return {
        "batch_size": [16, 32, 64],
        "p_drop_conv": [0.2, 0.5],
        "p_drop_fc": [0.2, 0.5],
        "filters_1": [32, 64, 128],
        "filters_2": [32, 64, 128],
        "kernel_size_1": [4, 6, 8],
        "kernel_size_2": [5, 7, 9],
        "dense_units": [16, 32, 64],
        "learning_rate": [0.001, 0.0001],
    }


def create_talos_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Función wrapper para Talos que crea, entrena y evalúa un modelo CNN2D.

    Args:
        x_train: Datos de entrenamiento (N, 1, H, W)
        y_train: Labels de entrenamiento (N,)
        x_val: Datos de validación (M, 1, H, W)
        y_val: Labels de validación (M,)
        params: Diccionario con hiperparámetros de Talos

    Returns:
        Tuple con (f1_score, metrics_dict)
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convertir a tensores PyTorch
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Crear DataLoaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0
    )

    # Crear modelo
    model = CNN2D(
        n_classes=2,
        p_drop_conv=params["p_drop_conv"],
        p_drop_fc=params["p_drop_fc"],
        input_shape=(65, 41),
        filters_1=params["filters_1"],
        filters_2=params["filters_2"],
        kernel_size_1=params["kernel_size_1"],
        kernel_size_2=params["kernel_size_2"],
        dense_units=params["dense_units"],
    ).to(device)

    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Entrenar por N épocas (sin early stopping para búsqueda rápida)
    n_epochs = 20  # Número fijo para búsqueda

    best_f1 = 0.0
    best_metrics = {}

    for epoch in range(n_epochs):
        # Entrenar una época
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluar en validación
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Guardar mejor F1-score
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {
                "f1": val_metrics["f1"],
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "train_loss": train_metrics["loss"],
            }

    # Retornar F1-score como métrica principal para Talos
    return best_f1, best_metrics


def evaluate_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    save_path: str = None,
) -> Dict[str, Any]:
    """
    Re-entrena el mejor modelo con early stopping y evalúa en test set.

    Args:
        x_train: Datos de entrenamiento
        y_train: Labels de entrenamiento
        x_val: Datos de validación
        y_val: Labels de validación
        x_test: Datos de test
        y_test: Labels de test
        best_params: Mejores hiperparámetros encontrados
        save_path: Ruta para guardar el modelo (opcional)

    Returns:
        Diccionario con métricas finales
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convertir a tensores PyTorch
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(y_val)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Crear DataLoaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=0
    )

    # Crear modelo
    model = CNN2D(
        n_classes=2,
        p_drop_conv=best_params["p_drop_conv"],
        p_drop_fc=best_params["p_drop_fc"],
        input_shape=(65, 41),
        filters_1=best_params["filters_1"],
        filters_2=best_params["filters_2"],
        kernel_size_1=best_params["kernel_size_1"],
        kernel_size_2=best_params["kernel_size_2"],
        dense_units=best_params["dense_units"],
    ).to(device)

    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Entrenar con early stopping
    train_history = {"loss": [], "f1": [], "accuracy": []}
    val_history = {"loss": [], "f1": [], "accuracy": []}

    for epoch in range(100):  # Máximo 100 épocas
        # Entrenar una época
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluar en validación
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Guardar historial
        train_history["loss"].append(train_metrics["loss"])
        train_history["f1"].append(train_metrics["f1"])
        train_history["accuracy"].append(train_metrics["accuracy"])

        val_history["loss"].append(val_metrics["loss"])
        val_history["f1"].append(val_metrics["f1"])
        val_history["accuracy"].append(val_metrics["accuracy"])

        # Early stopping
        early_stopping(val_metrics["loss"])
        if early_stopping.early_stop:
            print(f"Early stopping en época {epoch + 1}")
            break

    # Evaluar en test set
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Calcular AUC
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            specs = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(specs)
            probs = torch.softmax(logits, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidad clase positiva
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    test_metrics["auc"] = auc

    # Guardar modelo si se especifica ruta
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_params": best_params,
                "test_metrics": test_metrics,
                "train_history": train_history,
                "val_history": val_history,
            },
            save_path,
        )

        # Guardar configuración en JSON
        config_path = save_path.replace(".pth", "_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "test_metrics": test_metrics,
                    "final_epoch": epoch + 1,
                },
                f,
                indent=2,
            )

    return {
        "test_metrics": test_metrics,
        "train_history": train_history,
        "val_history": val_history,
        "final_epoch": epoch + 1,
        "best_params": best_params,
    }


def analyze_hyperparameter_importance(results_df):
    """
    Analiza la importancia de los hiperparámetros basado en los resultados de Talos.

    Args:
        results_df: DataFrame con resultados de Talos

    Returns:
        Diccionario con análisis de importancia
    """
    # Calcular correlaciones con F1-score
    f1_correlations = {}

    for col in results_df.columns:
        if col not in [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "val_loss",
            "train_loss",
        ]:
            try:
                corr = results_df[col].corr(results_df["f1"])
                f1_correlations[col] = abs(corr)
            except:
                f1_correlations[col] = 0.0

    # Ordenar por importancia
    sorted_importance = sorted(
        f1_correlations.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "correlations": f1_correlations,
        "sorted_importance": sorted_importance,
        "top_5_important": sorted_importance[:5],
    }


def print_optimization_summary(results_df, top_n=10):
    """
    Imprime un resumen de la optimización de hiperparámetros.

    Args:
        results_df: DataFrame con resultados de Talos
        top_n: Número de mejores configuraciones a mostrar
    """
    print("=" * 80)
    print("RESUMEN DE OPTIMIZACIÓN DE HIPERPARÁMETROS CNN2D")
    print("=" * 80)

    # Mejores configuraciones
    best_configs = results_df.nlargest(top_n, "f1")

    print(f"\nTop {top_n} configuraciones:")
    print("-" * 80)

    for i, (idx, row) in enumerate(best_configs.iterrows(), 1):
        print(f"\n{i}. F1-Score: {row['f1']:.4f}")
        print(f"   Accuracy: {row['accuracy']:.4f}")
        print(f"   Precision: {row['precision']:.4f}")
        print(f"   Recall: {row['recall']:.4f}")
        print(f"   Parámetros:")
        for param in [
            "batch_size",
            "p_drop_conv",
            "p_drop_fc",
            "filters_1",
            "filters_2",
            "kernel_size_1",
            "kernel_size_2",
            "dense_units",
            "learning_rate",
        ]:
            if param in row:
                print(f"     {param}: {row[param]}")

    # Análisis de importancia
    importance = analyze_hyperparameter_importance(results_df)

    print(f"\nTop 5 hiperparámetros más importantes:")
    print("-" * 40)
    for param, corr in importance["top_5_important"]:
        print(f"  {param}: {corr:.4f}")

    print("=" * 80)
