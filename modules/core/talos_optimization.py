"""
Talos Hyperparameter Optimization Core Module
==============================================
Módulo central para optimización de hiperparámetros usando Talos.

Este módulo proporciona funcionalidades reutilizables para cualquier arquitectura:
- Wrapper genérico para Talos
- Funciones de evaluación
- Análisis de resultados
- Integración con diferentes modelos

Usage:
    optimizer = TalosOptimizer(model_class=CNN2D, search_params=params)
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
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
from typing import Dict, Tuple, Any, Callable, Optional
import json
import pandas as pd
from abc import ABC, abstractmethod


class TalosModelWrapper(ABC):
    """
    Clase abstracta para wrappers de modelos compatibles con Talos.

    Cada arquitectura debe implementar esta interfaz para ser compatible
    con el sistema de optimización de Talos.
    """

    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """Crear modelo con parámetros específicos."""
        pass

    @abstractmethod
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        params: Dict[str, Any],
        n_epochs: int = 20,
    ) -> Tuple[float, Dict[str, float]]:
        """Entrenar modelo y retornar métricas."""
        pass

    @abstractmethod
    def get_search_params(self) -> Dict[str, list]:
        """Retornar espacio de búsqueda de hiperparámetros."""
        pass


class TalosOptimizer:
    """
    Optimizador principal de Talos para cualquier arquitectura.

    Esta clase proporciona una interfaz unificada para optimización
    de hiperparámetros usando Talos con cualquier modelo.
    """

    def __init__(
        self,
        model_wrapper: TalosModelWrapper,
        experiment_name: str = "talos_optimization",
        fraction_limit: float = None,
        round_limit: int = None,
        search_method: str = "random",
        random_method: str = "sobol",
        seed: int = 42,
    ):
        """
        Args:
            model_wrapper: Wrapper del modelo que implementa TalosModelWrapper
            experiment_name: Nombre del experimento
            fraction_limit: Fracción de combinaciones a evaluar (0.1 = 10%)
            round_limit: Número exacto de configuraciones a evaluar
            search_method: Método de búsqueda ('random', 'sobol', etc.)
            random_method: Método de muestreo para Talos
            seed: Semilla para reproducibilidad
        """
        self.model_wrapper = model_wrapper
        self.experiment_name = experiment_name
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.search_method = search_method
        self.random_method = random_method
        self.seed = seed
        self.results = None

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 20,
    ) -> pd.DataFrame:
        """
        Ejecutar optimización de hiperparámetros.

        Args:
            X_train: Datos de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Datos de validación
            y_val: Labels de validación
            n_epochs: Número de épocas por configuración

        Returns:
            DataFrame con resultados de la optimización
        """
        try:
            import talos
        except ImportError:
            raise ImportError("Talos no está instalado. Instala con: pip install talos")

        # Obtener parámetros de búsqueda
        search_params = self.model_wrapper.get_search_params()

        # Crear función wrapper para Talos
        def talos_model_function(x_train, y_train, x_val, y_val, params):
            return self.model_wrapper.train_model(
                self.model_wrapper.create_model(params),
                self._create_dataloader(x_train, y_train, params["batch_size"]),
                self._create_dataloader(x_val, y_val, params["batch_size"]),
                params,
                n_epochs,
            )

        # Ejecutar búsqueda Talos
        print("Iniciando optimización con Talos...")
        if self.round_limit is not None:
            print(f"Configuraciones a evaluar: {self.round_limit}")
        elif self.fraction_limit is not None:
            print(f"Fracción a evaluar: {self.fraction_limit * 100:.0f}%")

        # Preparar kwargs para Scan
        scan_kwargs = {
            "x": X_train,
            "y": y_train,
            "x_val": X_val,
            "y_val": y_val,
            "params": search_params,
            "model": talos_model_function,
            "experiment_name": self.experiment_name,
            "search_method": self.search_method,
            "random_method": self.random_method,
            "seed": self.seed,
        }

        # Agregar fraction_limit o round_limit según corresponda
        if self.round_limit is not None:
            scan_kwargs["round_limit"] = self.round_limit
        elif self.fraction_limit is not None:
            scan_kwargs["fraction_limit"] = self.fraction_limit
        else:
            scan_kwargs["fraction_limit"] = 0.1  # Default

        scan_object = talos.Scan(**scan_kwargs)

        self.results = scan_object.data
        return self.results

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> DataLoader:
        """Crear DataLoader a partir de arrays numpy."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def get_best_params(self) -> Dict[str, Any]:
        """Obtener mejores parámetros encontrados."""
        if self.results is None:
            raise ValueError("No se ha ejecutado la optimización aún.")

        best_idx = self.results["f1"].idxmax()
        return self.results.loc[best_idx].to_dict()

    def analyze_results(self, top_n: int = 10) -> Dict[str, Any]:
        """Analizar resultados de la optimización."""
        if self.results is None:
            raise ValueError("No se ha ejecutado la optimización aún.")

        return analyze_hyperparameter_importance(self.results, top_n)

    def print_summary(self, top_n: int = 10):
        """Imprimir resumen de resultados."""
        if self.results is None:
            raise ValueError("No se ha ejecutado la optimización aún.")

        print_optimization_summary(self.results, top_n)


def get_default_search_params() -> Dict[str, list]:
    """
    Parámetros de búsqueda por defecto para modelos de deep learning.

    Returns:
        Diccionario con parámetros comunes para optimización
    """
    return {
        "batch_size": [16, 32, 64],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "dropout_rate": [0.2, 0.3, 0.5],
        "weight_decay": [0.0, 1e-4, 1e-5],
        "optimizer": ["adam", "sgd"],
    }


def create_talos_model_wrapper(
    model_class, search_params: Dict[str, list], train_function: Callable
) -> TalosModelWrapper:
    """
    Crear wrapper genérico para cualquier modelo.

    Args:
        model_class: Clase del modelo a optimizar
        search_params: Parámetros de búsqueda
        train_function: Función de entrenamiento personalizada

    Returns:
        Wrapper compatible con Talos
    """

    class GenericModelWrapper(TalosModelWrapper):
        def __init__(self, model_class, search_params, train_function):
            self.model_class = model_class
            self.search_params = search_params
            self.train_function = train_function

        def create_model(self, params: Dict[str, Any]) -> nn.Module:
            return self.model_class(**params)

        def train_model(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            params: Dict[str, Any],
            n_epochs: int = 20,
        ) -> Tuple[float, Dict[str, float]]:
            return self.train_function(
                model, train_loader, val_loader, params, n_epochs
            )

        def get_search_params(self) -> Dict[str, list]:
            return self.search_params

    return GenericModelWrapper(model_class, search_params, train_function)


def evaluate_best_model(
    model_wrapper: TalosModelWrapper,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    save_path: Optional[str] = None,
    n_epochs: int = 100,
    patience: int = 10,
) -> Dict[str, Any]:
    """
    Re-entrenar el mejor modelo con early stopping y evaluar en test set.

    Args:
        model_wrapper: Wrapper del modelo
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test
        best_params: Mejores parámetros encontrados
        save_path: Ruta para guardar el modelo (opcional)
        n_epochs: Máximo número de épocas
        patience: Paciencia para early stopping

    Returns:
        Diccionario con resultados finales
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Crear modelo
    model = model_wrapper.create_model(best_params).to(device)

    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    from ..models.common.training_utils import EarlyStopping

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    # Entrenar con early stopping
    train_history = {"loss": [], "f1": [], "accuracy": []}
    val_history = {"loss": [], "f1": [], "accuracy": []}

    for epoch in range(n_epochs):
        # Entrenar una época
        train_metrics = model_wrapper.train_model(
            model, train_loader, val_loader, best_params, 1
        )

        # Evaluar en validación
        val_metrics = _evaluate_model(model, val_loader, criterion, device)

        # Guardar historial
        train_history["loss"].append(train_metrics[1]["loss"])
        train_history["f1"].append(train_metrics[1]["f1"])
        train_history["accuracy"].append(train_metrics[1]["accuracy"])

        val_history["loss"].append(val_metrics["loss"])
        val_history["f1"].append(val_metrics["f1"])
        val_history["accuracy"].append(val_metrics["accuracy"])

        # Early stopping
        early_stopping(val_metrics["loss"])
        if early_stopping.early_stop:
            print(f"Early stopping en época {epoch + 1}")
            break

    # Evaluar en test set
    test_metrics = _evaluate_model(model, test_loader, criterion, device)

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

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    test_metrics["auc"] = auc

    # Guardar modelo si se especifica
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

        # Guardar configuración
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


def _evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    """Evaluar modelo en un DataLoader."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            specs = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def analyze_hyperparameter_importance(
    results_df: pd.DataFrame, top_n: int = 10
) -> Dict[str, Any]:
    """
    Analizar importancia de hiperparámetros basado en correlaciones con F1-score.

    Args:
        results_df: DataFrame con resultados de Talos
        top_n: Número de top parámetros a retornar

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
                f1_correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                f1_correlations[col] = 0.0

    # Ordenar por importancia
    sorted_importance = sorted(
        f1_correlations.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "correlations": f1_correlations,
        "sorted_importance": sorted_importance,
        "top_important": sorted_importance[:top_n],
    }


def print_optimization_summary(results_df: pd.DataFrame, top_n: int = 10):
    """
    Imprimir resumen de la optimización de hiperparámetros.

    Args:
        results_df: DataFrame con resultados de Talos
        top_n: Número de mejores configuraciones a mostrar
    """
    print("=" * 80)
    print("RESUMEN DE OPTIMIZACIÓN DE HIPERPARÁMETROS")
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
        for param in ["batch_size", "learning_rate", "dropout_rate", "weight_decay"]:
            if param in row:
                print(f"     {param}: {row[param]}")

    # Análisis de importancia
    importance = analyze_hyperparameter_importance(results_df)

    top_important_count = min(5, len(importance["top_important"]))
    print(f"\nTop {top_important_count} hiperparámetros más importantes:")
    print("-" * 40)
    for param, corr in importance["top_important"][:5]:
        print(f"  {param}: {corr:.4f}")

    print("=" * 80)
