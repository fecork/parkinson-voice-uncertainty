"""
CNN2D Talos Wrapper
===================
Wrapper específico para CNN2D que implementa la interfaz TalosModelWrapper.

Este módulo conecta CNN2D con el sistema de optimización central,
permitiendo búsqueda de hiperparámetros con Talos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
# from sklearn.metrics import f1_score, accuracy_score  # No usado directamente

from .talos_optimization import TalosModelWrapper
from ..models.cnn2d.model import CNN2D
from ..models.cnn2d.training import train_one_epoch, evaluate


class CNN2DTalosWrapper(TalosModelWrapper):
    """
    Wrapper para CNN2D compatible con Talos.

    Implementa la interfaz TalosModelWrapper para permitir
    optimización de hiperparámetros con Talos.
    """

    def __init__(self):
        """Inicializar wrapper para CNN2D."""
        super().__init__()

    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """
        Crear modelo CNN2D con parámetros específicos.

        Args:
            params: Diccionario con hiperparámetros

        Returns:
            Modelo CNN2D configurado
        """
        return CNN2D(
            n_classes=2,
            p_drop_conv=params["p_drop_conv"],
            p_drop_fc=params["p_drop_fc"],
            input_shape=(65, 41),
            filters_1=params["filters_1"],
            filters_2=params["filters_2"],
            kernel_size_1=params["kernel_size_1"],
            kernel_size_2=params["kernel_size_2"],
            dense_units=params["dense_units"],
        )

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        params: Dict[str, Any],
        n_epochs: int = 20,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Entrenar modelo CNN2D y retornar métricas.

        Args:
            model: Modelo CNN2D
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            params: Parámetros de entrenamiento
            n_epochs: Número de épocas

        Returns:
            Tuple con (f1_score, metrics_dict)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Optimizador y función de pérdida
        optimizer = optim.Adam(model.parameters(), lr=0.1)  # Learning rate fijo
        criterion = nn.CrossEntropyLoss()

        # Scheduler para reducir learning rate automáticamente
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_f1 = 0.0
        best_metrics = {}

        for epoch in range(n_epochs):
            # Entrenar una época
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Evaluar en validación
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Actualizar scheduler con validation loss
            scheduler.step(val_metrics["loss"])

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

        return best_f1, best_metrics

    def get_search_params(self) -> Dict[str, list]:
        """
        Retornar espacio de búsqueda de hiperparámetros para CNN2D.

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
            # learning_rate se maneja con scheduler, no es hiperparámetro
        }


def create_cnn2d_optimizer(
    experiment_name: str = "cnn2d_talos_optimization",
    round_limit: int = None,
    fraction_limit: float = None,
    search_method: str = "random",
) -> "TalosOptimizer":
    """
    Crear optimizador Talos para CNN2D.

    Args:
        experiment_name: Nombre del experimento
        round_limit: Número exacto de configuraciones a evaluar
        fraction_limit: Fracción de combinaciones a evaluar
        search_method: Método de búsqueda ('random', 'sobol', etc.)

    Returns:
        TalosOptimizer configurado para CNN2D
    """
    from .talos_optimization import TalosOptimizer

    wrapper = CNN2DTalosWrapper()
    return TalosOptimizer(
        model_wrapper=wrapper,
        experiment_name=experiment_name,
        round_limit=round_limit,
        fraction_limit=fraction_limit,
        search_method=search_method,
    )


def optimize_cnn2d(
    X_train,
    y_train,
    X_val,
    y_val,
    experiment_name: str = "cnn2d_talos_optimization",
    round_limit: int = None,
    fraction_limit: float = None,
    search_method: str = "random",
    n_epochs: int = 20,
) -> Dict[str, Any]:
    """
    Función de conveniencia para optimizar CNN2D con Talos.

    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        experiment_name: Nombre del experimento
        round_limit: Número exacto de configuraciones a evaluar
        fraction_limit: Fracción de combinaciones a evaluar
        search_method: Método de búsqueda ('random', 'sobol', etc.)
        n_epochs: Número de épocas por configuración

    Returns:
        Diccionario con resultados de la optimización
    """
    optimizer = create_cnn2d_optimizer(
        experiment_name, round_limit, fraction_limit, search_method
    )

    # Ejecutar optimización
    results_df = optimizer.optimize(X_train, y_train, X_val, y_val, n_epochs)

    # Obtener mejores parámetros
    best_params = optimizer.get_best_params()

    # Análisis de resultados
    analysis = optimizer.analyze_results()

    return {
        "results_df": results_df,
        "best_params": best_params,
        "analysis": analysis,
        "optimizer": optimizer,
    }
