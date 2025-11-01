"""
Optuna Hyperparameter Optimization Core Module
===============================================
Módulo central para optimización de hiperparámetros usando Optuna.

Este módulo proporciona funcionalidades reutilizables para cualquier arquitectura:
- Wrapper genérico para Optuna
- Funciones de evaluación
- Análisis de resultados
- Integración con diferentes modelos

Ventajas sobre Talos:
- Más eficiente (pruning automático de trials malos)
- Mejor mantenido y actualizado
- Mejor integración con PyTorch
- Visualizaciones interactivas
- Soporte para distributed training

Usage:
    optimizer = OptunaOptimizer(model_class=CNN2D, search_params=params)
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
"""

import numpy as np
import torch
import torch.nn as nn
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
from pathlib import Path

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class OptunaModelWrapper(ABC):
    """
    Clase abstracta para wrappers de modelos compatibles con Optuna.

    Cada arquitectura debe implementar esta interfaz para ser compatible
    con el sistema de optimización de Optuna.
    """

    @abstractmethod
    def create_model(self, trial: Trial) -> nn.Module:
        """
        Crear modelo con parámetros sugeridos por Optuna.

        Args:
            trial: Objeto Trial de Optuna que sugiere hiperparámetros

        Returns:
            nn.Module: Modelo creado
        """
        pass

    @abstractmethod
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: Trial,
        n_epochs: int = 20,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Entrenar modelo y retornar métricas.

        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            trial: Trial de Optuna para reportar métricas intermedias
            n_epochs: Número de épocas

        Returns:
            tuple: (metric_principal, dict_metricas)
        """
        pass

    @abstractmethod
    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda de hiperparámetros.

        Args:
            trial: Objeto Trial de Optuna

        Returns:
            dict: Diccionario con hiperparámetros sugeridos
        """
        pass


class OptunaOptimizer:
    """
    Optimizador principal usando Optuna para cualquier arquitectura.

    Esta clase proporciona una interfaz unificada para optimización
    de hiperparámetros usando Optuna con cualquier modelo.
    """

    def __init__(
        self,
        model_wrapper: OptunaModelWrapper,
        experiment_name: str = "optuna_optimization",
        n_trials: int = 50,
        n_epochs_per_trial: int = 20,
        metric: str = "f1",
        direction: str = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        seed: int = 42,
        storage: Optional[str] = None,
    ):
        """
        Args:
            model_wrapper: Wrapper del modelo que implementa OptunaModelWrapper
            experiment_name: Nombre del experimento
            n_trials: Número de trials a ejecutar
            n_epochs_per_trial: Épocas por trial
            metric: Métrica a optimizar ('f1', 'accuracy', 'auc')
            direction: 'maximize' o 'minimize'
            pruner: Pruner de Optuna (None usa MedianPruner)
            sampler: Sampler de Optuna (None usa TPESampler)
            seed: Semilla para reproducibilidad
            storage: URL de storage para persistencia (None = memoria)
        """
        self.model_wrapper = model_wrapper
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.n_epochs_per_trial = n_epochs_per_trial
        self.metric = metric
        self.direction = direction
        self.seed = seed
        self.storage = storage

        # Configurar pruner (para early stopping automático)
        if pruner is None:
            self.pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1,
            )
        else:
            self.pruner = pruner

        # Configurar sampler (algoritmo de búsqueda)
        if sampler is None:
            self.sampler = TPESampler(seed=seed)
        else:
            self.sampler = sampler

        self.study = None
        self.results_df = None

    def _objective(
        self,
        trial: Trial,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> float:
        """
        Función objetivo para Optuna.

        Args:
            trial: Trial de Optuna
            X_train: Datos de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Datos de validación
            y_val: Labels de validación

        Returns:
            float: Valor de la métrica objetivo
        """
        # Crear modelo con hiperparámetros sugeridos
        model = self.model_wrapper.create_model(trial)

        # Crear DataLoaders
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Entrenar modelo
        metric_value, metrics_dict = self.model_wrapper.train_model(
            model, train_loader, val_loader, trial, self.n_epochs_per_trial
        )

        return metric_value

    def optimize(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Ejecutar optimización de hiperparámetros.

        Args:
            X_train: Tensor de datos de entrenamiento
            y_train: Tensor de labels de entrenamiento
            X_val: Tensor de datos de validación
            y_val: Tensor de labels de validación
            show_progress: Si mostrar barra de progreso

        Returns:
            DataFrame con resultados de todos los trials
        """
        # Crear o cargar study
        self.study = optuna.create_study(
            study_name=self.experiment_name,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            storage=self.storage,
            load_if_exists=True,
        )

        # Ejecutar optimización
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=show_progress,
        )

        # Convertir resultados a DataFrame
        self.results_df = self.study.trials_dataframe()

        return self.results_df

    def get_best_params(self) -> Dict[str, Any]:
        """
        Obtener mejores hiperparámetros encontrados.

        Returns:
            dict: Mejores hiperparámetros
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        return self.study.best_params

    def get_best_value(self) -> float:
        """
        Obtener mejor valor de métrica encontrado.

        Returns:
            float: Mejor valor de métrica
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        return self.study.best_value

    def get_best_trial(self) -> Trial:
        """
        Obtener mejor trial completo.

        Returns:
            Trial: Mejor trial
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        return self.study.best_trial

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analizar resultados de la optimización.

        Returns:
            dict: Análisis completo de resultados
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        analysis = {
            "best_params": self.get_best_params(),
            "best_value": self.get_best_value(),
            "n_trials": len(self.study.trials),
            "best_trial_number": self.study.best_trial.number,
            "optimization_history": [
                {"trial": t.number, "value": t.value}
                for t in self.study.trials
                if t.value is not None
            ],
        }

        # Importancia de hiperparámetros
        try:
            importance = optuna.importance.get_param_importances(self.study)
            analysis["param_importances"] = importance
        except Exception:
            analysis["param_importances"] = {}

        return analysis

    def save_results(self, save_dir: Path):
        """
        Guardar resultados de la optimización.

        Args:
            save_dir: Directorio donde guardar resultados
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Guardar mejores parámetros
        best_params_path = save_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(self.get_best_params(), f, indent=2)

        # Guardar análisis completo
        analysis_path = save_dir / "optimization_analysis.json"
        with open(analysis_path, "w") as f:
            analysis = self.analyze_results()
            # Convertir numpy types a Python types
            for key, value in analysis.items():
                if isinstance(value, np.integer):
                    analysis[key] = int(value)
                elif isinstance(value, np.floating):
                    analysis[key] = float(value)
            json.dump(analysis, f, indent=2)

        # Guardar DataFrame de resultados
        if self.results_df is not None:
            results_path = save_dir / "all_trials.csv"
            self.results_df.to_csv(results_path, index=False)

        print(f"Resultados guardados en: {save_dir}")

    def plot_optimization_history(self):
        """Graficar historial de optimización."""
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            return fig
        except Exception as e:
            print(f"Error al crear visualización: {e}")
            return None

    def plot_param_importances(self):
        """Graficar importancia de parámetros."""
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            return fig
        except Exception as e:
            print(f"Error al crear visualización: {e}")
            return None

    def plot_parallel_coordinate(self):
        """Graficar coordenadas paralelas de hiperparámetros."""
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")

        try:
            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            return fig
        except Exception as e:
            print(f"Error al crear visualización: {e}")
            return None
