"""
CNN2D Optuna Wrapper
====================
Wrapper específico para CNN2D compatible con el sistema de optimización Optuna.

Este módulo adapta el modelo CNN2D para trabajar con Optuna,
definiendo el espacio de búsqueda de hiperparámetros y la lógica de entrenamiento.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import optuna
from optuna.trial import Trial
from optuna.exceptions import TrialPruned

from modules.core.optuna_optimization import OptunaModelWrapper, OptunaOptimizer
from modules.models.cnn2d.model import CNN2D
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class CNN2DOptunaWrapper(OptunaModelWrapper):
    """
    Wrapper de CNN2D para optimización con Optuna.

    Define el espacio de búsqueda de hiperparámetros y la lógica de
    entrenamiento específica para CNN2D.
    """

    def __init__(self, input_shape: Tuple[int, int, int], device: str = "cpu"):
        """
        Args:
            input_shape: Shape de entrada (C, H, W)
            device: Dispositivo ('cpu' o 'cuda')
        """
        self.input_shape = input_shape
        self.device = device

    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda de hiperparámetros para CNN2D.

        Args:
            trial: Trial de Optuna

        Returns:
            dict: Hiperparámetros sugeridos
        """
        params = {
            # Arquitectura (nombres compatibles con CNN2D)
            "filters_1": trial.suggest_categorical("filters_1", [16, 32, 64]),
            "filters_2": trial.suggest_categorical("filters_2", [32, 64, 128]),
            "kernel_size_1": trial.suggest_categorical("kernel_size_1", [3, 5]),
            "kernel_size_2": trial.suggest_categorical("kernel_size_2", [3, 5]),
            "p_drop_conv": trial.suggest_float("p_drop_conv", 0.2, 0.5),
            "p_drop_fc": trial.suggest_float("p_drop_fc", 0.3, 0.6),
            "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
            # Optimización
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            # Batch size ya se sugiere en OptunaOptimizer._objective
        }

        return params

    def create_model(self, trial: Trial) -> nn.Module:
        """
        Crear modelo CNN2D con hiperparámetros sugeridos por Optuna.

        Args:
            trial: Trial de Optuna

        Returns:
            nn.Module: Modelo CNN2D
        """
        params = self.suggest_hyperparameters(trial)

        model = CNN2D(
            input_shape=self.input_shape[1:],  # (H, W) sin el canal
            filters_1=params["filters_1"],
            filters_2=params["filters_2"],
            kernel_size_1=params["kernel_size_1"],
            kernel_size_2=params["kernel_size_2"],
            p_drop_conv=params["p_drop_conv"],
            p_drop_fc=params["p_drop_fc"],
            dense_units=params["dense_units"],
        )

        return model.to(self.device)

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: Trial,
        n_epochs: int = 20,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Entrenar modelo CNN2D y retornar métricas.

        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            trial: Trial de Optuna (para reportar métricas intermedias)
            n_epochs: Número de épocas

        Returns:
            tuple: (f1_score_final, dict_metricas)
        """
        # Obtener hiperparámetros del trial
        params = trial.params

        # Configurar optimizer
        if params["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                momentum=0.9,
            )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_f1 = 0.0
        best_metrics = {}

        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)

                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())

            # Calcular métricas
            f1 = f1_score(val_labels, val_preds, average="macro")
            acc = accuracy_score(val_labels, val_preds)
            prec = precision_score(val_labels, val_preds, average="macro")
            rec = recall_score(val_labels, val_preds, average="macro")

            # Actualizar mejor F1
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "f1_macro": f1,
                    "accuracy": acc,
                    "precision_macro": prec,
                    "recall_macro": rec,
                }

            # Reportar métrica intermedia a Optuna (para pruning)
            trial.report(f1, epoch)

            # Verificar si el trial debe ser podado
            if trial.should_prune():
                raise TrialPruned()

        return best_f1, best_metrics


def create_cnn2d_optimizer(
    input_shape: Tuple[int, int, int],
    experiment_name: str = "cnn2d_optuna",
    n_trials: int = 50,
    n_epochs_per_trial: int = 20,
    device: str = "cpu",
) -> OptunaOptimizer:
    """
    Crear optimizador Optuna para CNN2D.

    Función de conveniencia para crear un optimizador configurado
    específicamente para CNN2D.

    Args:
        input_shape: Shape de entrada (C, H, W)
        experiment_name: Nombre del experimento
        n_trials: Número de trials a ejecutar
        n_epochs_per_trial: Épocas por trial
        device: Dispositivo ('cpu' o 'cuda')

    Returns:
        OptunaOptimizer: Optimizador configurado

    Example:
        >>> optimizer = create_cnn2d_optimizer(
        ...     input_shape=(1, 65, 41),
        ...     n_trials=30,
        ...     device='cuda'
        ... )
        >>> results = optimizer.optimize(X_train, y_train, X_val, y_val)
    """
    wrapper = CNN2DOptunaWrapper(input_shape=input_shape, device=device)

    optimizer = OptunaOptimizer(
        model_wrapper=wrapper,
        experiment_name=experiment_name,
        n_trials=n_trials,
        n_epochs_per_trial=n_epochs_per_trial,
        metric="f1",
        direction="maximize",
    )

    return optimizer


def optimize_cnn2d(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_shape: Tuple[int, int, int],
    n_trials: int = 50,
    n_epochs_per_trial: int = 20,
    device: str = "cpu",
    save_dir: str = None,
    checkpoint_dir: str = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Optimizar hiperparámetros de CNN2D con Optuna (función todo-en-uno).

    Args:
        X_train: Tensor de entrenamiento
        y_train: Labels de entrenamiento
        X_val: Tensor de validación
        y_val: Labels de validación
        input_shape: Shape de entrada (C, H, W)
        n_trials: Número de trials
        n_epochs_per_trial: Épocas por trial
        device: Dispositivo
        save_dir: Directorio donde guardar resultados (opcional)
        checkpoint_dir: Directorio para checkpoints (opcional)
        resume: Si reanudar desde checkpoint (opcional)

    Returns:
        dict: Resultados completos de la optimización

    Example:
        >>> results = optimize_cnn2d(
        ...     X_train, y_train, X_val, y_val,
        ...     input_shape=(1, 65, 41),
        ...     n_trials=30,
        ...     device='cuda',
        ...     save_dir='results/optuna',
        ...     checkpoint_dir='checkpoints',
        ...     resume=True
        ... )
        >>> print(f"Mejor F1: {results['best_value']}")
        >>> print(f"Mejores params: {results['best_params']}")
    """
    # Importar sistema de checkpointing
    from modules.core.optuna_checkpoint import OptunaCheckpoint

    # Configurar checkpointing
    if checkpoint_dir is None:
        checkpoint_dir = save_dir or "checkpoints"

    checkpoint = OptunaCheckpoint(
        checkpoint_dir=checkpoint_dir, experiment_name="cnn2d_optuna"
    )

    # Verificar si se puede reanudar
    if resume:
        resume_info = checkpoint.get_resume_info()
        if resume_info["can_resume"]:
            print(f"🔄 Reanudando optimización desde checkpoint:")
            print(f"   - Trials completados: {resume_info['completed_trials']}")
            print(f"   - Progreso: {resume_info['progress_percentage']:.1f}%")
            print(f"   - Mejor F1: {resume_info['best_value']:.4f}")
            print(f"   - Mejor trial: {resume_info['best_trial']}")

            # Crear estudio desde checkpoint
            study = checkpoint.create_study_from_checkpoint()

            # Calcular trials restantes
            remaining_trials = n_trials - resume_info["completed_trials"]
            if remaining_trials <= 0:
                print("✅ Optimización ya completada")
                results_df = checkpoint.create_dataframe_from_checkpoint()
                best_params = resume_info["best_params"]
                best_value = resume_info["best_value"]
            else:
                print(f"🚀 Continuando con {remaining_trials} trials restantes...")

                # Continuar optimización
                study.optimize(
                    lambda trial: _objective_with_checkpoint(
                        trial,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        input_shape,
                        n_epochs_per_trial,
                        device,
                        checkpoint,
                    ),
                    n_trials=remaining_trials,
                    show_progress_bar=True,
                )

                # Obtener resultados
                results_df = checkpoint.create_dataframe_from_checkpoint()
                best_params = study.best_params
                best_value = study.best_value
        else:
            print("🆕 Iniciando optimización desde cero...")
            study = _run_optimization_with_checkpoint(
                X_train,
                y_train,
                X_val,
                y_val,
                input_shape,
                n_trials,
                n_epochs_per_trial,
                device,
                checkpoint,
            )
            results_df = checkpoint.create_dataframe_from_checkpoint()
            best_params = study.best_params
            best_value = study.best_value
    else:
        print("🆕 Iniciando optimización desde cero...")
        study = _run_optimization_with_checkpoint(
            X_train,
            y_train,
            X_val,
            y_val,
            input_shape,
            n_trials,
            n_epochs_per_trial,
            device,
            checkpoint,
        )
        results_df = checkpoint.create_dataframe_from_checkpoint()
        best_params = study.best_params
        best_value = study.best_value

    # Preparar resultados
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "best_trial": study.best_trial.number,
        "results_df": results_df,
        "analysis": {"best_trial": study.best_trial},
    }

    # Guardar si se especificó directorio
    if save_dir:
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path / "optuna_trials_results.csv", index=False)
        with open(save_path / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

    print(f"\n🎉 Optimización completada!")
    print(f"Mejor F1: {best_value:.4f}")
    print(f"Mejores hiperparámetros:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return results


def _objective_with_checkpoint(
    trial: optuna.trial.Trial,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_shape: Tuple[int, int, int],
    n_epochs_per_trial: int,
    device: str,
    checkpoint: "OptunaCheckpoint",
) -> float:
    """
    Función objetivo con checkpointing automático.
    """
    # Crear modelo
    model = CNN2D(
        input_shape=input_shape[1:],
        filters_1=trial.suggest_categorical("filters_1", [16, 32, 64]),
        filters_2=trial.suggest_categorical("filters_2", [32, 64, 128]),
        kernel_size_1=trial.suggest_categorical("kernel_size_1", [3, 5]),
        kernel_size_2=trial.suggest_categorical("kernel_size_2", [3, 5]),
        p_drop_conv=trial.suggest_float("p_drop_conv", 0.2, 0.5),
        p_drop_fc=trial.suggest_float("p_drop_fc", 0.3, 0.6),
        dense_units=trial.suggest_categorical("dense_units", [32, 64, 128]),
    ).to(device)

    # Crear DataLoaders
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        shuffle=False,
    )

    # Optimizador
    if trial.suggest_categorical("optimizer", ["adam", "sgd"]) == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            momentum=0.9,
        )

    criterion = torch.nn.CrossEntropyLoss()

    # Entrenamiento
    best_f1 = 0.0
    best_metrics = {}

    for epoch in range(n_epochs_per_trial):
        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        # Calcular métricas
        from sklearn.metrics import (
            f1_score,
            accuracy_score,
            precision_score,
            recall_score,
        )

        f1 = f1_score(val_labels, val_preds, average="macro")
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, average="macro")
        rec = recall_score(val_labels, val_preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "f1_macro": f1,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
            }

        # Reportar a Optuna
        trial.report(f1, epoch)
        if trial.should_prune():
            raise TrialPruned()

    # Guardar trial en checkpoint
    checkpoint.save_trial(trial, best_metrics)

    return best_f1


def _run_optimization_with_checkpoint(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_shape: Tuple[int, int, int],
    n_trials: int,
    n_epochs_per_trial: int,
    device: str,
    checkpoint: "OptunaCheckpoint",
) -> optuna.Study:
    """
    Ejecutar optimización con checkpointing automático.
    """
    # Crear estudio
    study = optuna.create_study(
        study_name="cnn2d_optuna",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Ejecutar optimización
    study.optimize(
        lambda trial: _objective_with_checkpoint(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            input_shape,
            n_epochs_per_trial,
            device,
            checkpoint,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study
