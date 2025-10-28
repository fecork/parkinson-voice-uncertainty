"""
CNN Training Module
====================
Pipeline de entrenamiento con class weights, early stopping, y métricas.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time
import json
import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@contextlib.contextmanager
def disable_wandb_hooks():
    """
    Context manager para desactivar temporalmente los hooks de wandb.
    Útil cuando se evalúa un modelo fuera del contexto de wandb.
    """
    import wandb

    original_wandb_run = wandb.run
    wandb.run = None
    try:
        yield
    finally:
        wandb.run = original_wandb_run


# ============================================================
# EARLY STOPPING
# ============================================================


# EarlyStopping movida a modules.models.common.layers
from ..common.training_utils import (
    EarlyStopping,
    compute_metrics,
    compute_class_weights_auto,
)


# ============================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Entrena por una época.

    Args:
        model: Modelo PyTorch
        loader: DataLoader de entrenamiento
        optimizer: Optimizador
        criterion: Función de pérdida
        device: Device para cómputo

    Returns:
        Dict con métricas: loss, accuracy, precision, recall, f1
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        specs = batch["spectrogram"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(specs)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas
        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calcular métricas
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    """
    Evalúa el modelo.

    Args:
        model: Modelo PyTorch
        loader: DataLoader de evaluación
        criterion: Función de pérdida
        device: Device para cómputo

    Returns:
        Dict con métricas: loss, accuracy, precision, recall, f1
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        specs = batch["spectrogram"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(specs)
        loss = criterion(logits, labels)

        # Métricas
        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calcular métricas
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    return metrics


# ============================================================
# PIPELINE COMPLETO
# ============================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_epochs: int = 100,
    early_stopping_patience: int = 10,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    monitor_metric: str = "f1",  # NUEVO: métrica a monitorear ("loss" o "f1")
) -> Dict:
    """
    Pipeline completo de entrenamiento.

    Args:
        model: Modelo PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer: Optimizador
        criterion: Función de pérdida
        device: Device para cómputo
        n_epochs: Número máximo de épocas
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar checkpoints
        verbose: Si True, imprime progreso
        scheduler: Scheduler opcional para ajuste de learning rate
        monitor_metric: Métrica para early stopping ("loss" o "f1")
                       Default: "f1" (recomendado para datasets desbalanceados)

    Returns:
        Dict con historial de entrenamiento y mejor modelo
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    # Si monitoreamos F1, queremos maximizar; si monitoreamos loss, minimizar
    mode = "max" if monitor_metric == "f1" else "min"
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode=mode,
    )

    # Historial
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    # Inicializar best metric según lo que monitoreamos
    if monitor_metric == "f1":
        best_val_metric = -float("inf")  # Queremos maximizar F1
    else:
        best_val_metric = float("inf")  # Queremos minimizar loss

    best_model_state = None

    if verbose:
        print("\n" + "=" * 70)
        print("INICIO DE ENTRENAMIENTO")
        print("=" * 70)
        print(f"Épocas máximas: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Métrica monitoreada: val_{monitor_metric}")
        print(f"Device: {device}")
        print("=" * 70 + "\n")

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Entrenar
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validar
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Guardar historial
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        # Actualizar scheduler si está disponible
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        # Obtener métrica actual
        current_metric = val_metrics[monitor_metric]

        # Guardar mejor modelo según la métrica monitoreada
        is_better = False
        if monitor_metric == "f1":
            is_better = current_metric > best_val_metric
        else:
            is_better = current_metric < best_val_metric

        if is_better:
            best_val_metric = current_metric
            best_model_state = model.state_dict().copy()

            if save_dir is not None:
                checkpoint_path = save_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["f1"],
                        "val_metrics": val_metrics,
                        "monitor_metric": monitor_metric,
                        "best_val_metric": best_val_metric,
                    },
                    checkpoint_path,
                )

        epoch_time = time.time() - epoch_start

        # Imprimir progreso
        if verbose:
            print(
                f"Época {epoch + 1:3d}/{n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping usando la métrica correcta
        if early_stopping(current_metric, epoch):
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch + 1}")
                print(f"    Mejor época: {early_stopping.best_epoch + 1}")
                print(
                    f"    Mejor val_{monitor_metric}: {early_stopping.best_score:.4f}"
                )
            break

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print(f"Mejor val_{monitor_metric}: {best_val_metric:.4f}")
        print("=" * 70 + "\n")

    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Obtener best_val_loss para compatibilidad
    if monitor_metric == "loss":
        best_val_loss = best_val_metric
    else:
        best_f1_idx = history["val_f1"].index(max(history["val_f1"]))
        best_val_loss = history["val_loss"][best_f1_idx]

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "monitor_metric": monitor_metric,
        "total_time": total_time,
    }


# ============================================================
# EVALUACIÓN DETALLADA
# ============================================================


@torch.no_grad()
def detailed_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluación detallada con matriz de confusión y reporte.

    Args:
        model: Modelo PyTorch
        loader: DataLoader de test
        device: Device para cómputo
        class_names: Nombres de clases

    Returns:
        Dict con métricas detalladas
    """
    if class_names is None:
        class_names = ["HC", "PD"]

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # Desactivar wandb hooks temporalmente para evitar errores
    with disable_wandb_hooks():
        for batch in loader:
            specs = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(specs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Métricas
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
    }


def print_evaluation_report(eval_results: Dict, class_names: List[str] = None):
    """
    Imprime reporte de evaluación.

    Args:
        eval_results: Output de detailed_evaluation
        class_names: Nombres de clases
    """
    if class_names is None:
        class_names = ["HC", "PD"]

    print("\n" + "=" * 70)
    print("REPORTE DE EVALUACIÓN")
    print("=" * 70)

    print("\n📊 MATRIZ DE CONFUSIÓN:")
    cm = eval_results["confusion_matrix"]
    print(f"              Pred HC  Pred PD")
    print(f"Real HC       {cm[0, 0]:7d}  {cm[0, 1]:7d}")
    print(f"Real PD       {cm[1, 0]:7d}  {cm[1, 1]:7d}")

    print("\n📈 MÉTRICAS POR CLASE:")
    report = eval_results["classification_report"]
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {metrics['support']}")

    print("\n🎯 MÉTRICAS GLOBALES:")
    print(f"  Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"  F1 Macro:  {eval_results['f1_macro']:.4f}")

    print("=" * 70 + "\n")


# ============================================================
# GUARDADO/CARGA
# ============================================================


def save_training_results(results: Dict, save_dir: Path, prefix: str = "training"):
    """
    Guarda resultados de entrenamiento.

    Args:
        results: Dict con resultados
        save_dir: Directorio de guardado
        prefix: Prefijo para archivos
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Guardar historial como JSON
    history_path = save_dir / f"{prefix}_history.json"
    history = results["history"]

    # Convertir a listas serializables
    serializable_history = {
        k: [float(v) for v in values] for k, values in history.items()
    }

    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)

    print(f"💾 Historial guardado en: {history_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """
    Carga checkpoint.

    Args:
        checkpoint_path: Ruta al checkpoint
        model: Modelo PyTorch
        optimizer: Optimizador (opcional)

    Returns:
        Dict con información del checkpoint
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"✅ Checkpoint cargado desde: {checkpoint_path}")
    print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return checkpoint


# ============================================================
# DOMAIN ADAPTATION TRAINING
# ============================================================


def compute_lambda_schedule(
    epoch: int, max_epoch: int, gamma: float = 10.0, power: float = 0.75
) -> float:
    """
    Calcula el factor lambda para GRL según un scheduler progresivo.

    Formula (Ganin & Lempitsky 2015):
        lambda_p = 2 / (1 + exp(-gamma * p)) ** power - 1
        donde p = epoch / max_epoch

    Args:
        epoch: Época actual
        max_epoch: Número máximo de épocas
        gamma: Parámetro de progresión (default: 10.0)
        power: Exponente (default: 0.75)

    Returns:
        Valor de lambda entre 0 y ~1
    """
    p = epoch / max(max_epoch, 1)
    lambda_p = 2.0 / (1.0 + np.exp(-gamma * p)) ** power - 1.0
    return max(0.0, min(1.0, lambda_p))  # Clip entre [0, 1]


def train_one_epoch_da(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    alpha: float = 1.0,
    lambda_: float = 1.0,
) -> Dict[str, float]:
    """
    Entrena por una época con Domain Adaptation.

    Args:
        model: Modelo PyTorch con DA (debe retornar logits_pd, logits_domain)
        loader: DataLoader de entrenamiento (debe contener specs, labels_pd, labels_domain)
        optimizer: Optimizador
        criterion_pd: Función de pérdida para tarea PD
        criterion_domain: Función de pérdida para tarea de dominio
        device: Device para cómputo
        alpha: Peso de la pérdida de dominio (default: 1.0)
        lambda_: Factor para GRL (default: 1.0)

    Returns:
        Dict con métricas: loss_pd, loss_domain, loss_total, acc_pd, acc_domain
    """
    model.train()

    # Actualizar lambda del GRL
    if hasattr(model, "set_lambda"):
        model.set_lambda(lambda_)

    total_loss_pd = 0.0
    total_loss_domain = 0.0
    total_loss = 0.0
    all_preds_pd = []
    all_labels_pd = []
    all_preds_domain = []
    all_labels_domain = []

    for batch in loader:
        # Desempaquetar batch
        if len(batch) == 3:
            specs, labels_pd, labels_domain = batch
            specs = specs.to(device)
            labels_pd = labels_pd.to(device)
            labels_domain = labels_domain.to(device)
        else:
            # Compatibilidad con formato dict
            specs = batch["spectrogram"].to(device)
            labels_pd = batch["label"].to(device)
            labels_domain = batch.get("domain", torch.zeros_like(labels_pd)).to(device)

        # Forward pass
        logits_pd, logits_domain = model(specs)

        # Calcular pérdidas
        loss_pd = criterion_pd(logits_pd, labels_pd)
        loss_domain = criterion_domain(logits_domain, labels_domain)
        loss = loss_pd + alpha * loss_domain

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas
        batch_size = specs.size(0)
        total_loss_pd += loss_pd.item() * batch_size
        total_loss_domain += loss_domain.item() * batch_size
        total_loss += loss.item() * batch_size

        preds_pd = logits_pd.argmax(dim=1)
        preds_domain = logits_domain.argmax(dim=1)

        all_preds_pd.extend(preds_pd.cpu().numpy())
        all_labels_pd.extend(labels_pd.cpu().numpy())
        all_preds_domain.extend(preds_domain.cpu().numpy())
        all_labels_domain.extend(labels_domain.cpu().numpy())

    # Calcular métricas promedio
    n_samples = len(all_labels_pd)

    metrics = {
        "loss_pd": total_loss_pd / n_samples,
        "loss_domain": total_loss_domain / n_samples,
        "loss_total": total_loss / n_samples,
        "acc_pd": accuracy_score(all_labels_pd, all_preds_pd),
        "acc_domain": accuracy_score(all_labels_domain, all_preds_domain),
        "f1_pd": f1_score(
            all_labels_pd, all_preds_pd, average="macro", zero_division=0
        ),
    }

    return metrics


@torch.no_grad()
def evaluate_da(
    model: nn.Module,
    loader: DataLoader,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Evalúa el modelo con Domain Adaptation.

    Args:
        model: Modelo PyTorch con DA
        loader: DataLoader de evaluación
        criterion_pd: Función de pérdida para tarea PD
        criterion_domain: Función de pérdida para tarea de dominio
        device: Device para cómputo
        alpha: Peso de la pérdida de dominio

    Returns:
        Dict con métricas: loss_pd, loss_domain, loss_total, acc_pd, acc_domain
    """
    model.eval()

    total_loss_pd = 0.0
    total_loss_domain = 0.0
    total_loss = 0.0
    all_preds_pd = []
    all_labels_pd = []
    all_preds_domain = []
    all_labels_domain = []

    with disable_wandb_hooks():
        for batch in loader:
            # Desempaquetar batch
            if len(batch) == 3:
                specs, labels_pd, labels_domain = batch
                specs = specs.to(device)
                labels_pd = labels_pd.to(device)
                labels_domain = labels_domain.to(device)
            else:
                specs = batch["spectrogram"].to(device)
                labels_pd = batch["label"].to(device)
                labels_domain = batch.get("domain", torch.zeros_like(labels_pd)).to(
                    device
                )

            # Forward pass
            logits_pd, logits_domain = model(specs)

            # Calcular pérdidas
            loss_pd = criterion_pd(logits_pd, labels_pd)
            loss_domain = criterion_domain(logits_domain, labels_domain)
            loss = loss_pd + alpha * loss_domain

            # Métricas
            batch_size = specs.size(0)
            total_loss_pd += loss_pd.item() * batch_size
            total_loss_domain += loss_domain.item() * batch_size
            total_loss += loss.item() * batch_size

            preds_pd = logits_pd.argmax(dim=1)
            preds_domain = logits_domain.argmax(dim=1)

            all_preds_pd.extend(preds_pd.cpu().numpy())
            all_labels_pd.extend(labels_pd.cpu().numpy())
            all_preds_domain.extend(preds_domain.cpu().numpy())
            all_labels_domain.extend(labels_domain.cpu().numpy())

    # Calcular métricas promedio
    n_samples = len(all_labels_pd)

    metrics = {
        "loss_pd": total_loss_pd / n_samples,
        "loss_domain": total_loss_domain / n_samples,
        "loss_total": total_loss / n_samples,
        "acc_pd": accuracy_score(all_labels_pd, all_preds_pd),
        "acc_domain": accuracy_score(all_labels_domain, all_preds_domain),
        "f1_pd": f1_score(
            all_labels_pd, all_preds_pd, average="macro", zero_division=0
        ),
        "precision_pd": precision_score(all_labels_pd, all_preds_pd, zero_division=0),
        "recall_pd": recall_score(all_labels_pd, all_preds_pd, zero_division=0),
    }

    return metrics


def train_model_da(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    n_epochs: int = 100,
    alpha: float = 1.0,
    lambda_scheduler: Optional[callable] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: int = 15,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """
    Pipeline completo de entrenamiento con Domain Adaptation.

    Args:
        model: Modelo PyTorch con DA
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer: Optimizador (SGD recomendado según Ibarra 2023)
        criterion_pd: Función de pérdida para tarea PD
        criterion_domain: Función de pérdida para tarea de dominio
        device: Device para cómputo
        n_epochs: Número máximo de épocas
        alpha: Peso de la pérdida de dominio
        lambda_scheduler: Función para calcular lambda(epoch) -> float
                         Si None, usa lambda=1.0 constante (Ibarra 2023)
        lr_scheduler: Learning rate scheduler (StepLR recomendado)
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar checkpoints
        verbose: Si True, imprime progreso

    Returns:
        Dict con historial de entrenamiento y mejor modelo
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping basado en val_loss_pd (tarea principal)
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode="min",
    )

    # Historial
    history = {
        "train_loss_pd": [],
        "train_loss_domain": [],
        "train_loss_total": [],
        "train_acc_pd": [],
        "train_f1_pd": [],
        "val_loss_pd": [],
        "val_loss_domain": [],
        "val_loss_total": [],
        "val_acc_pd": [],
        "val_f1_pd": [],
        "lambda_values": [],
    }

    best_val_loss_pd = float("inf")
    best_model_state = None

    if verbose:
        print("\n" + "=" * 70)
        print("INICIO DE ENTRENAMIENTO CON DOMAIN ADAPTATION")
        print("=" * 70)
        print(f"Épocas máximas: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Alpha (peso dominio): {alpha}")
        print(f"Device: {device}")
        print("=" * 70 + "\n")

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Calcular lambda para esta época
        if lambda_scheduler is not None:
            lambda_ = lambda_scheduler(epoch)
        else:
            lambda_ = 1.0

        # Entrenar
        train_metrics = train_one_epoch_da(
            model,
            train_loader,
            optimizer,
            criterion_pd,
            criterion_domain,
            device,
            alpha=alpha,
            lambda_=lambda_,
        )

        # Validar
        val_metrics = evaluate_da(
            model, val_loader, criterion_pd, criterion_domain, device, alpha=alpha
        )

        # Guardar historial
        history["train_loss_pd"].append(train_metrics["loss_pd"])
        history["train_loss_domain"].append(train_metrics["loss_domain"])
        history["train_loss_total"].append(train_metrics["loss_total"])
        history["train_acc_pd"].append(train_metrics["acc_pd"])
        history["train_f1_pd"].append(train_metrics["f1_pd"])
        history["val_loss_pd"].append(val_metrics["loss_pd"])
        history["val_loss_domain"].append(val_metrics["loss_domain"])
        history["val_loss_total"].append(val_metrics["loss_total"])
        history["val_acc_pd"].append(val_metrics["acc_pd"])
        history["val_f1_pd"].append(val_metrics["f1_pd"])
        history["lambda_values"].append(lambda_)

        # Guardar mejor modelo
        if val_metrics["loss_pd"] < best_val_loss_pd:
            best_val_loss_pd = val_metrics["loss_pd"]
            best_model_state = model.state_dict().copy()

            if save_dir is not None:
                checkpoint_path = save_dir / "best_model_da.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss_pd": best_val_loss_pd,
                        "val_metrics": val_metrics,
                        "alpha": alpha,
                    },
                    checkpoint_path,
                )

        epoch_time = time.time() - epoch_start

        # Imprimir progreso
        if verbose:
            print(
                f"Época {epoch + 1:3d}/{n_epochs} | "
                f"λ={lambda_:.3f} | "
                f"Train: L_PD={train_metrics['loss_pd']:.4f} "
                f"L_Dom={train_metrics['loss_domain']:.4f} | "
                f"Val: L_PD={val_metrics['loss_pd']:.4f} "
                f"F1={val_metrics['f1_pd']:.4f} | "
                f"{epoch_time:.1f}s"
            )

        # Learning rate scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Early stopping
        if early_stopping(val_metrics["loss_pd"], epoch):
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch + 1}")
                print(f"    Mejor época: {early_stopping.best_epoch + 1}")
                print(f"    Mejor val_loss_pd: {early_stopping.best_score:.4f}")
            break

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print(f"Mejor val_loss_pd: {best_val_loss_pd:.4f}")
        print("=" * 70 + "\n")

    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "history": history,
        "best_val_loss_pd": best_val_loss_pd,
        "total_time": total_time,
    }


@torch.no_grad()
def evaluate_by_patient_da(
    model: nn.Module,
    test_data: tuple,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluación por paciente con agregación de probabilidades.

    Agrupa predicciones por subject_id y promedia probabilidades
    antes de hacer la predicción final (más realista que por segmento).

    Args:
        model: Modelo PyTorch con DA
        test_data: Tupla (X_test, y_test, metadata) donde metadata es lista de dicts
                   con 'subject_id'
        device: Device para cómputo
        class_names: Nombres de clases (default: ['HC', 'PD'])

    Returns:
        Dict con métricas a nivel paciente
    """
    if class_names is None:
        class_names = ["HC", "PD"]

    model.eval()

    X_test, y_test, metadata = test_data
    X_test = X_test.to(device)

    # Forward pass para obtener probabilidades
    with disable_wandb_hooks():
        logits_pd, _ = model(X_test)
        probs = torch.softmax(logits_pd, dim=1).cpu().numpy()

    # Agrupar por paciente
    patient_probs = {}
    patient_labels = {}

    for i, meta in enumerate(metadata):
        subject_id = meta.get("subject_id", f"unknown_{i}")
        label = y_test[i].item() if isinstance(y_test[i], torch.Tensor) else y_test[i]

        if subject_id not in patient_probs:
            patient_probs[subject_id] = []
            patient_labels[subject_id] = label

        patient_probs[subject_id].append(probs[i])

    # Agregar probabilidades por paciente
    patient_predictions = []
    patient_true_labels = []

    for subject_id in patient_probs.keys():
        # Promedio de probabilidades
        avg_probs = np.mean(patient_probs[subject_id], axis=0)
        pred = np.argmax(avg_probs)

        patient_predictions.append(pred)
        patient_true_labels.append(patient_labels[subject_id])

    # Calcular métricas
    patient_predictions = np.array(patient_predictions)
    patient_true_labels = np.array(patient_true_labels)

    cm = confusion_matrix(patient_true_labels, patient_predictions)
    report = classification_report(
        patient_true_labels,
        patient_predictions,
        target_names=class_names,
        output_dict=True,
    )

    return {
        "predictions": patient_predictions,
        "labels": patient_true_labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "n_patients": len(patient_predictions),
    }


# ============================================================
# 10-FOLD CROSS-VALIDATION TRAINING
# ============================================================


def train_model_da_kfold(
    model_class: type,
    model_params: Dict,
    dataset,
    metadata_list: List[dict],
    device: torch.device,
    n_folds: int = 10,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 0.1,
    alpha: float = 1.0,
    lambda_constant: float = 1.0,
    early_stopping_patience: int = 15,
    save_dir: Optional[Path] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Entrenamiento con 10-fold cross-validation según Ibarra (2023).

    Implementa:
    - 10-fold CV estratificada independiente por hablante
    - SGD con LR inicial 0.1 y scheduler StepLR
    - Cross-entropy ponderada automática para PD y dominio
    - Lambda constante para GRL (default: 1.0)

    Args:
        model_class: Clase del modelo (ej: CNN2D_DA)
        model_params: Parámetros para inicializar el modelo
        dataset: Dataset completo (Combined HC + PD)
        metadata_list: Lista de metadatos con 'subject_id' y 'label'
        device: Device para cómputo
        n_folds: Número de folds (default: 10)
        batch_size: Tamaño de batch (paper: probar 16/32/64)
        n_epochs: Épocas máximas por fold
        lr: Learning rate inicial (default: 0.1 según paper)
        alpha: Peso de pérdida de dominio
        lambda_constant: Valor constante de lambda para GRL (default: 1.0)
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar resultados
        seed: Semilla para reproducibilidad
        verbose: Si True, imprime progreso

    Returns:
        Dict con métricas agregadas de todos los folds
    """
    from torch.utils.data import DataLoader, Subset
    from .utils import (
        create_10fold_splits_by_speaker,
        compute_class_weights_auto,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Crear splits de 10-fold
    print("\n" + "=" * 70)
    print("PREPARANDO 10-FOLD CV ESTRATIFICADA POR HABLANTE")
    print("=" * 70)

    fold_splits = create_10fold_splits_by_speaker(
        metadata_list, n_folds=n_folds, seed=seed
    )

    # Almacenar resultados de cada fold
    all_fold_results = []
    fold_histories = []

    # Entrenar cada fold
    for fold_idx, split_indices in enumerate(fold_splits):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print("=" * 70)

        # Crear subsets
        train_subset = Subset(dataset, split_indices["train"])
        val_subset = Subset(dataset, split_indices["val"])

        # Crear DataLoaders
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"   Train: {len(train_subset)} muestras ({len(train_loader)} batches)")
        print(f"   Val:   {len(val_subset)} muestras ({len(val_loader)} batches)")

        # Extraer labels de train para calcular pesos
        train_labels_pd = []
        train_labels_domain = []

        for idx in split_indices["train"]:
            sample = dataset[idx]
            if len(sample) == 3:
                _, label_pd, label_domain = sample
            else:
                label_pd = sample[1]
                label_domain = torch.tensor(0)  # Default

            train_labels_pd.append(
                label_pd.item() if isinstance(label_pd, torch.Tensor) else label_pd
            )
            train_labels_domain.append(
                label_domain.item()
                if isinstance(label_domain, torch.Tensor)
                else label_domain
            )

        train_labels_pd = torch.tensor(train_labels_pd, dtype=torch.long)
        train_labels_domain = torch.tensor(train_labels_domain, dtype=torch.long)

        # Calcular pesos automáticos
        print("\n   📊 Calculando class weights automáticos:")
        print("      PD (tarea principal):")
        pd_weights = compute_class_weights_auto(train_labels_pd, threshold=0.4)

        print("      Dominio (tarea auxiliar):")
        domain_weights = compute_class_weights_auto(train_labels_domain, threshold=0.4)

        # Crear criterios
        if pd_weights is not None:
            criterion_pd = nn.CrossEntropyLoss(weight=pd_weights.to(device))
        else:
            criterion_pd = nn.CrossEntropyLoss()

        if domain_weights is not None:
            criterion_domain = nn.CrossEntropyLoss(weight=domain_weights.to(device))
        else:
            criterion_domain = nn.CrossEntropyLoss()

        # Crear modelo nuevo para este fold
        model = model_class(**model_params).to(device)

        # Crear optimizador SGD con LR 0.1 (Ibarra 2023)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

        # Crear scheduler StepLR
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )

        print(f"\n   ⚙️  Configuración:")
        print(f"      Optimizer: SGD (lr={lr}, momentum=0.9, wd=1e-4)")
        print(f"      LR Scheduler: StepLR (step=30, gamma=0.1)")
        print(f"      Lambda GRL: {lambda_constant} (constante)")
        print(f"      Alpha (dominio): {alpha}")

        # Lambda scheduler constante
        def lambda_scheduler(epoch):
            return lambda_constant

        # Guardar en subdirectorio del fold
        fold_save_dir = save_dir / f"fold_{fold_idx + 1}" if save_dir else None

        # Entrenar
        fold_results = train_model_da(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion_pd=criterion_pd,
            criterion_domain=criterion_domain,
            device=device,
            n_epochs=n_epochs,
            alpha=alpha,
            lambda_scheduler=lambda_scheduler,
            lr_scheduler=lr_scheduler,
            early_stopping_patience=early_stopping_patience,
            save_dir=fold_save_dir,
            verbose=verbose,
        )

        # Guardar resultados del fold
        all_fold_results.append(
            {
                "fold": fold_idx + 1,
                "best_val_loss_pd": fold_results["best_val_loss_pd"],
                "total_time": fold_results["total_time"],
                "n_epochs_trained": len(fold_results["history"]["train_loss_pd"]),
            }
        )

        fold_histories.append(fold_results["history"])

        print(f"\n   ✅ Fold {fold_idx + 1} completado:")
        print(f"      Best val_loss_pd: {fold_results['best_val_loss_pd']:.4f}")
        print(f"      Tiempo: {fold_results['total_time'] / 60:.1f} min")

    # Calcular estadísticas agregadas
    print("\n" + "=" * 70)
    print("RESUMEN 10-FOLD CROSS-VALIDATION")
    print("=" * 70)

    val_losses = [fold["best_val_loss_pd"] for fold in all_fold_results]
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)

    print(f"\n📊 Métricas agregadas (10 folds):")
    print(f"   Val Loss PD: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"\n   Por fold:")
    for fold in all_fold_results:
        print(
            f"      Fold {fold['fold']}: {fold['best_val_loss_pd']:.4f} "
            f"({fold['n_epochs_trained']} épocas)"
        )

    total_time = sum(fold["total_time"] for fold in all_fold_results)
    print(f"\n   Tiempo total: {total_time / 60:.1f} min")

    # Guardar resultados agregados
    if save_dir is not None:
        results_path = save_dir / "kfold_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "n_folds": n_folds,
                    "mean_val_loss_pd": float(mean_loss),
                    "std_val_loss_pd": float(std_loss),
                    "all_folds": all_fold_results,
                    "config": {
                        "batch_size": batch_size,
                        "lr_initial": lr,
                        "alpha": alpha,
                        "lambda": lambda_constant,
                        "n_epochs": n_epochs,
                        "early_stopping_patience": early_stopping_patience,
                    },
                },
                f,
                indent=2,
            )
        print(f"\n💾 Resultados guardados: {results_path}")

    print("=" * 70 + "\n")

    return {
        "fold_results": all_fold_results,
        "fold_histories": fold_histories,
        "mean_val_loss_pd": mean_loss,
        "std_val_loss_pd": std_loss,
        "total_time": total_time,
    }
