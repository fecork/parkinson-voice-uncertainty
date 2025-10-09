"""
CNN Training Module
====================
Pipeline de entrenamiento con class weights, early stopping, y métricas.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time
import json

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


# ============================================================
# EARLY STOPPING
# ============================================================


class EarlyStopping:
    """
    Early stopping para detener entrenamiento cuando no mejora.
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
        "f1": f1_score(all_labels, all_preds, zero_division=0),
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
        "f1": f1_score(all_labels, all_preds, zero_division=0),
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

    Returns:
        Dict con historial de entrenamiento y mejor modelo
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode="min",  # Minimizar val_loss
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

    best_val_loss = float("inf")
    best_model_state = None

    if verbose:
        print("\n" + "=" * 70)
        print("INICIO DE ENTRENAMIENTO")
        print("=" * 70)
        print(f"Épocas máximas: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
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

        # Guardar mejor modelo
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = model.state_dict().copy()

            if save_dir is not None:
                checkpoint_path = save_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_metrics": val_metrics,
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

        # Early stopping
        if early_stopping(val_metrics["loss"], epoch):
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch + 1}")
                print(f"    Mejor época: {early_stopping.best_epoch + 1}")
                print(f"    Mejor val_loss: {early_stopping.best_score:.4f}")
            break

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print(f"Mejor val_loss: {best_val_loss:.4f}")
        print("=" * 70 + "\n")

    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
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
