"""
CNN 1D Training Module
=======================
Pipeline de entrenamiento para CNN1D con Domain Adaptation.
Incluye agregación por paciente según paper Ibarra et al. 2023.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# EARLY STOPPING (reutilizado)
# ============================================================


class EarlyStopping:
    """Early stopping para detener entrenamiento cuando no mejora."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"
    ):
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
# TRAINING FUNCTIONS
# ============================================================


def train_one_epoch_da(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Entrena una época con Domain Adaptation para CNN1D.

    Args:
        model: Modelo CNN1D_DA
        loader: DataLoader de entrenamiento
        optimizer: Optimizer
        criterion_pd: Pérdida para tarea PD
        criterion_domain: Pérdida para tarea de dominio
        device: Device
        alpha: Peso de la pérdida de dominio

    Returns:
        Dict con métricas promedio de la época
    """
    model.train()

    running_loss_pd = 0.0
    running_loss_domain = 0.0
    running_loss_total = 0.0
    n_batches = 0

    for batch in loader:
        specs, labels_pd, labels_domain = batch
        specs = specs.to(device)
        labels_pd = labels_pd.to(device)
        labels_domain = labels_domain.to(device)

        # Forward pass
        logits_pd, logits_domain, _ = model(specs)

        # Losses
        loss_pd = criterion_pd(logits_pd, labels_pd)
        loss_domain = criterion_domain(logits_domain, labels_domain)
        loss_total = loss_pd + alpha * loss_domain

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Acumular
        running_loss_pd += loss_pd.item()
        running_loss_domain += loss_domain.item()
        running_loss_total += loss_total.item()
        n_batches += 1

    # Promedios
    metrics = {
        "loss_pd": running_loss_pd / n_batches,
        "loss_domain": running_loss_domain / n_batches,
        "loss_total": running_loss_total / n_batches,
    }

    return metrics


def evaluate_da(
    model: nn.Module,
    loader: DataLoader,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    alpha: float = 1.0,
    return_embeddings: bool = False,
) -> Dict[str, float]:
    """
    Evalúa modelo DA en dataset de validación/test.

    Args:
        model: Modelo CNN1D_DA
        loader: DataLoader
        criterion_pd: Pérdida PD
        criterion_domain: Pérdida dominio
        device: Device
        alpha: Peso de pérdida de dominio
        return_embeddings: Si retornar embeddings para t-SNE

    Returns:
        Dict con métricas + opcionalmente arrays de probs, embeddings, labels
    """
    model.eval()

    running_loss_pd = 0.0
    running_loss_domain = 0.0
    running_loss_total = 0.0
    n_batches = 0

    all_preds_pd = []
    all_labels_pd = []
    all_preds_domain = []
    all_labels_domain = []
    all_probs_pd = []
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            specs, labels_pd, labels_domain = batch
            specs = specs.to(device)
            labels_pd = labels_pd.to(device)
            labels_domain = labels_domain.to(device)

            # Forward
            logits_pd, logits_domain, embeddings = model(
                specs, return_embeddings=return_embeddings
            )

            # Losses
            loss_pd = criterion_pd(logits_pd, labels_pd)
            loss_domain = criterion_domain(logits_domain, labels_domain)
            loss_total = loss_pd + alpha * loss_domain

            # Predicciones
            preds_pd = logits_pd.argmax(dim=1)
            preds_domain = logits_domain.argmax(dim=1)
            probs_pd = F.softmax(logits_pd, dim=1)

            # Acumular
            running_loss_pd += loss_pd.item()
            running_loss_domain += loss_domain.item()
            running_loss_total += loss_total.item()
            n_batches += 1

            all_preds_pd.extend(preds_pd.cpu().numpy())
            all_labels_pd.extend(labels_pd.cpu().numpy())
            all_preds_domain.extend(preds_domain.cpu().numpy())
            all_labels_domain.extend(labels_domain.cpu().numpy())
            all_probs_pd.extend(probs_pd.cpu().numpy())

            if return_embeddings and embeddings is not None:
                all_embeddings.extend(embeddings.cpu().numpy())

    # Convertir a arrays
    all_preds_pd = np.array(all_preds_pd)
    all_labels_pd = np.array(all_labels_pd)
    all_preds_domain = np.array(all_preds_domain)
    all_labels_domain = np.array(all_labels_domain)
    all_probs_pd = np.array(all_probs_pd)

    # Métricas PD
    acc_pd = accuracy_score(all_labels_pd, all_preds_pd)
    precision_pd = precision_score(
        all_labels_pd, all_preds_pd, average="binary", zero_division=0
    )
    recall_pd = recall_score(
        all_labels_pd, all_preds_pd, average="binary", zero_division=0
    )
    f1_pd = f1_score(
        all_labels_pd, all_preds_pd, average="binary", zero_division=0
    )

    # Métricas Domain
    acc_domain = accuracy_score(all_labels_domain, all_preds_domain)

    metrics = {
        "loss_pd": running_loss_pd / n_batches,
        "loss_domain": running_loss_domain / n_batches,
        "loss_total": running_loss_total / n_batches,
        "acc_pd": acc_pd,
        "precision_pd": precision_pd,
        "recall_pd": recall_pd,
        "f1_pd": f1_pd,
        "acc_domain": acc_domain,
    }

    if return_embeddings:
        metrics["probs_pd"] = all_probs_pd
        metrics["embeddings"] = np.array(all_embeddings)
        metrics["labels_pd"] = all_labels_pd
        metrics["labels_domain"] = all_labels_domain

    return metrics


# ============================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================


def aggregate_patient_predictions(
    probs_array: np.ndarray, patient_ids: List[str], method: str = "mean"
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Agrupa predicciones por paciente según paper.

    Combina probabilidades de todos los segmentos de cada paciente
    para obtener predicción final patient-level.

    Args:
        probs_array: [N_segments, n_classes] probabilidades
        patient_ids: [N_segments] IDs de paciente
        method: 'mean' (promedio) o 'log_mean' (suma log-median)

    Returns:
        patient_probs: dict {patient_id: array[n_classes]}
        patient_labels: dict {patient_id: predicted_class}
    """
    patient_probs_dict = defaultdict(list)

    # Agrupar probs por paciente
    for prob, pid in zip(probs_array, patient_ids):
        patient_probs_dict[pid].append(prob)

    # Agregar según método
    patient_probs = {}
    if method == "mean":
        patient_probs = {
            pid: np.mean(p_list, axis=0)
            for pid, p_list in patient_probs_dict.items()
        }
    elif method == "log_mean":
        # Log-mean (equivalente a producto de probs)
        patient_probs = {
            pid: np.exp(np.mean(np.log(np.array(p_list) + 1e-10), axis=0))
            for pid, p_list in patient_probs_dict.items()
        }
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Predicción final por paciente (argmax)
    patient_labels = {
        pid: int(np.argmax(probs)) for pid, probs in patient_probs.items()
    }

    return patient_probs, patient_labels


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


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
    lambda_scheduler: Optional[Callable] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: int = 15,
    save_dir: str = "results/cnn1d_da",
    verbose: bool = True,
) -> Dict:
    """
    Entrena modelo CNN1D_DA con Domain Adaptation.

    Args:
        model: Modelo CNN1D_DA
        train_loader: DataLoader de train
        val_loader: DataLoader de validación
        optimizer: Optimizer
        criterion_pd: Loss para PD
        criterion_domain: Loss para dominio
        device: Device
        n_epochs: Épocas máximas
        alpha: Peso de loss_domain
        lambda_scheduler: Scheduler para lambda de GRL (epoch → lambda)
        lr_scheduler: LR scheduler
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar modelo
        verbose: Imprimir progreso

    Returns:
        Dict con modelo entrenado, history y métricas
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stop = EarlyStopping(patience=early_stopping_patience, mode="min")

    # History
    history = {
        "train_loss_pd": [],
        "train_loss_domain": [],
        "train_loss_total": [],
        "val_loss_pd": [],
        "val_loss_domain": [],
        "val_loss_total": [],
        "val_f1_pd": [],
        "lambda": [],
    }

    best_val_loss_pd = float("inf")
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("INICIO DE ENTRENAMIENTO CNN1D_DA")
        print("=" * 70)
        print(f"Épocas máximas: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Alpha (peso dominio): {alpha}")
        print(f"Device: {device}")
        print("=" * 70 + "\n")

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Actualizar lambda de GRL
        if lambda_scheduler is not None:
            current_lambda = lambda_scheduler(epoch)
            model.set_lambda(current_lambda)
        else:
            current_lambda = 1.0

        # Training
        train_metrics = train_one_epoch_da(
            model,
            train_loader,
            optimizer,
            criterion_pd,
            criterion_domain,
            device,
            alpha,
        )

        # Validation
        val_metrics = evaluate_da(
            model, val_loader, criterion_pd, criterion_domain, device, alpha
        )

        # Actualizar LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Guardar history
        history["train_loss_pd"].append(train_metrics["loss_pd"])
        history["train_loss_domain"].append(train_metrics["loss_domain"])
        history["train_loss_total"].append(train_metrics["loss_total"])
        history["val_loss_pd"].append(val_metrics["loss_pd"])
        history["val_loss_domain"].append(val_metrics["loss_domain"])
        history["val_loss_total"].append(val_metrics["loss_total"])
        history["val_f1_pd"].append(val_metrics["f1_pd"])
        history["lambda"].append(current_lambda)

        epoch_time = time.time() - epoch_start

        if verbose:
            print(
                f"Época {epoch+1:3d}/{n_epochs} | λ={current_lambda:.3f} | "
                f"Train: L_PD={train_metrics['loss_pd']:.4f} "
                f"L_Dom={train_metrics['loss_domain']:.4f} | "
                f"Val: L_PD={val_metrics['loss_pd']:.4f} "
                f"F1={val_metrics['f1_pd']:.4f} | "
                f"{epoch_time:.1f}s"
            )

        # Guardar mejor modelo
        if val_metrics["loss_pd"] < best_val_loss_pd:
            best_val_loss_pd = val_metrics["loss_pd"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss_pd": best_val_loss_pd,
                    "history": history,
                },
                save_path / "best_model_1d_da.pth",
            )

        # Early stopping
        if early_stop(val_metrics["loss_pd"], epoch):
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch+1}")
                print(f"    Mejor época: {early_stop.best_epoch+1}")
                print(f"    Mejor val_loss_pd: {best_val_loss_pd:.4f}\n")
            break

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time/60:.1f} minutos")
        print(f"Mejor val_loss_pd: {best_val_loss_pd:.4f}")
        print("=" * 70 + "\n")

    # Cargar mejor modelo
    checkpoint = torch.load(save_path / "best_model_1d_da.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "model": model,
        "history": history,
        "best_val_loss_pd": best_val_loss_pd,
        "total_time": total_time,
    }


# ============================================================
# PATIENT-LEVEL EVALUATION
# ============================================================


def evaluate_patient_level(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    patient_ids: List[str],
    method: str = "mean",
) -> Dict[str, float]:
    """
    Evalúa métricas a nivel de paciente (agregando segmentos).

    Args:
        all_probs: [N_segments, n_classes] probabilidades
        all_labels: [N_segments] labels verdaderos
        patient_ids: [N_segments] IDs de paciente
        method: Método de agregación ('mean' o 'log_mean')

    Returns:
        Dict con métricas patient-level y confusion matrix
    """
    # Agregar predicciones por paciente
    patient_probs, patient_preds = aggregate_patient_predictions(
        all_probs, patient_ids, method=method
    )

    # Obtener true labels por paciente
    # (tomar label del primer segmento)
    patient_true_labels = {}
    for label, pid in zip(all_labels, patient_ids):
        if pid not in patient_true_labels:
            patient_true_labels[pid] = int(label)

    # Ordenar para comparación
    sorted_patients = sorted(patient_true_labels.keys())
    y_true = [patient_true_labels[pid] for pid in sorted_patients]
    y_pred = [patient_preds[pid] for pid in sorted_patients]

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_patients": len(sorted_patients),
    }


# ============================================================
# UTILIDADES
# ============================================================


def save_metrics(metrics: Dict, save_path: Path):
    """
    Guarda métricas en JSON.

    Args:
        metrics: Dict con métricas
        save_path: Ruta de guardado
    """
    # Convertir arrays numpy a listas
    metrics_json = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            metrics_json[key] = val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            metrics_json[key] = float(val)
        else:
            metrics_json[key] = val

    with open(save_path, "w") as f:
        json.dump(metrics_json, f, indent=2)


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: CNN1D_DA Training Functions")
    print("=" * 70)

    # Test patient aggregation
    print("\nTest 1: Patient Aggregation")
    print("-" * 70)

    # Simular 10 segmentos de 3 pacientes
    probs = np.random.rand(10, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)  # Normalizar
    patient_ids = ["P1", "P1", "P1", "P2", "P2", "P2", "P2", "P3", "P3", "P3"]

    patient_probs, patient_labels = aggregate_patient_predictions(
        probs, patient_ids, method="mean"
    )

    print(f"Segmentos: {len(probs)}")
    print(f"Pacientes únicos: {len(patient_probs)}")
    print(f"Patient probs keys: {list(patient_probs.keys())}")
    print(f"Patient labels: {patient_labels}")

    print("\n[OK] Tests básicos pasaron correctamente")
