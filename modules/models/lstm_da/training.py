"""
Time-CNN-BiLSTM Training Module
================================
Pipeline de entrenamiento para Time-CNN-BiLSTM con Domain Adaptation.
Incluye 10-fold CV speaker-independent según paper Ibarra et al. 2023.
"""

from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Reutilizar EarlyStopping de CNN2D
from ..cnn2d.training import EarlyStopping


# ============================================================
# LAMBDA SCHEDULE
# ============================================================


def grl_lambda(epoch: int, max_epoch: int = 5) -> float:
    """
    Calcula lambda para GRL con warm-up.

    Args:
        epoch: Época actual
        max_epoch: Épocas para llegar a lambda=1.0

    Returns:
        Lambda value (0 a 1)
    """
    return min(1.0, epoch / max_epoch)


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
    current_epoch: int = 0,
    lambda_warmup_epochs: int = 5,
) -> Dict[str, float]:
    """
    Entrena una época con Domain Adaptation para Time-CNN-BiLSTM.

    Args:
        model: Modelo TimeCNNBiLSTM_DA
        loader: DataLoader de entrenamiento
        optimizer: Optimizer
        criterion_pd: Pérdida para tarea PD
        criterion_domain: Pérdida para tarea de dominio
        device: Device
        alpha: Peso de la pérdida de dominio
        current_epoch: Época actual (para lambda schedule)
        lambda_warmup_epochs: Épocas de warm-up para lambda

    Returns:
        Dict con métricas promedio de la época
    """
    model.train()

    # Actualizar lambda de GRL con warm-up
    lambda_grl = grl_lambda(current_epoch, lambda_warmup_epochs)
    model.set_lambda(lambda_grl)

    running_loss_pd = 0.0
    running_loss_domain = 0.0
    running_loss_total = 0.0

    all_preds_pd = []
    all_labels_pd = []
    all_preds_domain = []
    all_labels_domain = []

    total_samples = 0

    for batch in loader:
        # Obtener datos
        X = batch["X"].to(device)  # (B, T, 1, H, W)
        lengths = batch["length"]  # (B,)
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, dtype=torch.long)
        lengths = lengths.to(device)

        y_pd = batch["y_task"].to(device)  # (B,)
        y_domain = batch["y_domain"].to(device)  # (B,)

        batch_size = X.size(0)

        # Forward pass
        logits_pd, logits_domain, _ = model(X, lengths=lengths, return_embeddings=False)

        # Compute losses
        loss_pd = criterion_pd(logits_pd, y_pd)
        loss_domain = criterion_domain(logits_domain, y_domain)
        loss_total = loss_pd + alpha * loss_domain

        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Acumular losses
        running_loss_pd += loss_pd.item() * batch_size
        running_loss_domain += loss_domain.item() * batch_size
        running_loss_total += loss_total.item() * batch_size
        total_samples += batch_size

        # Predicciones
        preds_pd = logits_pd.argmax(dim=1)
        preds_domain = logits_domain.argmax(dim=1)

        all_preds_pd.extend(preds_pd.cpu().numpy())
        all_labels_pd.extend(y_pd.cpu().numpy())
        all_preds_domain.extend(preds_domain.cpu().numpy())
        all_labels_domain.extend(y_domain.cpu().numpy())

    # Calcular métricas promedio
    metrics = {
        "loss_pd": running_loss_pd / total_samples,
        "loss_domain": running_loss_domain / total_samples,
        "loss_total": running_loss_total / total_samples,
        "lambda_grl": lambda_grl,
        "accuracy_pd": accuracy_score(all_labels_pd, all_preds_pd),
        "f1_pd": f1_score(all_labels_pd, all_preds_pd, zero_division=0),
        "accuracy_domain": accuracy_score(all_labels_domain, all_preds_domain),
    }

    return metrics


def validate_epoch_da(
    model: nn.Module,
    loader: DataLoader,
    criterion_pd: nn.Module,
    criterion_domain: nn.Module,
    device: torch.device,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Valida una época con Domain Adaptation.

    Args:
        model: Modelo TimeCNNBiLSTM_DA
        loader: DataLoader de validación
        criterion_pd: Pérdida para tarea PD
        criterion_domain: Pérdida para tarea de dominio
        device: Device
        alpha: Peso de la pérdida de dominio

    Returns:
        Dict con métricas de validación
    """
    model.eval()

    running_loss_pd = 0.0
    running_loss_domain = 0.0
    running_loss_total = 0.0

    all_preds_pd = []
    all_labels_pd = []
    all_preds_domain = []
    all_labels_domain = []

    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Obtener datos
            X = batch["X"].to(device)
            lengths = batch["length"]
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, dtype=torch.long)
            lengths = lengths.to(device)

            y_pd = batch["y_task"].to(device)
            y_domain = batch["y_domain"].to(device)

            batch_size = X.size(0)

            # Forward pass
            logits_pd, logits_domain, _ = model(
                X, lengths=lengths, return_embeddings=False
            )

            # Compute losses
            loss_pd = criterion_pd(logits_pd, y_pd)
            loss_domain = criterion_domain(logits_domain, y_domain)
            loss_total = loss_pd + alpha * loss_domain

            # Acumular losses
            running_loss_pd += loss_pd.item() * batch_size
            running_loss_domain += loss_domain.item() * batch_size
            running_loss_total += loss_total.item() * batch_size
            total_samples += batch_size

            # Predicciones
            preds_pd = logits_pd.argmax(dim=1)
            preds_domain = logits_domain.argmax(dim=1)

            all_preds_pd.extend(preds_pd.cpu().numpy())
            all_labels_pd.extend(y_pd.cpu().numpy())
            all_preds_domain.extend(preds_domain.cpu().numpy())
            all_labels_domain.extend(y_domain.cpu().numpy())

    # Calcular métricas
    metrics = {
        "loss_pd": running_loss_pd / total_samples,
        "loss_domain": running_loss_domain / total_samples,
        "loss_total": running_loss_total / total_samples,
        "accuracy_pd": accuracy_score(all_labels_pd, all_preds_pd),
        "precision_pd": precision_score(all_labels_pd, all_preds_pd, zero_division=0),
        "recall_pd": recall_score(all_labels_pd, all_preds_pd, zero_division=0),
        "f1_pd": f1_score(all_labels_pd, all_preds_pd, zero_division=0),
        "accuracy_domain": accuracy_score(all_labels_domain, all_preds_domain),
    }

    return metrics


# ============================================================
# K-FOLD CROSS-VALIDATION
# ============================================================


def train_model_da_kfold(
    model_class,
    model_params: Dict,
    dataset,
    metadata_list: List[Dict],
    device: torch.device,
    n_folds: int = 10,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 0.1,
    alpha: float = 1.0,
    lambda_warmup_epochs: int = 5,
    early_stopping_patience: int = 15,
    save_dir: Path = Path("results/lstm_da_kfold"),
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Entrena modelo con 10-fold CV speaker-independent.

    Args:
        model_class: Clase del modelo (TimeCNNBiLSTM_DA)
        model_params: Parámetros del modelo
        dataset: Dataset completo (ConcatDataset)
        metadata_list: Lista de metadata con labels
        device: Device
        n_folds: Número de folds (default: 10)
        batch_size: Tamaño de batch
        n_epochs: Épocas máximas por fold
        lr: Learning rate inicial (SGD)
        alpha: Peso de pérdida de dominio
        lambda_warmup_epochs: Épocas de warm-up para lambda GRL
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar resultados
        seed: Semilla aleatoria
        verbose: Si imprimir progreso

    Returns:
        Dict con resultados agregados de K-fold CV
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extraer subject_ids y labels para stratified K-fold
    subject_ids = []
    labels = []
    subject_to_indices = defaultdict(list)

    for idx, meta in enumerate(metadata_list):
        subject_id = meta.get("subject_id", meta.get("filename", f"subj_{idx}"))
        label = meta["label"]

        subject_ids.append(subject_id)
        labels.append(label)
        subject_to_indices[subject_id].append(idx)

    # Obtener unique subjects con sus labels
    unique_subjects = list(subject_to_indices.keys())
    subject_labels = []
    for subj in unique_subjects:
        indices = subject_to_indices[subj]
        subj_label = labels[indices[0]]  # Asumir mismo label por sujeto
        subject_labels.append(subj_label)

    # Stratified K-Fold sobre subjects
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    start_time = time.time()

    for fold, (train_subjects_idx, val_subjects_idx) in enumerate(
        skf.split(unique_subjects, subject_labels), 1
    ):
        if verbose:
            print("\n" + "=" * 70)
            print(f"FOLD {fold}/{n_folds}")
            print("=" * 70)

        # Obtener subject names
        train_subjects = [unique_subjects[i] for i in train_subjects_idx]
        val_subjects = [unique_subjects[i] for i in val_subjects_idx]

        # Obtener índices de samples
        train_indices = []
        val_indices = []

        for subj in train_subjects:
            train_indices.extend(subject_to_indices[subj])
        for subj in val_subjects:
            val_indices.extend(subject_to_indices[subj])

        if verbose:
            print(
                f"Train: {len(train_subjects)} subjects → {len(train_indices)} samples"
            )
            print(f"Val:   {len(val_subjects)} subjects → {len(val_indices)} samples")

        # Crear subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Crear modelo
        model = model_class(**model_params).to(device)

        # Optimizer: SGD con momentum (según paper)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # LR Scheduler: StepLR (decay cada 30 épocas según paper)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )

        # Class weights para PD task
        train_labels_pd = [labels[i] for i in train_indices]
        class_counts = np.bincount(train_labels_pd)
        class_weights_pd = torch.tensor(
            [len(train_labels_pd) / (len(class_counts) * c) for c in class_counts],
            dtype=torch.float32,
        ).to(device)

        # Class weights para domain task (4 dominios fijos)
        train_labels_domain = [
            metadata_list[i].get("domain_label", 0) for i in train_indices
        ]
        domain_counts = np.bincount(
            train_labels_domain, minlength=4
        )  # Forzar 4 dominios
        class_weights_domain = torch.tensor(
            [
                len(train_labels_domain) / (4 * c) if c > 0 else 1.0
                for c in domain_counts
            ],
            dtype=torch.float32,
        ).to(device)

        # Verificar que tenemos exactamente 4 pesos
        if class_weights_domain.shape[0] != 4:
            class_weights_domain = torch.tensor(
                [1.0, 1.0, 1.0, 1.0], dtype=torch.float32
            ).to(device)

        # Loss functions
        criterion_pd = nn.CrossEntropyLoss(weight=class_weights_pd)
        criterion_domain = nn.CrossEntropyLoss(weight=class_weights_domain)

        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")

        # Training loop
        best_val_loss = float("inf")
        best_metrics = None
        history = {"train": [], "val": []}

        for epoch in range(n_epochs):
            # Train
            train_metrics = train_one_epoch_da(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion_pd=criterion_pd,
                criterion_domain=criterion_domain,
                device=device,
                alpha=alpha,
                current_epoch=epoch,
                lambda_warmup_epochs=lambda_warmup_epochs,
            )

            # Validate
            val_metrics = validate_epoch_da(
                model=model,
                loader=val_loader,
                criterion_pd=criterion_pd,
                criterion_domain=criterion_domain,
                device=device,
                alpha=alpha,
            )

            # Update scheduler
            scheduler.step()

            # Save history
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs} | "
                    f"Train Loss: {train_metrics['loss_pd']:.4f} | "
                    f"Val Loss: {val_metrics['loss_pd']:.4f} | "
                    f"Val F1: {val_metrics['f1_pd']:.4f} | "
                    f"λ: {train_metrics['lambda_grl']:.2f}"
                )

            # Early stopping
            if val_metrics["loss_pd"] < best_val_loss:
                best_val_loss = val_metrics["loss_pd"]
                best_metrics = val_metrics.copy()

                # Guardar mejor modelo
                fold_dir = save_dir / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    fold_dir / "best_model.pth",
                )

            if early_stopping(val_metrics["loss_pd"], epoch):
                if verbose:
                    print(f"Early stopping en época {epoch + 1}")
                break

        # Guardar resultados del fold
        fold_result = {
            "fold": fold,
            "best_val_loss_pd": best_val_loss,
            "best_metrics": best_metrics,
            "history": history,
        }
        fold_results.append(fold_result)

        # Guardar métricas del fold
        fold_dir = save_dir / f"fold_{fold}"
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=2)

        if verbose:
            print(f"\nFold {fold} completado:")
            print(f"  Val Loss PD: {best_val_loss:.4f}")
            print(f"  Val Acc PD:  {best_metrics['accuracy_pd']:.4f}")
            print(f"  Val F1 PD:   {best_metrics['f1_pd']:.4f}")

    total_time = time.time() - start_time

    # Agregar resultados de todos los folds
    all_val_losses = [r["best_val_loss_pd"] for r in fold_results]
    all_val_accs = [r["best_metrics"]["accuracy_pd"] for r in fold_results]
    all_val_f1s = [r["best_metrics"]["f1_pd"] for r in fold_results]

    aggregated_results = {
        "n_folds": n_folds,
        "mean_val_loss_pd": np.mean(all_val_losses),
        "std_val_loss_pd": np.std(all_val_losses),
        "mean_val_acc_pd": np.mean(all_val_accs),
        "std_val_acc_pd": np.std(all_val_accs),
        "mean_val_f1_pd": np.mean(all_val_f1s),
        "std_val_f1_pd": np.std(all_val_f1s),
        "total_time": total_time,
        "fold_results": fold_results,
    }

    # Guardar resultados agregados
    with open(save_dir / "kfold_results.json", "w") as f:
        # Remover history para no sobrecargar JSON
        results_to_save = aggregated_results.copy()
        results_to_save["fold_results"] = [
            {k: v for k, v in r.items() if k != "history"} for r in fold_results
        ]
        json.dump(results_to_save, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print(f"K-FOLD CV COMPLETADO ({n_folds} folds)")
        print("=" * 70)
        print(
            f"Val Loss PD:  {aggregated_results['mean_val_loss_pd']:.4f} ± {aggregated_results['std_val_loss_pd']:.4f}"
        )
        print(
            f"Val Acc PD:   {aggregated_results['mean_val_acc_pd']:.4f} ± {aggregated_results['std_val_acc_pd']:.4f}"
        )
        print(
            f"Val F1 PD:    {aggregated_results['mean_val_f1_pd']:.4f} ± {aggregated_results['std_val_f1_pd']:.4f}"
        )
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print("=" * 70)

    return aggregated_results
