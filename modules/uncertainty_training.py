"""
Funciones de entrenamiento y evaluación para modelos con incertidumbre.
"""

import time
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

from .uncertainty_loss import (
    heteroscedastic_classification_loss,
    compute_nll,
    compute_brier_score,
    compute_ece,
)


def train_uncertainty_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    n_epochs=60,
    n_noise_samples=5,
    early_stopping_patience=15,
    save_dir=None,
    verbose=True,
):
    """
    Entrena modelo con pérdida heteroscedástica.

    Args:
        model: Modelo UncertaintyCNN
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer: Optimizador
        device: Device (cuda/cpu)
        n_epochs: Número máximo de épocas
        n_noise_samples: T_noise para el entrenamiento
        early_stopping_patience: Paciencia para early stopping
        save_dir: Directorio para guardar el modelo
        verbose: Si imprimir progreso

    Returns:
        dict con modelo, history, mejor val_loss y tiempo total
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("INICIO DE ENTRENAMIENTO CON INCERTIDUMBRE")
        print("=" * 70)
        print(f"Épocas máximas: {n_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"T_noise (ruido en entrenamiento): {n_noise_samples}")
        print(f"Device: {device}")
        print("=" * 70 + "\n")

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # ============================================================
        # TRAINING
        # ============================================================
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            X = batch["spectrogram"].to(device)
            y = batch["label"].to(device)

            # Forward
            logits, s_logit = model(X)

            # Loss heteroscedástica
            loss = heteroscedastic_classification_loss(
                logits, s_logit, y, n_noise_samples=n_noise_samples
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Métricas
            train_loss += loss.item() * X.size(0)

            # Accuracy (usando logits sin ruido)
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += X.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ============================================================
        # VALIDATION
        # ============================================================
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                X = batch["spectrogram"].to(device)
                y = batch["label"].to(device)

                # Forward (sin ruido para validación)
                logits, s_logit = model(X)

                loss = heteroscedastic_classification_loss(
                    logits, s_logit, y, n_noise_samples=n_noise_samples
                )

                val_loss += loss.item() * X.size(0)

                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += X.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Guardar history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        if verbose:
            print(
                f"Época {epoch + 1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # Guardar mejor modelo
            if save_dir:
                torch.save(model.state_dict(), save_dir / "best_model_uncertainty.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\n⚠️  Early stopping en época {epoch + 1}")
                print(f"    Mejor época: {best_epoch}")
                print(f"    Mejor val_loss: {best_val_loss:.4f}")
            break

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Tiempo total: {total_time / 60:.1f} minutos")
        print(f"Mejor val_loss: {best_val_loss:.4f}")
        print("=" * 70 + "\n")

    # Cargar mejor modelo
    if save_dir:
        model.load_state_dict(torch.load(save_dir / "best_model_uncertainty.pth"))

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
    }


def evaluate_with_uncertainty(model, loader, device, n_mc_samples=30, class_names=None):
    """
    Evalúa modelo con MC Dropout para obtener incertidumbres.

    Args:
        model: Modelo UncertaintyCNN
        loader: DataLoader
        device: Device
        n_mc_samples: T_test para MC Dropout
        class_names: Nombres de clases

    Returns:
        dict con métricas e incertidumbres
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_probs_mean = []
    all_confidence = []
    all_entropy = []
    all_epistemic = []
    all_aleatoric = []

    for batch in loader:
        X = batch["spectrogram"].to(device)
        y = batch["label"]

        # Predicción con incertidumbre
        results = model.predict_with_uncertainty(X, n_samples=n_mc_samples)

        all_preds.append(results["pred"].cpu())
        all_targets.append(y)
        all_probs_mean.append(results["probs_mean"].cpu())
        all_confidence.append(results["confidence"].cpu())
        all_entropy.append(results["entropy_total"].cpu())
        all_epistemic.append(results["epistemic"].cpu())
        all_aleatoric.append(results["aleatoric"].cpu())

    # Concatenar todo
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    probs_mean = torch.cat(all_probs_mean)
    confidence = torch.cat(all_confidence)
    entropy = torch.cat(all_entropy)
    epistemic = torch.cat(all_epistemic)
    aleatoric = torch.cat(all_aleatoric)

    # Métricas básicas
    accuracy = accuracy_score(targets.numpy(), preds.numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.numpy(), preds.numpy(), average="macro"
    )

    # NLL, Brier, ECE
    nll = compute_nll(probs_mean, targets)
    brier = compute_brier_score(probs_mean, targets, n_classes=probs_mean.shape[1])
    ece = compute_ece(probs_mean, targets)

    # Incertidumbres promedio
    mean_entropy = entropy.mean().item()
    mean_epistemic = epistemic.mean().item()
    mean_aleatoric = aleatoric.mean().item()

    # Separar correctos vs incorrectos
    correct_mask = preds == targets

    entropy_correct = (
        entropy[correct_mask].mean().item() if correct_mask.sum() > 0 else 0
    )
    entropy_incorrect = (
        entropy[~correct_mask].mean().item() if (~correct_mask).sum() > 0 else 0
    )

    epistemic_correct = (
        epistemic[correct_mask].mean().item() if correct_mask.sum() > 0 else 0
    )
    epistemic_incorrect = (
        epistemic[~correct_mask].mean().item() if (~correct_mask).sum() > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_entropy": mean_entropy,
        "mean_epistemic": mean_epistemic,
        "mean_aleatoric": mean_aleatoric,
        "entropy_correct": entropy_correct,
        "entropy_incorrect": entropy_incorrect,
        "epistemic_correct": epistemic_correct,
        "epistemic_incorrect": epistemic_incorrect,
        "predictions": preds.numpy(),
        "targets": targets.numpy(),
        "probs_mean": probs_mean.numpy(),
        "confidence": confidence.numpy(),
        "entropy": entropy.numpy(),
        "epistemic": epistemic.numpy(),
        "aleatoric": aleatoric.numpy(),
    }


def print_uncertainty_results(metrics, class_names=None):
    """Imprime resultados con incertidumbre."""
    print("\n" + "=" * 70)
    print("RESULTADOS CON INCERTIDUMBRE")
    print("=" * 70)

    print("\n📊 MÉTRICAS DE CLASIFICACIÓN:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    print("\n📈 CALIBRACIÓN:")
    print(f"  NLL:         {metrics['nll']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  ECE:         {metrics['ece']:.4f}")

    print("\n🎲 INCERTIDUMBRES PROMEDIO:")
    print(f"  Entropía total (predictiva): {metrics['mean_entropy']:.4f}")
    print(f"  Epistémica (BALD):           {metrics['mean_epistemic']:.4f}")
    print(f"  Aleatoria (σ²):              {metrics['mean_aleatoric']:.4f}")

    print("\n✅ ❌ INCERTIDUMBRE POR ACIERTO/ERROR:")
    print(f"  Entropía (correctos):   {metrics['entropy_correct']:.4f}")
    print(f"  Entropía (incorrectos): {metrics['entropy_incorrect']:.4f}")
    print(f"  Epistémica (correctos):   {metrics['epistemic_correct']:.4f}")
    print(f"  Epistémica (incorrectos): {metrics['epistemic_incorrect']:.4f}")

    print("=" * 70 + "\n")
