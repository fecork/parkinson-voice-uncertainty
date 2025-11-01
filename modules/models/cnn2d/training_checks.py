"""
training_checks.py
Utilities to run quick exploratory checks before a long training run.

Usage in a notebook:
--------------------
from training_checks import run_all_checks

ready, report = run_all_checks(
    build_model=build_model,  # callable returns NEW model instance
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,  # torch.device("cuda" if available else "cpu")
    long_run_params={
        "optimizer": "SGD",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "scheduler": "LambdaLR",
        "lr_lambda_str": "lambda epoch: 0.95**epoch",
        "drop_conv": 0.2,
        "drop_fc": 0.2,
    },
    toy_samples=40,
    toy_steps=120,
    lr_start=1e-4,
    lr_end=1.0,
    mini_epochs=5,
    early_stop_patience=3,
)
print(report)
if ready:
    print("✅ Procede al entrenamiento largo.")
else:
    print("⚠️ Aún no listo; revisa el reporte.")

Uso directo de overfit_toy con AdamW (modo debug):
---------------------------------------------------
from training_checks import overfit_toy

result = overfit_toy(
    build_model=build_model,
    train_loader=train_loader,
    device=device,
    toy_samples=40,
    steps=120,
    lr=1e-3,  # LR seguro para AdamW
    optimizer="AdamW",  # Usar AdamW en lugar de SGD
    weight_decay=1e-4,
    debug=True,  # Mostrar información adicional
)
print(f"Overfit OK: {result['ok']}, "
      f"best_acc: {result['best']['acc']:.3f}")

Nota: La función overfit_toy fija automáticamente seed=42 para
garantizar reproducibilidad del batch exacto. Con augmentation
desactivada (tensor fijo X, Y), el mismo batch se usa en cada
ejecución.
"""

from typing import Callable, Tuple, Dict, Any, Optional, List
import math
import torch
from torch import nn, optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)


def _extract_x_y(batch):
    """Extrae xb, yb de un batch que puede ser dict o tuple."""
    if isinstance(batch, dict):
        xb = batch.get("spectrogram", batch.get("X", None))
        yb = batch.get("label", batch.get("y_task", None))
        if xb is None or yb is None:
            msg = (
                f"DictDataset debe tener claves 'spectrogram'/'X' "
                f"y 'label'/'y_task'. Claves encontradas: {list(batch.keys())}"
            )
            raise ValueError(msg)
        return xb, yb
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    else:
        msg = (
            f"Formato de batch no soportado: {type(batch)}. "
            f"Esperado: dict o tuple/list con al menos 2 elementos"
        )
        raise ValueError(msg)


def _first_batch(loader):
    """
    Extrae el primer batch del loader.
    Maneja tanto formatos de tupla (xb, yb) como diccionario.
    """
    batch = next(iter(loader))
    return _extract_x_y(batch)


def _softmax_ok(logits):
    with torch.no_grad():
        p = torch.softmax(logits, dim=1)
        s = p.sum(dim=1)
        return torch.allclose(s, torch.ones_like(s), atol=1e-4)


def smoke_test(model: torch.nn.Module, train_loader, device) -> Dict[str, Any]:
    model = model.to(device).eval()
    xb, yb = _first_batch(train_loader)
    xb = xb.to(device)
    yb = yb.to(device)
    out: Dict[str, Any] = {}
    with torch.no_grad():
        logits = model(xb)
        out["x_shape"] = tuple(xb.shape)
        out["logits_shape"] = tuple(logits.shape)
        out["softmax_sums_to_1"] = _softmax_ok(logits)
        crit = nn.CrossEntropyLoss()
        loss = crit(logits, yb)
        out["loss_value"] = float(loss)
        out["ok"] = (
            len(out["logits_shape"]) == 2
            and out["logits_shape"][1] >= 2
            and out["softmax_sums_to_1"]
            and math.isfinite(out["loss_value"])
        )
    return out


def overfit_toy(
    build_model: Callable[[], torch.nn.Module],
    train_loader,
    device,
    toy_samples: int = 40,
    steps: int = 120,
    lr: float = 3e-4,  # default seguro para memorizar
    momentum: float = 0.9,  # momentum útil si usas SGD
    weight_decay: float = 0.0,
    optimizer: str = "SGD",  # "SGD" | "AdamW"
    debug: bool = False,
) -> Dict[str, Any]:
    # Fijar seed para reproducibilidad del batch exacto
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Determinismo del kernel (para que el toy sea idéntico)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build fresh model for the toy task
    model = build_model().to(device).train()
    # Collect a small batch of data
    xs, ys = [], []
    for batch in train_loader:
        xb, yb = _extract_x_y(batch)
        xs.append(xb)
        ys.append(yb)
        if sum(len(t) for t in xs) >= toy_samples:
            break
    X = torch.cat(xs, dim=0)[:toy_samples].to(device)
    Y = torch.cat(ys, dim=0)[:toy_samples].to(device)

    # Seleccionar optimizador
    if optimizer.upper() == "SGD":
        opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer.upper() == "ADAMW":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        msg = f"Optimizer '{optimizer}' no soportado. Use 'SGD' o 'AdamW'"
        raise ValueError(msg)

    if debug:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[DEBUG] Optimizer={optimizer.upper()} lr={lr} wd={weight_decay}")
        print(
            f"[DEBUG] Model params: {total_params:,} total | "
            f"{trainable_params:,} trainable"
        )

    crit = nn.CrossEntropyLoss()

    best = {"acc": 0.0, "loss": float("inf"), "step": 0}
    history: List[Tuple[int, float, float]] = []  # (step, loss, acc)

    for step in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, Y)
        loss.backward()
        if debug:
            # Norma L2 del gradiente de la primera capa (sanity check)
            first_param = next(model.parameters())
            grad_norm = first_param.grad.detach().data.norm(2).item()
            if step % 10 == 0:
                print(
                    f"[DEBUG] step={step:03d} loss={loss.item():.4f} "
                    f"grad||={grad_norm:.4e}"
                )
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == Y).float().mean().item()
        history.append((step, float(loss.item()), float(acc)))
        if acc > best["acc"] or loss.item() < best["loss"]:
            best = {
                "acc": float(acc),
                "loss": float(loss.item()),
                "step": step,
            }
        # Early stop por éxito (ahorra tiempo cuando ya memorizó)
        if acc >= 0.99:
            if debug:
                print(f"[DEBUG] Early stop @ step {step}: acc={acc:.4f} >= 0.99")
            break

    ok = best["acc"] >= 0.95
    return {"ok": ok, "best": best, "history": history}


def lr_range_test(
    build_model: Callable[[], torch.nn.Module],
    train_loader,
    device,
    lr_start: float = 1e-4,
    lr_end: float = 1.0,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> Dict[str, Any]:
    model = build_model().to(device).train()
    opt = optim.SGD(
        model.parameters(),
        lr=lr_start,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    crit = nn.CrossEntropyLoss()

    num_steps = max(1, len(train_loader))
    best_lr, best_loss = None, float("inf")
    exploded_at = None
    hist = []  # (lr, loss)

    for i, batch in enumerate(train_loader):
        xb, yb = _extract_x_y(batch)
        xb, yb = xb.to(device), yb.to(device)
        lr = lr_start * (lr_end / lr_start) ** (i / (num_steps - 1))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            exploded_at = lr
            break

        loss.backward()
        opt.step()

        lval = float(loss.item())
        hist.append((float(lr), lval))
        if lval < best_loss:
            best_lr, best_loss = float(lr), lval

    ok = best_lr is not None and exploded_at is None
    return {
        "ok": ok,
        "best_lr": best_lr,
        "best_loss": best_loss,
        "exploded_at": exploded_at,
        "history": hist,
    }


def _run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        xb, yb = _extract_x_y(batch)
        xb, yb = xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(xb)
        if criterion:
            loss = criterion(logits, yb)
        else:
            loss = torch.tensor(0.0, device=device)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        preds.append(logits.argmax(1).detach().cpu())
        gts.append(yb.detach().cpu())
    import torch as _torch

    preds = _torch.cat(preds).numpy()
    gts = _torch.cat(gts).numpy()
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    cm = confusion_matrix(gts, preds)
    return avg_loss, acc, f1, cm


def mini_train_valid(
    build_model: Callable[[], torch.nn.Module],
    train_loader,
    val_loader,
    device,
    epochs: int = 5,
    lr: float = 0.1,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    early_stop_patience: int = 3,
) -> Dict[str, Any]:
    model = build_model().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best = {
        "val_loss": float("inf"),
        "val_acc": None,
        "val_f1": None,
        "epoch": 0,
    }
    bad = 0
    history = []  # (ep, tr_loss, tr_acc, va_loss, va_acc)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _ = _run_epoch(
            model, train_loader, device, optimizer, criterion
        )
        va_loss, va_acc, va_f1, va_cm = _run_epoch(
            model, val_loader, device, None, criterion
        )
        history.append((ep, tr_loss, tr_acc, va_loss, va_acc, va_f1))

        improved = va_loss < best["val_loss"] - 1e-4
        if improved:
            best.update(
                {
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "val_f1": va_f1,
                    "epoch": ep,
                    "cm": va_cm,
                }
            )
            bad = 0
        else:
            bad += 1
            if bad >= early_stop_patience:
                break

    # Expectations: train loss down; val loss not exploding; acc > 0.5 (binary)
    ok = (
        len(history) > 0
        and history[-1][2] >= 0.55  # last train acc
        and best["val_acc"] is not None
        and best["val_acc"] >= 0.55
        and math.isfinite(best["val_loss"])
    )

    return {"ok": ok, "best": best, "history": history}


def quick_diag(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device,
) -> Dict[str, Any]:
    """
    Quick diagnostic: forward pass, manual training step, validation metrics.

    This function performs:
    1. Forward pass validation with softmax check
    2. Manual training step to verify gradients and weight updates
    3. Validation metrics calculation (accuracy, recall, specificity, f1)

    Args:
        model: The model to diagnose
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on

    Returns:
        Dictionary with 'ok' flag and diagnostic metrics
    """
    model = model.to(device)

    # === 1) Forward pass + CE loss ===
    model.eval()
    batch = next(iter(train_loader))
    xb, yb = _extract_x_y(batch)

    if not torch.is_tensor(yb):
        yb = torch.tensor(yb)
    xb, yb = xb.to(device), yb.to(device).long()

    with torch.no_grad():
        logits = model(xb)

    logits_shape = tuple(logits.shape)

    # Check if output is already probabilities (softmax in forward)
    # If logits are probs, they should be in [0,1] and sum to 1 per row
    row_sums = logits.sum(1)
    looks_like_probs = (
        (logits.min().item() >= 0.0)
        and (logits.max().item() <= 1.0)
        and torch.allclose(
            row_sums.mean(), torch.tensor(1.0, device=logits.device), atol=1e-3
        )
    )

    # If not probabilities, verify softmax would sum to 1 (sanity check)
    if not looks_like_probs:
        softmax_sums_to_1 = bool(
            torch.allclose(
                torch.softmax(logits, 1).sum(1).mean(),
                torch.tensor(1.0, device=logits.device),
                atol=1e-6,
            )
        )
    else:
        softmax_sums_to_1 = True

    ce_loss = nn.CrossEntropyLoss()
    eval_loss = ce_loss(logits, yb)

    # === 2) Manual training step ===
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    logits_before = model(xb)
    loss_before = ce_loss(logits_before, yb)
    opt.zero_grad(set_to_none=True)
    loss_before.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    params_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt.step()

    with torch.no_grad():
        loss_after = ce_loss(model(xb), yb)
        weight_update_norm = torch.sqrt(
            sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), params_before))
        ).item()

    # === 3) Validation metrics ===
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            xb, yb = _extract_x_y(batch)
            if not torch.is_tensor(yb):
                yb = torch.tensor(yb)
            xb, yb = xb.to(device), yb.to(device).long()
            logits = model(xb)
            preds = logits.argmax(1)
            y_true += yb.cpu().tolist()
            y_pred += preds.cpu().tolist()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp + 1e-12) if (tn + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)

    # Check if diagnostics are OK
    ok = (
        softmax_sums_to_1
        and math.isfinite(eval_loss.item())
        and math.isfinite(loss_before.item())
        and math.isfinite(loss_after.item())
        and math.isfinite(float(grad_norm))
        and math.isfinite(weight_update_norm)
        and len(logits_shape) == 2
        and logits_shape[1] >= 2
    )

    return {
        "ok": ok,
        "logits_shape": logits_shape,
        "softmax_sums_to_1": softmax_sums_to_1,
        "looks_like_probs": looks_like_probs,
        "eval_loss": float(eval_loss.item()),
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "grad_norm": float(grad_norm),
        "weight_update_norm": float(weight_update_norm),
        "val_accuracy": accuracy,
        "val_recall": recall,
        "val_specificity": specificity,
        "val_f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_all_checks(
    build_model: Callable[[], torch.nn.Module],
    train_loader,
    val_loader,
    device,
    long_run_params: Optional[Dict[str, Any]] = None,
    toy_samples: int = 40,
    toy_steps: int = 120,
    lr_start: float = 1e-4,
    lr_end: float = 1.0,
    mini_epochs: int = 5,
    early_stop_patience: int = 3,
) -> Tuple[bool, str]:
    # 1) Smoke test
    smoke = smoke_test(build_model().to(device), train_loader, device)

    # 2) LR range test (antes que el toy)
    lrtest = lr_range_test(
        build_model, train_loader, device, lr_start=lr_start, lr_end=lr_end
    )
    # LR recomendado para toy (clamp a rango estable)
    toy_lr = 3e-4
    if (
        lrtest["ok"]
        and lrtest["best_lr"] is not None
        and math.isfinite(lrtest["best_lr"])
    ):
        toy_lr = float(max(1e-5, min(1e-2, lrtest["best_lr"])))

    # 3) Overfit toy con LR recomendado
    toy = overfit_toy(
        build_model,
        train_loader,
        device,
        toy_samples=toy_samples,
        steps=toy_steps,
        lr=toy_lr,
        momentum=0.9,
        weight_decay=0.0,
        optimizer="SGD",  # si quieres: "AdamW"
        debug=False,
    )

    # 4) Mini-train with validation
    mini = mini_train_valid(
        build_model,
        train_loader,
        val_loader,
        device,
        epochs=mini_epochs,
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        early_stop_patience=early_stop_patience,
    )

    # 5) Quick diagnostic
    diag = quick_diag(build_model().to(device), train_loader, val_loader, device)

    # Aggregate decision
    ready = smoke["ok"] and toy["ok"] and lrtest["ok"] and mini["ok"] and diag["ok"]

    # Build report
    lines = []
    lines.append("🧪 Exploratory Training Checks Report")
    lines.append("--------------------------------------------------")
    smoke_status = "OK" if smoke["ok"] else "FAIL"
    smoke_line = (
        f"Smoke test: {smoke_status}  | X {smoke['x_shape']} → "
        f"logits {smoke['logits_shape']} | "
        f"softmax_ok={smoke['softmax_sums_to_1']} | "
        f"loss={smoke['loss_value']:.4f}"
    )
    lines.append(smoke_line)
    toy_status = "OK" if toy["ok"] else "FAIL"
    toy_line = (
        f"Overfit toy: {toy_status}   | "
        f"best_acc={toy['best']['acc']:.3f} @ step {toy['best']['step']} | "
        f"best_loss={toy['best']['loss']:.4f} | toy_lr={toy_lr:.5f}"
    )
    lines.append(toy_line)
    if lrtest["ok"]:
        lr_line = (
            f"LR range:  OK   | best_lr≈{lrtest['best_lr']:.5f} "
            f"(min_loss={lrtest['best_loss']:.4f})"
        )
        lines.append(lr_line)
    else:
        lines.append(f"LR range:  FAIL | exploded_at={lrtest['exploded_at']}")
    if mini["ok"]:
        mini_line = (
            f"Mini-train: OK  | best_val_loss={mini['best']['val_loss']:.4f} "
            f"@ ep {mini['best']['epoch']} | "
            f"best_val_acc={mini['best']['val_acc']:.3f} | "
            f"best_val_f1={mini['best']['val_f1']:.3f}"
        )
        lines.append(mini_line)
    else:
        mini_fail_line = (
            f"Mini-train: FAIL | last/best -> "
            f"val_loss={mini['best']['val_loss']:.4f} | "
            f"val_acc={mini['best']['val_acc']}"
        )
        lines.append(mini_fail_line)
    diag_status = "OK" if diag["ok"] else "FAIL"
    diag_line = (
        f"Quick diag: {diag_status} | "
        f"grad_norm={diag['grad_norm']:.4f} | "
        f"weight_update={diag['weight_update_norm']:.6f} | "
        f"val_acc={diag['val_accuracy']:.3f} | "
        f"sen/rec={diag['val_recall']:.3f} | "
        f"spec={diag['val_specificity']:.3f} | "
        f"f1={diag['val_f1']:.3f}"
    )
    lines.append(diag_line)
    if long_run_params:
        lines.append("Long-run params (target): " + str(long_run_params))

    report = "\n".join(lines)
    return ready, report
