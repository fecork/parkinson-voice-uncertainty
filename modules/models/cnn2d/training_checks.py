
"""
training_checks.py
Utilities to run quick exploratory checks before a long training run.

Usage in a notebook:
--------------------
from training_checks import run_all_checks

ready, report = run_all_checks(
    build_model=build_model,           # callable that returns a NEW model instance
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,                     # torch.device("cuda" if available else "cpu")
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
"""

from typing import Callable, Tuple, Dict, Any, Optional, List
import math
import copy
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def _first_batch(loader):
    for xb, yb in loader:
        return xb, yb
    raise RuntimeError("Loader vacío")

def _softmax_ok(logits):
    with torch.no_grad():
        p = torch.softmax(logits, dim=1)
        s = p.sum(dim=1)
        return torch.allclose(s, torch.ones_like(s), atol=1e-4)

def smoke_test(model: torch.nn.Module, train_loader, device) -> Dict[str, Any]:
    model = model.to(device).eval()
    xb, yb = _first_batch(train_loader)
    xb = xb.to(device); yb = yb.to(device)
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
            len(out["logits_shape"]) == 2 and
            out["logits_shape"][1] >= 2 and
            out["softmax_sums_to_1"] and
            math.isfinite(out["loss_value"])
        )
    return out

def overfit_toy(build_model: Callable[[], torch.nn.Module],
                train_loader,
                device,
                toy_samples: int = 40,
                steps: int = 120,
                lr: float = 0.1,
                momentum: float = 0.0,
                weight_decay: float = 0.0) -> Dict[str, Any]:

    # Build fresh model for the toy task
    model = build_model().to(device).train()
    # Collect a small batch of data
    xs, ys = [], []
    for xb, yb in train_loader:
        xs.append(xb); ys.append(yb)
        if sum(len(t) for t in xs) >= toy_samples:
            break
    X = torch.cat(xs, dim=0)[:toy_samples].to(device)
    Y = torch.cat(ys, dim=0)[:toy_samples].to(device)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best = {"acc": 0.0, "loss": float("inf"), "step": 0}
    history: List[Tuple[int, float, float]] = []  # (step, loss, acc)

    for step in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, Y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == Y).float().mean().item()
        history.append((step, float(loss.item()), float(acc)))
        if acc > best["acc"] or loss.item() < best["loss"]:
            best = {"acc": float(acc), "loss": float(loss.item()), "step": step}

    ok = best["acc"] >= 0.95  # should roughly reach ~1.0
    return {"ok": ok, "best": best, "history": history}

def lr_range_test(build_model: Callable[[], torch.nn.Module],
                  train_loader,
                  device,
                  lr_start: float = 1e-4,
                  lr_end: float = 1.0,
                  momentum: float = 0.0,
                  weight_decay: float = 0.0) -> Dict[str, Any]:

    model = build_model().to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr_start, momentum=momentum, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    num_steps = max(1, len(train_loader))
    best_lr, best_loss = None, float("inf")
    exploded_at = None
    hist = []  # (lr, loss)

    for i, (xb, yb) in enumerate(train_loader):
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
    return {"ok": ok, "best_lr": best_lr, "best_loss": best_loss, "exploded_at": exploded_at, "history": hist}

def _run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, preds, gts = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb) if criterion else torch.tensor(0.0, device=device)
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

def mini_train_valid(build_model: Callable[[], torch.nn.Module],
                     train_loader,
                     val_loader,
                     device,
                     epochs: int = 5,
                     lr: float = 0.1,
                     momentum: float = 0.0,
                     weight_decay: float = 0.0,
                     early_stop_patience: int = 3) -> Dict[str, Any]:

    model = build_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = {"val_loss": float("inf"), "val_acc": None, "val_f1": None, "epoch": 0}
    bad = 0
    history = []  # (ep, tr_loss, tr_acc, va_loss, va_acc)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _ = _run_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, va_f1, va_cm = _run_epoch(model, val_loader, device, None, criterion)
        history.append((ep, tr_loss, tr_acc, va_loss, va_acc, va_f1))

        improved = va_loss < best["val_loss"] - 1e-4
        if improved:
            best.update({"val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1, "epoch": ep, "cm": va_cm})
            bad = 0
        else:
            bad += 1
            if bad >= early_stop_patience:
                break

    # Expectations: train loss down; val loss not exploding; acc > 0.5 (binary)
    ok = (
        len(history) > 0 and
        history[-1][2] >= 0.55 and   # last train acc
        best["val_acc"] is not None and best["val_acc"] >= 0.55 and
        math.isfinite(best["val_loss"])
    )

    return {"ok": ok, "best": best, "history": history}

def run_all_checks(build_model: Callable[[], torch.nn.Module],
                   train_loader,
                   val_loader,
                   device,
                   long_run_params: Optional[Dict[str, Any]] = None,
                   toy_samples: int = 40,
                   toy_steps: int = 120,
                   lr_start: float = 1e-4,
                   lr_end: float = 1.0,
                   mini_epochs: int = 5,
                   early_stop_patience: int = 3) -> Tuple[bool, str]:

    # 1) Smoke test on a fresh model
    smoke = smoke_test(build_model().to(device), train_loader, device)

    # 2) Overfit toy
    toy = overfit_toy(build_model, train_loader, device, toy_samples=toy_samples, steps=toy_steps)

    # 3) LR range test
    lrtest = lr_range_test(build_model, train_loader, device, lr_start=lr_start, lr_end=lr_end)

    # 4) Mini-train with validation
    mini = mini_train_valid(build_model, train_loader, val_loader, device,
                            epochs=mini_epochs, lr=0.1, momentum=0.0, weight_decay=0.0,
                            early_stop_patience=early_stop_patience)

    # Aggregate decision
    ready = smoke["ok"] and toy["ok"] and lrtest["ok"] and mini["ok"]

    # Build report
    lines = []
    lines.append("🧪 Exploratory Training Checks Report")
    lines.append("--------------------------------------------------")
    lines.append(f"Smoke test: {'OK' if smoke['ok'] else 'FAIL'}  | X {smoke['x_shape']} → logits {smoke['logits_shape']} | softmax_ok={smoke['softmax_sums_to_1']} | loss={smoke['loss_value']:.4f}")
    lines.append(f"Overfit toy: {'OK' if toy['ok'] else 'FAIL'}   | best_acc={toy['best']['acc']:.3f} @ step {toy['best']['step']} | best_loss={toy['best']['loss']:.4f}")
    if lrtest["ok"]:
        lines.append(f"LR range:  OK   | best_lr≈{lrtest['best_lr']:.5f} (min_loss={lrtest['best_loss']:.4f})")
    else:
        lines.append(f"LR range:  FAIL | exploded_at={lrtest['exploded_at']}")
    if mini["ok"]:
        lines.append(f"Mini-train: OK  | best_val_loss={mini['best']['val_loss']:.4f} @ ep {mini['best']['epoch']} | best_val_acc={mini['best']['val_acc']:.3f} | best_val_f1={mini['best']['val_f1']:.3f}")
    else:
        lines.append(f"Mini-train: FAIL | last/best -> val_loss={mini['best']['val_loss']:.4f} | val_acc={mini['best']['val_acc']}")
    if long_run_params:
        lines.append("Long-run params (target): " + str(long_run_params))

    report = "\n".join(lines)
    return ready, report
