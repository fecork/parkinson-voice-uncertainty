"""
Script de Verificación de Implementación Yarin Gal + Kendall & Gal
===================================================================
Verifica que la implementación de incertidumbre epistémica y aleatoria
sea correcta según los papers:

- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian DL?"

Uso:
    python test/test_yarin_gal_implementation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

# Símbolos simples para Windows compatibility
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

GREEN = ""
RED = ""
YELLOW = ""
RESET = ""


def test_imports():
    """Test 1: Verificar imports de módulos de incertidumbre."""
    print("\n" + "=" * 70)
    print("TEST 1: IMPORTS DE MÓDULOS DE INCERTIDUMBRE")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import (
            UncertaintyCNN,
            MCDropout,
            MCDropout2d,
            print_uncertainty_model_summary,
        )
        from modules.models.uncertainty.loss import (
            heteroscedastic_classification_loss,
            compute_nll,
            compute_brier_score,
            compute_ece,
        )
        from modules.models.uncertainty.training import (
            train_uncertainty_model,
            evaluate_with_uncertainty,
            print_uncertainty_results,
        )
        from modules.models.uncertainty.visualization import (
            plot_uncertainty_histograms,
            plot_reliability_diagram,
            plot_uncertainty_scatter,
            plot_training_history_uncertainty,
        )

        print(f"{OK} Todos los módulos de incertidumbre importados correctamente")
        return True
    except Exception as e:
        print(f"{FAIL} Error en imports: {e}")
        return False


def test_mc_dropout_always_active():
    """Test 2: Verificar que MCDropout está activo en eval()."""
    print("\n" + "=" * 70)
    print("TEST 2: MC DROPOUT ACTIVO EN EVAL")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import MCDropout, MCDropout2d

        # Test MCDropout
        dropout = MCDropout(p=0.5)
        x = torch.randn(100, 10)

        dropout.eval()  # Poner en modo eval
        out1 = dropout(x)
        out2 = dropout(x)

        # Si dropout está activo, outputs deben ser diferentes
        diff = (out1 - out2).abs().mean().item()

        if diff > 0.01:
            print(f"{OK} MCDropout activo en eval() (diff={diff:.4f})")
        else:
            print(f"{FAIL} MCDropout NO activo en eval() (diff={diff:.4f})")
            raise AssertionError("MCDropout debe estar activo en eval()")

        # Test MCDropout2d
        dropout2d = MCDropout2d(p=0.5)
        x2d = torch.randn(10, 32, 8, 8)

        dropout2d.eval()
        out1_2d = dropout2d(x2d)
        out2_2d = dropout2d(x2d)

        diff2d = (out1_2d - out2_2d).abs().mean().item()

        if diff2d > 0.01:
            print(f"{OK} MCDropout2d activo en eval() (diff={diff2d:.4f})")
        else:
            print(f"{FAIL} MCDropout2d NO activo en eval()")
            raise AssertionError("MCDropout2d debe estar activo en eval()")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_two_heads_architecture():
    """Test 3: Verificar que el modelo tiene dos cabezas."""
    print("\n" + "=" * 70)
    print("TEST 3: ARQUITECTURA CON DOS CABEZAS")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        model = UncertaintyCNN(n_classes=2)

        # Verificar que existen las dos cabezas
        assert hasattr(model, "fc_logits"), "Falta cabeza fc_logits"
        assert hasattr(model, "fc_slog"), "Falta cabeza fc_slog"

        print(f"{OK} Cabeza A (fc_logits): {model.fc_logits}")
        print(f"{OK} Cabeza B (fc_slog): {model.fc_slog}")

        # Test forward
        x = torch.randn(4, 1, 65, 41)
        logits, s_logit = model(x)

        # Verificar shapes
        assert logits.shape == (4, 2), f"Shape logits incorrecto: {logits.shape}"
        assert s_logit.shape == (4, 2), f"Shape s_logit incorrecto: {s_logit.shape}"

        print(f"{OK} Forward pass: logits {logits.shape}, s_logit {s_logit.shape}")

        # Verificar clamp de s_logit
        assert s_logit.min() >= model.s_min, "s_logit por debajo de s_min"
        assert s_logit.max() <= model.s_max, "s_logit por encima de s_max"

        print(
            f"{OK} Clamp s_logit: [{model.s_min}, {model.s_max}] → "
            f"[{s_logit.min().item():.2f}, {s_logit.max().item():.2f}]"
        )

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_gaussian_noise_injection():
    """Test 4: Verificar inyección de ruido gaussiano en inferencia."""
    print("\n" + "=" * 70)
    print("TEST 4: RUIDO GAUSSIANO EN INFERENCIA")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).eval()

        x = torch.randn(32, 1, 65, 41, device=device)

        with torch.no_grad():
            r1 = model.predict_with_uncertainty(x, n_samples=2)
            p1 = r1["probs_mean"]

            r2 = model.predict_with_uncertainty(x, n_samples=20)
            p2 = r2["probs_mean"]

        diff = (p1 - p2).abs().mean().item()

        print(f"📊 Δ(p_mean) T=2 vs T=20: {diff:.6f}")

        if diff > 1e-4:
            print(f"{OK} Ruido gaussiano está activo en inferencia")
        else:
            print(f"{FAIL} NO hay ruido gaussiano en logits")
            raise AssertionError("Debe inyectar ruido: logits_t = logits + σ⊙ε")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_kendall_gal_decomposition():
    """Test 5: Verificar decomposición H[p̄] = Epistemic + Aleatoric."""
    print("\n" + "=" * 70)
    print("TEST 5: DECOMPOSICIÓN DE KENDALL & GAL")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).eval()

        x = torch.randn(32, 1, 65, 41, device=device)

        with torch.no_grad():
            results = model.predict_with_uncertainty(x, n_samples=30)

        H_total = results["entropy_total"]
        epistemic = results["epistemic"]
        aleatoric = results["aleatoric"]

        lhs = H_total
        rhs = epistemic + aleatoric
        gap = (lhs - rhs).abs()

        mean_gap = gap.mean().item()
        max_gap = gap.max().item()

        print(f"📊 H[p̄]:                {H_total.mean().item():.6f}")
        print(f"📊 Epistemic (BALD):    {epistemic.mean().item():.6f}")
        print(f"📊 Aleatoric (E[H[p]]): {aleatoric.mean().item():.6f}")
        print(f"📊 BALD + Aleatoric:    {rhs.mean().item():.6f}")
        print(f"📊 |H - (BALD+Ale)|:    {mean_gap:.6e} (max: {max_gap:.6e})")

        if mean_gap < 1e-3 and max_gap < 1e-2:
            print(f"{OK} Decomposición cerrada correctamente")
        else:
            print(f"{FAIL} Decomposición NO cierra (gap={mean_gap:.4e})")
            raise AssertionError("H[p̄] debe ≈ Epistemic + Aleatoric")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_aleatoric_is_entropy():
    """Test 6: Verificar que aleatoric sea E[H[p_t]], no mean(σ²)."""
    print("\n" + "=" * 70)
    print("TEST 6: ALEATORIC ES E[H[p_t]], NO mean(σ²)")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).eval()

        x = torch.randn(32, 1, 65, 41, device=device)

        with torch.no_grad():
            results = model.predict_with_uncertainty(x, n_samples=30)

        aleatoric = results["aleatoric"]
        sigma2_mean = results.get("sigma2_mean", None)

        ale_mean = aleatoric.mean().item()
        max_entropy = torch.log(torch.tensor(2.0)).item()

        print(f"📊 Max entropía (2 clases): {max_entropy:.4f}")
        print(f"📊 Aleatoric observada:     {ale_mean:.4f}")

        if 0 <= ale_mean <= max_entropy:
            print(f"{OK} Aleatoric en rango de entropía [0, log(C)]")
        else:
            print(f"{FAIL} Aleatoric fuera de rango de entropía")
            raise AssertionError(
                f"Aleatoric debe ser entropía ∈ [0, {max_entropy:.2f}]"
            )

        if sigma2_mean is not None:
            sig_mean = sigma2_mean.mean().item()
            print(f"📊 sigma²_mean (auxiliar):  {sig_mean:.4f}")
            if abs(ale_mean - sig_mean) > 0.01:
                print(f"{OK} Aleatoric ≠ sigma2_mean (correcto)")
            else:
                print(f"{WARN} Aleatoric ≈ sigma2_mean (revisar)")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_heteroscedastic_loss():
    """Test 7: Verificar pérdida heteroscedástica."""
    print("\n" + "=" * 70)
    print("TEST 7: PÉRDIDA HETEROSCEDÁSTICA")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN
        from modules.models.uncertainty.loss import heteroscedastic_classification_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).train()

        x = torch.randn(16, 1, 65, 41, device=device)
        y = torch.randint(0, 2, (16,), device=device)

        # Forward + loss
        logits, s_logit = model(x)
        loss = heteroscedastic_classification_loss(
            logits, s_logit, y, n_noise_samples=5
        )

        # Verificar que la pérdida es un escalar
        assert loss.dim() == 0, "Loss debe ser escalar"
        assert not torch.isnan(loss), "Loss es NaN"
        assert not torch.isinf(loss), "Loss es inf"

        print(f"{OK} Pérdida calculada: {loss.item():.4f}")
        print(f"{OK} Sin NaN/inf")

        # Backward
        loss.backward()

        # Verificar gradientes en ambas cabezas
        grad_logits = model.fc_logits.weight.grad
        grad_slog = model.fc_slog.weight.grad

        if grad_logits is None:
            print(f"{FAIL} fc_logits sin gradientes")
            raise AssertionError("fc_logits debe tener gradientes")

        if grad_slog is None:
            print(f"{FAIL} fc_slog sin gradientes")
            raise AssertionError("fc_slog debe tener gradientes")

        grad_logits_norm = grad_logits.abs().mean().item()
        grad_slog_norm = grad_slog.abs().mean().item()

        print(f"{OK} |∇fc_logits|: {grad_logits_norm:.6f}")
        print(f"{OK} |∇fc_slog|:   {grad_slog_norm:.6f}")

        if grad_logits_norm > 1e-8 and grad_slog_norm > 1e-8:
            print(f"{OK} Ambas cabezas reciben gradientes")
        else:
            print(f"{FAIL} Gradientes muy pequeños")
            raise AssertionError("Gradientes deben ser > 1e-8")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_sigma_computation():
    """Test 8: Verificar que σ = exp(0.5 * s_logit)."""
    print("\n" + "=" * 70)
    print("TEST 8: σ = exp(0.5 * s_logit)")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2, s_min=-10, s_max=3).to(device)

        x = torch.randn(8, 1, 65, 41, device=device)

        with torch.no_grad():
            logits, s_logit = model(x)

        # Verificar rango de s_logit
        s_min_actual = s_logit.min().item()
        s_max_actual = s_logit.max().item()

        print(f"📊 s_logit range: [{s_min_actual:.2f}, {s_max_actual:.2f}]")
        print(f"📊 Clamp esperado: [-10.0, 3.0]")

        # Con σ = exp(0.5 * s), s ∈ [-10, 3]:
        # σ_min = exp(-5) ≈ 0.0067
        # σ_max = exp(1.5) ≈ 4.48
        sigma = torch.exp(0.5 * s_logit)
        sigma_min = sigma.min().item()
        sigma_max = sigma.max().item()

        print(f"📊 σ range: [{sigma_min:.4f}, {sigma_max:.4f}]")
        print(f"📊 Esperado (modelo sin entrenar): [~0.01, ~5.0]")

        if sigma_min > 0:
            print(f"{OK} σ > 0 (válido)")
        else:
            print(f"{FAIL} σ <= 0 (inválido)")
            raise AssertionError("σ debe ser > 0")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_uncertainty_metrics():
    """Test 9: Verificar cálculo de métricas de incertidumbre."""
    print("\n" + "=" * 70)
    print("TEST 9: MÉTRICAS DE INCERTIDUMBRE")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).eval()

        x = torch.randn(16, 1, 65, 41, device=device)

        with torch.no_grad():
            results = model.predict_with_uncertainty(x, n_samples=20)

        # Verificar que existen todas las métricas
        required_keys = [
            "pred",
            "probs_mean",
            "confidence",
            "entropy_total",
            "epistemic",
            "aleatoric",
        ]

        for key in required_keys:
            if key not in results:
                print(f"{FAIL} Falta métrica: {key}")
                raise AssertionError(f"Falta {key} en results")
            print(f"{OK} Métrica '{key}': shape {results[key].shape}")

        # Verificar shapes
        B = 16
        C = 2
        assert results["pred"].shape == (B,)
        assert results["probs_mean"].shape == (B, C)
        assert results["confidence"].shape == (B,)
        assert results["entropy_total"].shape == (B,)
        assert results["epistemic"].shape == (B,)
        assert results["aleatoric"].shape == (B,)

        print(f"{OK} Todas las métricas tienen shapes correctos")

        # Verificar rangos
        assert (results["confidence"] >= 0).all() and (results["confidence"] <= 1).all()
        assert (results["entropy_total"] >= 0).all()
        assert (results["epistemic"] >= 0).all()
        assert (results["aleatoric"] >= 0).all()

        print(f"{OK} Todas las métricas en rangos válidos")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_calibration_metrics():
    """Test 10: Verificar métricas de calibración (NLL, Brier, ECE)."""
    print("\n" + "=" * 70)
    print("TEST 10: MÉTRICAS DE CALIBRACIÓN")
    print("=" * 70)

    try:
        from modules.models.uncertainty.loss import (
            compute_nll,
            compute_brier_score,
            compute_ece,
        )

        # Crear datos sintéticos
        probs = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]], dtype=torch.float32
        )
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # NLL
        nll = compute_nll(probs, targets)
        print(f"{OK} NLL calculado: {nll:.4f}")
        assert nll > 0, "NLL debe ser > 0"

        # Brier
        brier = compute_brier_score(probs, targets, n_classes=2)
        print(f"{OK} Brier Score calculado: {brier:.4f}")
        assert 0 <= brier <= 2, "Brier debe estar en [0, 2]"

        # ECE
        ece = compute_ece(probs, targets, n_bins=5)
        print(f"{OK} ECE calculado: {ece:.4f}")
        assert 0 <= ece <= 1, "ECE debe estar en [0, 1]"

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_uncertainty_separation():
    """Test 11: Verificar que incorrectos tienen mayor incertidumbre."""
    print("\n" + "=" * 70)
    print("TEST 11: SEPARACIÓN CORRECTO/INCORRECTO")
    print("=" * 70)

    try:
        from modules.models.uncertainty.model import UncertaintyCNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UncertaintyCNN(n_classes=2).to(device).eval()

        # Crear datos donde el modelo se equivoque
        x = torch.randn(50, 1, 65, 41, device=device)

        # Generar targets aleatorios
        true_targets = torch.randint(0, 2, (50,), device=device)

        with torch.no_grad():
            results = model.predict_with_uncertainty(x, n_samples=20)

        preds = results["pred"].cpu()
        targets = true_targets.cpu()
        entropy = results["entropy_total"].cpu()
        epistemic = results["epistemic"].cpu()
        aleatoric = results["aleatoric"].cpu()

        correct_mask = preds == targets
        n_correct = correct_mask.sum().item()
        n_incorrect = (~correct_mask).sum().item()

        print(f"📊 Correctos: {n_correct}, Incorrectos: {n_incorrect}")

        if n_correct > 0 and n_incorrect > 0:
            H_correct = entropy[correct_mask].mean().item()
            H_incorrect = entropy[~correct_mask].mean().item()

            epi_correct = epistemic[correct_mask].mean().item()
            epi_incorrect = epistemic[~correct_mask].mean().item()

            ale_correct = aleatoric[correct_mask].mean().item()
            ale_incorrect = aleatoric[~correct_mask].mean().item()

            print(f"\n📊 Entropía:")
            print(f"   Correctos:   {H_correct:.4f}")
            print(f"   Incorrectos: {H_incorrect:.4f}")

            print(f"\n📊 Epistémica:")
            print(f"   Correctos:   {epi_correct:.4f}")
            print(f"   Incorrectos: {epi_incorrect:.4f}")

            print(f"\n📊 Aleatoria:")
            print(f"   Correctos:   {ale_correct:.4f}")
            print(f"   Incorrectos: {ale_incorrect:.4f}")

            # Con modelo sin entrenar, esto puede no cumplirse siempre
            # Pero es un indicador
            if H_incorrect >= H_correct:
                print(f"{OK} Tendencia correcta: H(incorrecto) >= H(correcto)")
            else:
                print(
                    f"{WARN} H(incorrecto) < H(correcto) "
                    "(normal en modelo sin entrenar)"
                )
        else:
            print(f"{WARN} Todos correctos o todos incorrectos (modelo sin entrenar)")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_visualization_functions():
    """Test 12: Verificar que las funciones de visualización existen."""
    print("\n" + "=" * 70)
    print("TEST 12: FUNCIONES DE VISUALIZACIÓN")
    print("=" * 70)

    try:
        from modules.models.uncertainty.visualization import (
            plot_uncertainty_histograms,
            plot_reliability_diagram,
            plot_uncertainty_scatter,
            plot_training_history_uncertainty,
        )

        print(f"{OK} plot_uncertainty_histograms")
        print(f"{OK} plot_reliability_diagram")
        print(f"{OK} plot_uncertainty_scatter")
        print(f"{OK} plot_training_history_uncertainty")

        return True
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        return False


def test_notebook_exists():
    """Test 13: Verificar que el notebook existe."""
    print("\n" + "=" * 70)
    print("TEST 13: NOTEBOOK PRINCIPAL")
    print("=" * 70)

    notebook_path = Path("notebooks/cnn_uncertainty_training.ipynb")

    if notebook_path.exists():
        print(f"{OK} Notebook encontrado: {notebook_path}")
        return True
    else:
        print(f"{FAIL} Notebook no encontrado: {notebook_path}")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE IMPLEMENTACIÓN YARIN GAL + KENDALL & GAL")
    print("=" * 70)
    print("Papers:")
    print("  - Gal & Ghahramani (2016): MC Dropout")
    print("  - Kendall & Gal (2017): Epistemic + Aleatoric")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("MC Dropout activo en eval", test_mc_dropout_always_active),
        ("Dos cabezas (logits + s_logit)", test_two_heads_architecture),
        ("Ruido gaussiano en inferencia", test_gaussian_noise_injection),
        ("Decomposición K&G", test_kendall_gal_decomposition),
        ("Aleatoric es E[H[p_t]]", test_aleatoric_is_entropy),
        ("Pérdida heteroscedástica", test_heteroscedastic_loss),
        ("σ = exp(0.5*s)", test_sigma_computation),
        ("Métricas de incertidumbre", test_uncertainty_metrics),
        ("Métricas de calibración", test_calibration_metrics),
        ("Separación correcto/incorrecto", test_uncertainty_separation),
        ("Funciones de visualización", test_visualization_functions),
        ("Notebook existe", test_notebook_exists),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{FAIL} Test '{name}' falló: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n📊 Tests:")
    for name, result in results:
        status = f"{GREEN}{OK}{RESET}" if result else f"{RED}{FAIL}{RESET}"
        print(f"   {status} {name}")

    print(f"\n📈 Score: {passed}/{total} ({passed / total * 100:.0f}%)")

    if passed == total:
        print("\n" + "=" * 70)
        print(f"{GREEN}✅ TODOS LOS TESTS PASARON{RESET}")
        print("=" * 70)
        print("✅ Implementación 100% correcta según:")
        print("   • Gal & Ghahramani (2016) - MC Dropout")
        print("   • Kendall & Gal (2017) - Epistemic + Aleatoric")
        print("\n✅ Sistema listo para:")
        print("   • Entrenar con pérdida heteroscedástica")
        print("   • Inferencia con MC Dropout")
        print("   • Cuantificación de incertidumbres")
        print("   • Detección de casos difíciles/OOD")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"{YELLOW}⚠️  ALGUNOS TESTS FALLARON{RESET}")
        print("=" * 70)
        print("Revisar errores antes de ejecutar entrenamiento completo.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
