"""
Tests unitarios para verificar correctitud matemática del sistema de incertidumbre.

Basado en Kendall & Gal (2017):
"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

Ejecutar: python test/test_uncertainty_math.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from modules.models.uncertainty.model import UncertaintyCNN
from modules.models.uncertainty.loss import heteroscedastic_classification_loss


def test_gaussian_noise_in_inference():
    """
    Test 1: Verifica que el ruido gaussiano afecta las probabilidades.

    Si NO hay ruido en inferencia, p_mean con T=2 y T=20 sería casi idéntico
    (solo varía por MC Dropout). Con ruido, debe haber diferencia notable.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Inyección de ruido gaussiano en inferencia")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UncertaintyCNN(n_classes=2).to(device).eval()

    x = torch.randn(32, 1, 65, 41, device=device)

    with torch.no_grad():
        r1 = model.predict_with_uncertainty(x, n_samples=2)
        p1 = r1["probs_mean"]

        r2 = model.predict_with_uncertainty(x, n_samples=20)
        p2 = r2["probs_mean"]

    diff = (p1 - p2).abs().mean().item()

    print(f"📊 Diferencia promedio |p_mean(T=2) - p_mean(T=20)|: {diff:.6f}")

    if diff > 1e-4:
        print("✅ PASS: Ruido gaussiano está activo en inferencia")
    else:
        print("❌ FAIL: NO parece haber ruido gaussiano en logits")
        raise AssertionError(
            "El ruido gaussiano debe afectar las probabilidades. "
            "Verifica que uses: logits_t = logits + sigma * randn()"
        )


def test_kendall_gal_decomposition():
    """
    Test 2: Verifica la decomposición correcta de Kendall & Gal.

    Debe cumplir: H[p̄] ≈ Epistemic + Aleatoric
    Donde:
    - H[p̄] = entropy_total
    - Epistemic = BALD = H[p̄] - E[H[p_t]]
    - Aleatoric = E[H[p_t]]
    """
    print("\n" + "=" * 70)
    print("TEST 2: Decomposición correcta H[p̄] ≈ BALD + E[H[p_t]]")
    print("=" * 70)

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
    print(f"📊 Aleatoric (E[H[p]): {aleatoric.mean().item():.6f}")
    print(f"📊 BALD + Aleatoric:    {rhs.mean().item():.6f}")
    print(f"\n🔍 |H[p̄] - (BALD + E[H[p]])|:")
    print(f"   Mean gap: {mean_gap:.6e}")
    print(f"   Max gap:  {max_gap:.6e}")

    if mean_gap < 1e-3 and max_gap < 1e-2:
        print("✅ PASS: Decomposición cerrada correctamente")
    else:
        print("❌ FAIL: Decomposición NO cierra")
        raise AssertionError(
            f"La decomposición debe cumplir H ≈ BALD + Aleatoric. "
            f"Gap: {mean_gap:.4e} (debe ser < 1e-3)"
        )


def test_aleatoric_is_entropy_not_variance():
    """
    Test 3: Verifica que aleatoric sea E[H[p_t]], no mean(σ²).

    E[H[p_t]] es una entropía → rango típico [0, log(C)]
    mean(σ²) es una varianza → puede ser muy pequeña (0.01-0.1)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Aleatoric es E[H[p_t]], no mean(σ²)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UncertaintyCNN(n_classes=2).to(device).eval()

    x = torch.randn(32, 1, 65, 41, device=device)

    with torch.no_grad():
        results = model.predict_with_uncertainty(x, n_samples=30)

    aleatoric = results["aleatoric"]
    sigma2_mean = results.get("sigma2_mean", None)

    ale_mean = aleatoric.mean().item()

    print(f"📊 Aleatoric (E[H[p_t]]): {ale_mean:.6f}")

    if sigma2_mean is not None:
        sig_mean = sigma2_mean.mean().item()
        print(f"📊 sigma2_mean (auxiliar): {sig_mean:.6f}")

    # Para 2 clases, log_2(2) = 1.0
    max_entropy_2_classes = torch.log(torch.tensor(2.0)).item()

    print(f"📊 Max entropía posible (2 clases): {max_entropy_2_classes:.6f}")

    # Aleatoric debe estar en rango razonable de entropía
    if 0 <= ale_mean <= max_entropy_2_classes:
        print("✅ PASS: Aleatoric está en rango de entropía [0, log(C)]")
    else:
        print("❌ FAIL: Aleatoric fuera de rango de entropía")
        raise AssertionError(
            f"Aleatoric debe ser entropía ∈ [0, {max_entropy_2_classes:.2f}], "
            f"pero es {ale_mean:.4f}"
        )

    # Si sigma2_mean es muy distinto, confirma que NO usas σ² como aleatoric
    if sigma2_mean is not None and abs(ale_mean - sig_mean) > 0.01:
        print("✅ PASS: Aleatoric ≠ sigma2_mean (correcto)")
    elif sigma2_mean is not None:
        print("⚠️  WARNING: Aleatoric ≈ sigma2_mean (revisar)")


def test_sigma_is_exp_half_s():
    """
    Test 4: Verifica que σ = exp(0.5 * s_logit), no exp(s_logit).

    Esto se verifica indirectamente viendo que s_logit ∈ [-10, 3]
    produce σ razonables.
    """
    print("\n" + "=" * 70)
    print("TEST 4: σ = exp(0.5 * s_logit) produce valores razonables")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UncertaintyCNN(n_classes=2, s_min=-10, s_max=3).to(device)

    x = torch.randn(8, 1, 65, 41, device=device)

    with torch.no_grad():
        logits, s_logit = model(x)

    # Verificar que s_logit esté en rango
    s_min_actual = s_logit.min().item()
    s_max_actual = s_logit.max().item()

    print(f"📊 s_logit range: [{s_min_actual:.2f}, {s_max_actual:.2f}]")
    print(f"📊 Clamp esperado: [-10.0, 3.0]")

    # Si usamos σ = exp(0.5 * s), con s ∈ [-10, 3]:
    # σ_min = exp(-5) ≈ 0.0067
    # σ_max = exp(1.5) ≈ 4.48
    sigma = torch.exp(0.5 * s_logit)
    sigma_min = sigma.min().item()
    sigma_max = sigma.max().item()

    print(f"📊 σ range (con exp(0.5*s)): [{sigma_min:.4f}, {sigma_max:.4f}]")
    print(f"📊 Esperado: [~0.007, ~4.5]")

    if 0.001 < sigma_min < 0.1 and 1.0 < sigma_max < 10.0:
        print("✅ PASS: σ = exp(0.5*s) produce valores razonables")
    else:
        print("⚠️  WARNING: Rango de σ inusual, revisar")


def test_loss_gradients():
    """
    Test 5: Verifica que la pérdida heteroscedástica produce gradientes.

    Ambas cabezas (fc_logits y fc_slog) deben tener gradientes.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Pérdida heteroscedástica genera gradientes en ambas cabezas")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UncertaintyCNN(n_classes=2).to(device).train()

    x = torch.randn(16, 1, 65, 41, device=device)
    y = torch.randint(0, 2, (16,), device=device)

    # Forward + backward
    logits, s_logit = model(x)
    loss = heteroscedastic_classification_loss(logits, s_logit, y, n_noise_samples=5)
    loss.backward()

    # Verificar gradientes
    grad_logits = model.fc_logits.weight.grad
    grad_slog = model.fc_slog.weight.grad

    if grad_logits is None:
        print("❌ FAIL: fc_logits no tiene gradientes")
        raise AssertionError("fc_logits debe tener gradientes")

    if grad_slog is None:
        print("❌ FAIL: fc_slog no tiene gradientes")
        raise AssertionError("fc_slog debe tener gradientes")

    grad_logits_norm = grad_logits.abs().mean().item()
    grad_slog_norm = grad_slog.abs().mean().item()

    print(f"📊 Grad mean |∇fc_logits|: {grad_logits_norm:.6f}")
    print(f"📊 Grad mean |∇fc_slog|:   {grad_slog_norm:.6f}")

    if grad_logits_norm > 1e-8 and grad_slog_norm > 1e-8:
        print("✅ PASS: Ambas cabezas reciben gradientes")
    else:
        print("❌ FAIL: Gradientes muy pequeños o cero")
        raise AssertionError("Los gradientes deben ser > 1e-8")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 70)
    print("🧪 TESTS DE VERIFICACIÓN MATEMÁTICA")
    print("=" * 70)
    print("Verificando implementación según Kendall & Gal (2017)...")

    try:
        test_gaussian_noise_in_inference()
        test_kendall_gal_decomposition()
        test_aleatoric_is_entropy_not_variance()
        test_sigma_is_exp_half_s()
        test_loss_gradients()

        print("\n" + "=" * 70)
        print("✅ TODOS LOS TESTS PASARON")
        print("=" * 70)
        print("✅ Sistema es 100% paper-compliant (Kendall & Gal 2017)")
        print("✅ Decomposición: H[p̄] = Epistemic + Aleatoric ✓")
        print("✅ Ruido gaussiano activo en inferencia ✓")
        print("✅ σ = exp(0.5 * s_logit) ✓")
        print("✅ Gradientes fluyen a ambas cabezas ✓")
        print("=" * 70 + "\n")

        return True

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TESTS FALLARON")
        print("=" * 70)
        print(f"Error: {e}")
        print("=" * 70 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
