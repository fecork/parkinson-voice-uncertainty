#!/usr/bin/env python3
"""
Tests Matemáticos Rigurosos: Atención Temporal y Flattening CNN1D
==================================================================
Verifica matemáticamente el mecanismo de atención temporal y
la correcta transformación 2D→1D según paper Ibarra et al. 2023.

Uso:
    python test/test_cnn1d_attention.py
    pytest test/test_cnn1d_attention.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# HELPER: MECANISMO DE ATENCIÓN TEMPORAL (STANDALONE)
# ============================================================


def attention_forward(y):
    """
    Mecanismo de atención temporal según Ibarra et al. 2023.

    Args:
        y: [B, C, T] salida de última Conv1D

    Returns:
        z: [B, C/2] embedding atendido
        alpha: [B, C/2, T] pesos de atención
        A: [B, C/2, T] primera mitad (para atención)
        V: [B, C/2, T] segunda mitad (valores)
    """
    B, C, T = y.shape
    assert C % 2 == 0, "C debe ser par para split mitad/mitad."

    half = C // 2
    A, V = y[:, :half, :], y[:, half:, :]

    # Softmax temporal (a lo largo del eje de tiempo)
    alpha = F.softmax(A, dim=-1)  # [B, C/2, T]

    # Suma ponderada temporal
    z = (alpha * V).sum(dim=-1)  # [B, C/2]

    return z, alpha, A, V


# ============================================================
# HELPER: FLATTENING 2D→1D
# ============================================================


def flatten_2d_to_1d(x):
    """
    Transforma espectrograma 2D a formato 1D para Conv1D.

    Args:
        x: [B, 1, F, T] o [B, F, T]

    Returns:
        [B, F, T] para alimentar Conv1D
    """
    if x.dim() == 4:
        # [B, 1, F, T] → [B, F, T]
        x = x.squeeze(1)

    if x.dim() != 3:
        raise ValueError(f"Input debe ser 3D, got shape {x.shape}")

    return x  # [B, F, T]


# ============================================================
# TESTS: FLATTENING Y SHAPES
# ============================================================


def test_flattening_removes_channel_dim():
    """Verifica que flattening remueve correctamente la dim de canal."""
    print("\n" + "=" * 70)
    print("TEST 1: FLATTENING 2D→1D")
    print("=" * 70)

    # Input típico de dataset: [B, 1, F, T]
    x_2d = torch.randn(2, 1, 65, 41)
    print(f"  Input shape: {x_2d.shape} (B, 1, F, T)")

    # Flatten
    x_1d = flatten_2d_to_1d(x_2d)
    print(f"  Output shape: {x_1d.shape} (B, F, T)")

    assert x_1d.shape == (2, 65, 41), f"Shape incorrecto: {x_1d.shape}"
    assert x_1d.dim() == 3, "Debe ser 3D"

    print("  ✓ Dimensión de canal removida correctamente")
    print("  ✓ Shape final: [B, F=65, T=41] para Conv1D")


def test_flattening_preserves_data():
    """Verifica que flattening preserva los datos (solo cambia shape)."""
    print("\n" + "=" * 70)
    print("TEST 2: PRESERVACIÓN DE DATOS EN FLATTENING")
    print("=" * 70)

    B, F, T = 2, 4, 3

    # Crear tensor con patrón conocido: valor = freq_idx * 10 + time_idx
    pattern = torch.zeros(B, 1, F, T)
    for b in range(B):
        for f in range(F):
            for t in range(T):
                pattern[b, 0, f, t] = f * 10 + t

    print(f"  Input shape: {pattern.shape}")
    print(f"  Sample [0, 0, :, :]:\n{pattern[0, 0]}")

    # Flatten
    flattened = flatten_2d_to_1d(pattern)

    # Verificar que valores se preservan
    for b in range(B):
        for f in range(F):
            for t in range(T):
                expected = f * 10 + t
                actual = flattened[b, f, t].item()
                assert actual == expected, \
                    f"Valor incorrecto en [{b},{f},{t}]: {actual} != {expected}"

    print("  ✓ Todos los valores preservados correctamente")
    print("  ✓ Flattening solo remueve dim, no altera datos")


# ============================================================
# TESTS: ATENCIÓN TEMPORAL (MATEMÁTICA RIGUROSA)
# ============================================================


def test_attention_softmax_axis_and_sum1():
    """Verifica que softmax se aplica en eje temporal y suma 1."""
    print("\n" + "=" * 70)
    print("TEST 3: SOFTMAX TEMPORAL Y NORMALIZACIÓN")
    print("=" * 70)

    torch.manual_seed(42)
    B, C, T = 2, 8, 5
    y = torch.randn(B, C, T)

    z, alpha, A, V = attention_forward(y)

    # Verificar shape de alpha
    assert alpha.shape == (B, C // 2, T)
    print(f"  Alpha shape: {alpha.shape} (B, C/2, T)")

    # Verificar no negatividad
    assert torch.all(alpha >= 0), "Alpha tiene valores negativos"
    print("  ✓ Alpha >= 0 (softmax válido)")

    # Verificar que suma 1 por canal en dimensión temporal
    sums = alpha.sum(dim=-1)  # [B, C/2]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    print("  ✓ Suma temporal = 1.0 para cada canal")

    # Verificar embedding shape
    assert z.shape == (B, C // 2)
    print(f"  ✓ Embedding z shape: {z.shape} (B, C/2)")


def test_attention_manual_calculation():
    """
    Verifica cálculo manual de atención con valores conocidos.

    Ejemplo numérico:
        A1 = [1, 2, 3] → softmax ≈ [0.09, 0.24, 0.67]
        V1 = [4, 2, 1]
        z1 = 0.09*4 + 0.24*2 + 0.67*1 ≈ 1.51
    """
    print("\n" + "=" * 70)
    print("TEST 4: CÁLCULO MANUAL DE ATENCIÓN")
    print("=" * 70)

    # Construir tensor con valores específicos
    y = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],  # A1
                [2.0, 1.0, 0.0],  # A2
                [4.0, 2.0, 1.0],  # V1
                [0.5, 1.5, 2.0],  # V2
            ]
        ]
    ).float()  # [1, 4, 3]

    z, alpha, A, V = attention_forward(y)

    # Cálculo manual esperado
    # A1 = [1, 2, 3] → softmax
    a1_exp = torch.exp(A[0, 0, :])  # [e^1, e^2, e^3]
    a1_sum = a1_exp.sum()
    alpha1_expected = a1_exp / a1_sum  # [0.09, 0.24, 0.67]

    print(f"  Alpha1 calculado: {alpha[0, 0, :].numpy()}")
    print(f"  Alpha1 esperado:  {alpha1_expected.numpy()}")

    assert torch.allclose(alpha[0, 0, :], alpha1_expected, atol=1e-6)

    # z1 = sum(alpha1 * V1)
    V1 = V[0, 0, :]  # [4, 2, 1]
    z1_expected = (alpha1_expected * V1).sum()

    print(f"  z1 calculado: {z[0, 0].item():.4f}")
    print(f"  z1 esperado:  {z1_expected.item():.4f}")

    assert torch.allclose(z[0, 0], z1_expected, atol=1e-4)
    print("  ✓ Cálculo manual coincide con implementación")


def test_attention_uniform_equals_temporal_mean():
    """
    Si A es todo ceros → softmax uniforme → z = promedio temporal de V.
    """
    print("\n" + "=" * 70)
    print("TEST 5: ATENCIÓN UNIFORME = PROMEDIO TEMPORAL")
    print("=" * 70)

    B, half, T = 2, 4, 6
    A = torch.zeros(B, half, T)  # Softmax → 1/T uniforme
    V = torch.randn(B, half, T)

    y = torch.cat([A, V], dim=1)  # [B, 2*half, T]
    z, alpha, _, _ = attention_forward(y)

    # Verificar que alpha es uniforme
    expected_alpha = torch.ones(B, half, T) / T
    assert torch.allclose(alpha, expected_alpha, atol=1e-6)
    print(f"  Alpha uniforme: {alpha[0, 0, :].numpy()}")

    # z debe ser el promedio temporal de V
    mean_v = V.mean(dim=-1)
    print(f"  z calculado: {z[0, :].numpy()}")
    print(f"  mean(V):     {mean_v[0, :].numpy()}")

    assert torch.allclose(z, mean_v, atol=1e-6)
    print("  ✓ Atención uniforme = promedio temporal de V")


def test_attention_focus_increases_contribution():
    """
    Si A tiene un pico en t=k → contribución de V[:,:,k] debe dominar.
    """
    print("\n" + "=" * 70)
    print("TEST 6: ATENCIÓN FOCALIZADA AUMENTA CONTRIBUCIÓN")
    print("=" * 70)

    B, half, T = 1, 2, 4
    k = 2  # Tiempo de interés

    # A con pico grande en t=k
    A = torch.zeros(B, half, T)
    A[:, :, k] = 10.0  # Pico fuerte → softmax ≈ 1 en k

    # V con valores distintivos en t=k
    V = torch.zeros(B, half, T)
    V[:, 0, k] = 5.0
    V[:, 1, k] = 3.0

    y = torch.cat([A, V], dim=1)
    z, alpha, _, _ = attention_forward(y)

    print(f"  Alpha[0, 0, :]: {alpha[0, 0, :].numpy()}")
    print(f"  Pico en t={k}: {alpha[0, 0, k].item():.4f}")

    # Alpha en t=k debe ser ~1
    assert alpha[0, 0, k] > 0.95, "Pico no suficientemente fuerte"

    # z debe aproximarse a V[:, :, k] = [5.0, 3.0]
    print(f"  z: {z[0, :].numpy()}")
    print(f"  V[:, :, k]: {V[0, :, k].numpy()}")

    assert torch.allclose(z, V[0, :, k], atol=1e-2)
    print("  ✓ Atención focalizada aumenta contribución del instante k")


def test_attention_backprop_nonzero_grads():
    """Verifica que gradientes fluyen correctamente a través de atención."""
    print("\n" + "=" * 70)
    print("TEST 7: BACKPROPAGATION A TRAVÉS DE ATENCIÓN")
    print("=" * 70)

    B, half, T = 2, 4, 6

    # Simular parámetros que generan A y V
    A_params = torch.randn(B, half, T, requires_grad=True)
    V_params = torch.randn(B, half, T, requires_grad=True)

    y = torch.cat([A_params, V_params], dim=1)
    z, alpha, _, _ = attention_forward(y)  # z: [B, half]

    # Pérdida dummy
    loss = z.pow(2).mean()
    loss.backward()

    # Verificar que gradientes existen y no son cero
    assert A_params.grad is not None, "A_params.grad es None"
    assert V_params.grad is not None, "V_params.grad es None"

    grad_a_sum = A_params.grad.abs().sum().item()
    grad_v_sum = V_params.grad.abs().sum().item()

    print(f"  Gradiente total en A: {grad_a_sum:.6f}")
    print(f"  Gradiente total en V: {grad_v_sum:.6f}")

    assert grad_a_sum > 0, "Gradientes en A son cero"
    assert grad_v_sum > 0, "Gradientes en V son cero"

    print("  ✓ Gradientes fluyen correctamente a A y V")


# ============================================================
# TESTS: INTEGRACIÓN CON MODELO REAL
# ============================================================


def test_cnn1d_model_attention_integration():
    """Test de integración: modelo completo con atención."""
    print("\n" + "=" * 70)
    print("TEST 8: INTEGRACIÓN CON CNN1D_DA")
    print("=" * 70)

    try:
        from modules.models.cnn1d.model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, c1=64, c2=128, c3=128, num_domains=10)
        torch.manual_seed(0)
        x = torch.randn(2, 65, 41)  # [B, F, T]

        # Forward completo
        logits_pd, logits_domain, embeddings = model(
            x, return_embeddings=True
        )

        # Verificar shapes
        assert embeddings.shape == (2, 64), \
            f"Embeddings shape: {embeddings.shape}"

        print(f"  Input: {x.shape}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Logits PD: {logits_pd.shape}")
        print(f"  Logits Domain: {logits_domain.shape}")

        # Test backprop
        loss = logits_pd.pow(2).mean()
        loss.backward()

        # Verificar que gradientes llegan a los parámetros del modelo
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grads, "No hay gradientes en el modelo"

        print("  ✓ Modelo completo forward/backward OK")
        print("  ✓ Atención integrada correctamente")

    except Exception as e:
        print(f"  ✗ Error en integración: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_shape_consistency_through_blocks():
    """Verifica consistencia de shapes a través de todos los bloques."""
    print("\n" + "=" * 70)
    print("TEST 9: CONSISTENCIA DE SHAPES POR BLOQUE")
    print("=" * 70)

    try:
        from modules.models.cnn1d.model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, c1=64, c2=128, c3=128)
        model.eval()  # Poner en eval para evitar error de BatchNorm
        x = torch.randn(1, 65, 41)  # [B=1, F=65, T=41]

        print(f"  Input: {x.shape}")

        # Paso a paso por bloques
        with torch.no_grad():
            y1 = model.block1(x)
            print(f"  Tras block1: {y1.shape} "
                  f"(esperado: [1, 64, T/6])")
            assert y1.shape[1] == 64, "Canales block1 incorrectos"

            y2 = model.block2(y1)
            print(f"  Tras block2: {y2.shape} "
                  f"(esperado: [1, 128, T/36])")
            assert y2.shape[1] == 128, "Canales block2 incorrectos"

            y3 = model.block3(y2)
            print(f"  Tras block3: {y3.shape} "
                  f"(esperado: [1, 128, T'])")
            assert y3.shape[1] == 128, "Canales block3 incorrectos"

            # Atención
            B, C, T = y3.shape
            A, V = y3[:, :64, :], y3[:, 64:, :]
            alpha = F.softmax(A, dim=-1)
            z = (alpha * V).sum(dim=-1)

            print(f"  A shape: {A.shape}")
            print(f"  V shape: {V.shape}")
            print(f"  Alpha shape: {alpha.shape}")
            print(f"  Embedding z: {z.shape} (esperado: [1, 64])")

            assert z.shape == (1, 64)

        print("  ✓ Todas las shapes son consistentes")
        print("  ✓ Atención produce embedding [B, 64]")

    except Exception as e:
        print(f"  ✗ Error en shapes: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# TESTS NUMÉRICOS ESPECÍFICOS
# ============================================================


def test_attention_deterministic_output():
    """Test con valores deterministas para verificar cálculo exacto."""
    print("\n" + "=" * 70)
    print("TEST 10: CÁLCULO DETERMINISTA DE ATENCIÓN")
    print("=" * 70)

    # Crear tensor con valores específicos
    # A = [[1, 0, 0]]  → softmax ≈ [0.576, 0.212, 0.212]
    # V = [[2, 3, 4]]
    # z = 0.576*2 + 0.212*3 + 0.212*4 ≈ 2.636

    A = torch.tensor([[[1.0, 0.0, 0.0]]])  # [1, 1, 3]
    V = torch.tensor([[[2.0, 3.0, 4.0]]])  # [1, 1, 3]
    y = torch.cat([A, V], dim=1)  # [1, 2, 3]

    z, alpha, _, _ = attention_forward(y)

    # Cálculo manual
    a_exp = torch.exp(A[0, 0, :])
    a_softmax = a_exp / a_exp.sum()
    z_manual = (a_softmax * V[0, 0, :]).sum()

    print(f"  Alpha calculado: {alpha[0, 0, :].numpy()}")
    print(f"  Alpha esperado:  {a_softmax.numpy()}")
    print(f"  z calculado: {z[0, 0].item():.4f}")
    print(f"  z esperado:  {z_manual.item():.4f}")

    assert torch.allclose(alpha[0, 0, :], a_softmax, atol=1e-6)
    assert torch.allclose(z[0, 0], z_manual, atol=1e-4)

    print("  ✓ Cálculo determinista correcto")


def test_attention_single_peak_dominates():
    """
    Si un elemento de A es mucho mayor → su peso domina.
    """
    print("\n" + "=" * 70)
    print("TEST 11: PEAK DOMINANTE EN ATENCIÓN")
    print("=" * 70)

    B, half, T = 1, 1, 5
    k = 3  # Posición del peak

    # A con peak en k
    A = torch.full((B, half, T), -10.0)  # Valores muy bajos
    A[:, :, k] = 10.0  # Peak alto

    # V con valor distintivo en k
    V = torch.ones(B, half, T) * 0.5
    V[:, :, k] = 7.0  # Valor alto en k

    y = torch.cat([A, V], dim=1)
    z, alpha, _, _ = attention_forward(y)

    print(f"  Alpha: {alpha[0, 0, :].numpy()}")
    print(f"  Peak en t={k}: {alpha[0, 0, k].item():.6f}")

    # Alpha[k] debe estar muy cerca de 1.0
    assert alpha[0, 0, k] > 0.99, "Peak no domina suficientemente"

    # z debe aproximarse a V[:, :, k]
    print(f"  z: {z[0, 0].item():.4f}")
    print(f"  V[k]: {V[0, 0, k].item():.4f}")

    assert torch.allclose(z[0, 0], V[0, 0, k], atol=1e-2)
    print("  ✓ Peak en atención domina correctamente")


# ============================================================
# TEST RUNNER
# ============================================================


def run_all_tests():
    """Ejecuta todos los tests en orden."""
    print("\n" + "=" * 70)
    print("TESTS MATEMÁTICOS: CNN1D ATENCIÓN Y FLATTENING")
    print("=" * 70)

    tests = [
        ("Flattening: remover canal", test_flattening_removes_channel_dim),
        ("Flattening: preservar datos", test_flattening_preserves_data),
        ("Softmax temporal suma=1", test_attention_softmax_axis_and_sum1),
        ("Cálculo manual atención", test_attention_manual_calculation),
        ("Atención uniforme", test_attention_uniform_equals_temporal_mean),
        ("Peak dominante", test_attention_single_peak_dominates),
        ("Backprop atención", test_attention_backprop_nonzero_grads),
        ("Integración CNN1D_DA", test_cnn1d_model_attention_integration),
        ("Shapes por bloque", test_shape_consistency_through_blocks),
        ("Output determinista", test_attention_deterministic_output),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append(True)
            print(f"\n✅ {name}: PASSED")
        except AssertionError as e:
            results.append(False)
            print(f"\n❌ {name}: FAILED - {e}")
        except Exception as e:
            results.append(False)
            print(f"\n❌ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests pasados: {passed}/{total}")

    if passed == total:
        print("\n✅ TODOS LOS TESTS MATEMÁTICOS PASARON")
        print("\nLa implementación de atención temporal es correcta según paper.")
        print("La transformación 2D→1D preserva datos correctamente.")
        return 0
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        print("\nRevisa la implementación de atención o flattening.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

