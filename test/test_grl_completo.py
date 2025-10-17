#!/usr/bin/env python3
"""
Test Completo del Gradient Reversal Layer
==========================================
Pruebas matemáticas exhaustivas para verificar el GRL.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from modules.cnn_model import CNN2D_DA, GradientReversalLayer


def test_1_forward_identity():
    """Test 1: Forward pass debe ser identidad (no modifica los valores)."""
    print("\n" + "=" * 70)
    print("TEST 1: FORWARD PASS ES IDENTIDAD")
    print("=" * 70)
    
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(4, 64, 17, 11)
    y = grl(x)
    
    # Verificar que forward no modifica nada
    if torch.allclose(x, y, atol=1e-8):
        print("[OK] Forward pass es identidad: f(x) = x")
        print(f"   Max diferencia: {(x - y).abs().max().item():.10f}")
        return True
    else:
        print("[ERROR] Forward pass modifica los valores")
        return False


def test_2_backward_inversion():
    """Test 2: Backward debe invertir exactamente (lambda=1.0)."""
    print("\n" + "=" * 70)
    print("TEST 2: BACKWARD INVIERTE GRADIENTES (lambda=1.0)")
    print("=" * 70)
    
    x = torch.randn(4, 64, 17, 11, requires_grad=True)
    
    # Sin GRL
    y1 = x.sum()
    y1.backward()
    grad_normal = x.grad.clone()
    
    # Con GRL
    x.grad.zero_()
    grl = GradientReversalLayer(lambda_=1.0)
    y2 = grl(x).sum()
    y2.backward()
    grad_reversed = x.grad.clone()
    
    # Verificar: grad_reversed = -grad_normal
    if torch.allclose(grad_reversed, -grad_normal, atol=1e-6):
        print("[OK] Inversion exacta: grad_out = -grad_in")
        print(f"   Max error: {(grad_reversed + grad_normal).abs().max().item():.10f}")
        return True
    else:
        print("[ERROR] Inversion incorrecta")
        return False


def test_3_lambda_scaling():
    """Test 3: Verificar escalado para diferentes valores de lambda."""
    print("\n" + "=" * 70)
    print("TEST 3: ESCALADO CON DIFERENTES LAMBDAS")
    print("=" * 70)
    
    x = torch.randn(4, 64, 17, 11, requires_grad=True)
    grl = GradientReversalLayer()
    
    # Referencia con lambda=1.0
    grl.set_lambda(1.0)
    y_ref = grl(x).sum()
    y_ref.backward()
    grad_ref = x.grad.clone()
    
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_passed = True
    
    for lam in lambdas:
        x.grad.zero_()
        grl.set_lambda(lam)
        y = grl(x).sum()
        y.backward()
        grad = x.grad.clone()
        
        # Verificar: grad = lam * grad_ref
        expected = lam * grad_ref
        error = (grad - expected).abs().max().item()
        
        if torch.allclose(grad, expected, atol=1e-6):
            print(f"[OK] lambda={lam:.2f}: grad = {lam:.2f} * grad_ref (error: {error:.2e})")
        else:
            print(f"[ERROR] lambda={lam:.2f}: escalado incorrecto (error: {error:.2e})")
            all_passed = False
    
    return all_passed


def test_4_chain_rule():
    """Test 4: Verificar que la regla de la cadena funciona correctamente."""
    print("\n" + "=" * 70)
    print("TEST 4: REGLA DE LA CADENA (GRL + operaciones)")
    print("=" * 70)
    
    x = torch.randn(4, 64, requires_grad=True)
    grl = GradientReversalLayer(lambda_=1.0)
    
    # Operaciones después del GRL
    y = grl(x)
    z = y * 2.0 + 3.0  # Operaciones lineales
    loss = z.sum()
    
    loss.backward()
    grad_with_ops = x.grad.clone()
    
    # Sin GRL (para comparar)
    x.grad.zero_()
    y2 = x * 2.0 + 3.0
    loss2 = y2.sum()
    loss2.backward()
    grad_normal = x.grad.clone()
    
    # Con GRL: grad debe ser -1 * grad_normal
    if torch.allclose(grad_with_ops, -grad_normal, atol=1e-6):
        print("[OK] Regla de la cadena funciona correctamente")
        print(f"   Grad normal: {grad_normal.mean().item():.6f}")
        print(f"   Grad con GRL: {grad_with_ops.mean().item():.6f}")
        return True
    else:
        print("[ERROR] Regla de la cadena incorrecta")
        return False


def test_5_multiple_backward_passes():
    """Test 5: Verificar que funciona en múltiples backward passes."""
    print("\n" + "=" * 70)
    print("TEST 5: MULTIPLES BACKWARD PASSES")
    print("=" * 70)
    
    grl = GradientReversalLayer(lambda_=1.0)
    all_passed = True
    
    for i in range(5):
        x = torch.randn(2, 32, requires_grad=True)
        y = grl(x).sum()
        y.backward()
        
        # Verificar que el gradiente es -1 (inverso de 1)
        expected = torch.ones_like(x) * -1.0
        
        if torch.allclose(x.grad, expected, atol=1e-6):
            print(f"[OK] Backward pass {i+1}: correcto")
        else:
            print(f"[ERROR] Backward pass {i+1}: incorrecto")
            all_passed = False
    
    return all_passed


def test_6_integration_with_loss():
    """Test 6: Verificar integración con una pérdida real."""
    print("\n" + "=" * 70)
    print("TEST 6: INTEGRACION CON PERDIDA REAL")
    print("=" * 70)
    
    # Crear un modelo simple con GRL
    model = CNN2D_DA(n_domains=10)
    criterion_pd = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    x = torch.randn(4, 1, 65, 41)
    labels_pd = torch.tensor([0, 1, 0, 1])
    labels_domain = torch.tensor([0, 1, 2, 3])
    
    # Forward
    logits_pd, logits_domain = model(x)
    loss_pd = criterion_pd(logits_pd, labels_pd)
    loss_domain = criterion_domain(logits_domain, labels_domain)
    loss = loss_pd + loss_domain
    
    # Backward
    loss.backward()
    
    # Verificar que hay gradientes en feature extractor
    has_grads = False
    for param in model.feature_extractor.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break
    
    if has_grads:
        print("[OK] Gradientes fluyen correctamente con perdida real")
        print(f"   Loss PD: {loss_pd.item():.4f}")
        print(f"   Loss Domain: {loss_domain.item():.4f}")
        return True
    else:
        print("[ERROR] No hay gradientes en feature extractor")
        return False


def test_7_gradient_magnitude():
    """Test 7: Verificar que la magnitud del gradiente se mantiene."""
    print("\n" + "=" * 70)
    print("TEST 7: MAGNITUD DEL GRADIENTE")
    print("=" * 70)
    
    x = torch.randn(4, 64, 17, 11, requires_grad=True)
    
    # Sin GRL
    y1 = x.sum()
    y1.backward()
    mag_normal = x.grad.norm().item()
    
    # Con GRL (lambda=1.0)
    x.grad.zero_()
    grl = GradientReversalLayer(lambda_=1.0)
    y2 = grl(x).sum()
    y2.backward()
    mag_reversed = x.grad.norm().item()
    
    # La magnitud debe ser la misma (solo cambia el signo)
    if abs(mag_normal - mag_reversed) < 1e-5:
        print("[OK] Magnitud se conserva")
        print(f"   Magnitud sin GRL: {mag_normal:.6f}")
        print(f"   Magnitud con GRL: {mag_reversed:.6f}")
        return True
    else:
        print("[ERROR] Magnitud no se conserva")
        return False


def test_8_lambda_zero():
    """Test 8: Verificar comportamiento con lambda=0 (sin inversión)."""
    print("\n" + "=" * 70)
    print("TEST 8: LAMBDA=0 (SIN INVERSION)")
    print("=" * 70)
    
    x = torch.randn(4, 64, requires_grad=True)
    grl = GradientReversalLayer(lambda_=0.0)
    
    y = grl(x).sum()
    y.backward()
    
    # Con lambda=0, el gradiente debe ser 0
    if torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-6):
        print("[OK] Lambda=0: gradiente es cero (sin propagacion)")
        print(f"   Max grad: {x.grad.abs().max().item():.10f}")
        return True
    else:
        print("[ERROR] Lambda=0: gradiente no es cero")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("TESTS MATEMATICOS COMPLETOS DEL GRL")
    print("=" * 70)
    
    tests = [
        ("Forward es identidad", test_1_forward_identity),
        ("Backward invierte gradientes", test_2_backward_inversion),
        ("Escalado con lambda", test_3_lambda_scaling),
        ("Regla de la cadena", test_4_chain_rule),
        ("Multiples backward passes", test_5_multiple_backward_passes),
        ("Integracion con perdida real", test_6_integration_with_loss),
        ("Magnitud del gradiente", test_7_gradient_magnitude),
        ("Lambda=0 sin inversion", test_8_lambda_zero),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] Test '{name}' fallo con excepcion: {e}")
            results.append((name, False))
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTests pasados: {passed}/{total}")
    
    if passed == total:
        print("\n[OK] TODOS LOS TESTS MATEMATICOS PASARON")
        print("El GRL esta funcionando correctamente desde el punto de vista matematico.")
        return 0
    else:
        print("\n[FAIL] ALGUNOS TESTS FALLARON")
        return 1


if __name__ == "__main__":
    sys.exit(main())

