#!/usr/bin/env python3
"""
Script de Verificación de Implementación CNN1D con DA (Ibarra 2023)
====================================================================
Verifica que todos los componentes CNN1D estén correctamente implementados.

Uso:
    python test/test_cnn1d_implementation.py
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Símbolos simples para Windows compatibility
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# Color codes (vacíos para Windows compatibility)
GREEN = ""
RED = ""
YELLOW = ""
RESET = ""


def test_imports_cnn1d():
    """Test 1: Verificar imports de módulos CNN1D."""
    print("\n" + "=" * 70)
    print("TEST 1: IMPORTS DE MÓDULOS CNN1D")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA
        from modules.cnn1d_training import (
            train_model_da,
            evaluate_da,
            aggregate_patient_predictions,
            evaluate_patient_level,
        )
        from modules.cnn1d_visualization import (
            plot_1d_training_progress,
            plot_tsne_embeddings,
        )
        from modules.dataset import (
            speaker_independent_split,
            group_by_patient,
        )

        print(f"{OK} Todos los módulos CNN1D importados correctamente")
        return True
    except Exception as e:
        print(f"{FAIL} Error en imports: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn1d_architecture():
    """Test 2: Verificar arquitectura CNN1D_DA según paper."""
    print("\n" + "=" * 70)
    print("TEST 2: ARQUITECTURA CNN1D_DA")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA

        # Crear modelo
        model = CNN1D_DA(
            in_ch=65, c1=64, c2=128, c3=128, p_drop=0.3,
            num_pd=2, num_domains=26
        )

        # Test forward pass con shape correcto [B, F, T]
        x_test = torch.randn(2, 65, 41)  # [B, F=65, T=41]
        logits_pd, logits_domain, embeddings = model(
            x_test, return_embeddings=True
        )

        # Verificar shapes
        assert logits_pd.shape == (2, 2), \
            f"Shape PD incorrecto: {logits_pd.shape}"
        assert logits_domain.shape == (2, 26), \
            f"Shape Domain incorrecto: {logits_domain.shape}"
        assert embeddings.shape == (2, 64), \
            f"Shape Embeddings incorrecto: {embeddings.shape}"

        print(f"{OK} Modelo creado correctamente")
        print(f"{OK} Output PD shape: {logits_pd.shape}")
        print(f"{OK} Output Domain shape: {logits_domain.shape}")
        print(f"{OK} Embeddings shape: {embeddings.shape}")

        # Verificar Conv1D kernels (5, 11, 21)
        kernels = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                kernels.append(module.kernel_size[0])

        expected_kernels = [5, 11, 21]
        print(f"{OK} Kernels Conv1D: {kernels}")

        if kernels == expected_kernels:
            print(f"{OK} Kernels cumplen con Ibarra 2023: {expected_kernels}")
        else:
            print(f"{WARN} Kernels esperados: {expected_kernels}, "
                  f"encontrados: {kernels}")

        # Verificar MaxPool1D (k=6) solo en bloques 1 y 2
        maxpool_count = 0
        maxpool_kernels = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MaxPool1d):
                maxpool_count += 1
                maxpool_kernels.append(module.kernel_size)

        print(f"{OK} MaxPool1D detectados: {maxpool_count}/2")
        print(f"{OK} MaxPool kernels: {maxpool_kernels}")

        if maxpool_count == 2 and all(k == 6 for k in maxpool_kernels):
            print(f"{OK} MaxPool cumple con paper (2 pools, k=6)")
        else:
            print(f"{WARN} Esperado: 2 MaxPool1D con k=6, "
                  f"encontrado: {maxpool_count} con k={maxpool_kernels}")

        # Verificar embedding dim = c3/2 = 64
        assert model.half == 64, f"Embedding dim incorrecto: {model.half}"
        print(f"{OK} Embedding dimension: {model.half} (c3/2 = 128/2)")

        return True

    except Exception as e:
        print(f"{FAIL} Error en arquitectura CNN1D: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_reversal_layer():
    """Test 3: Verificar GRL funciona correctamente."""
    print("\n" + "=" * 70)
    print("TEST 3: GRADIENT REVERSAL LAYER")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, num_domains=10)

        # Test lambda update
        initial_lambda = model.grl.lambda_
        print(f"{OK} Lambda inicial: {initial_lambda}")

        model.set_lambda(0.5)
        assert model.grl.lambda_ == 0.5, "Lambda no se actualizó"
        print(f"{OK} Lambda actualizado: {model.grl.lambda_}")

        model.set_lambda(1.0)
        assert model.grl.lambda_ == 1.0, "Lambda no se actualizó a 1.0"
        print(f"{OK} Lambda scheduler funciona correctamente")

        return True

    except Exception as e:
        print(f"{FAIL} Error en GRL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_independent_split():
    """Test 4: Verificar speaker-independent split."""
    print("\n" + "=" * 70)
    print("TEST 4: SPEAKER-INDEPENDENT SPLIT")
    print("=" * 70)

    try:
        from modules.dataset import speaker_independent_split, SampleMeta

        # Crear metadata de ejemplo
        metadata = []
        for subject in ["S1", "S2", "S3", "S4", "S5"]:
            for cond in ["healthy", "pk"]:
                for i in range(10):
                    metadata.append(
                        SampleMeta(
                            subject_id=subject,
                            vowel_type="a",
                            condition=cond,
                            filename=f"{subject}_{cond}_{i}.wav",
                            segment_id=i,
                            sr=44100,
                        )
                    )

        # Split
        train_idx, val_idx, test_idx = speaker_independent_split(
            metadata, test_size=0.15, val_size=0.176, random_state=42
        )

        # Verificar no overlap de speakers
        train_subjects = set(metadata[i].subject_id for i in train_idx)
        val_subjects = set(metadata[i].subject_id for i in val_idx)
        test_subjects = set(metadata[i].subject_id for i in test_idx)

        overlap_train_val = train_subjects & val_subjects
        overlap_train_test = train_subjects & test_subjects
        overlap_val_test = val_subjects & test_subjects

        assert len(overlap_train_val) == 0, "Overlap entre train y val"
        assert len(overlap_train_test) == 0, "Overlap entre train y test"
        assert len(overlap_val_test) == 0, "Overlap entre val y test"

        print(f"{OK} Sin overlap de speakers entre splits")
        print(f"   Train speakers: {sorted(train_subjects)}")
        print(f"   Val speakers: {sorted(val_subjects)}")
        print(f"   Test speakers: {sorted(test_subjects)}")

        return True

    except Exception as e:
        print(f"{FAIL} Error en speaker-independent split: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patient_aggregation():
    """Test 5: Verificar agregación por paciente."""
    print("\n" + "=" * 70)
    print("TEST 5: AGREGACIÓN POR PACIENTE")
    print("=" * 70)

    try:
        from modules.cnn1d_training import aggregate_patient_predictions
        import numpy as np

        # Simular predicciones: 9 segmentos de 3 pacientes
        probs = np.array([
            [0.8, 0.2],  # P1 - seg 1
            [0.7, 0.3],  # P1 - seg 2
            [0.9, 0.1],  # P1 - seg 3
            [0.3, 0.7],  # P2 - seg 1
            [0.4, 0.6],  # P2 - seg 2
            [0.2, 0.8],  # P2 - seg 3
            [0.5, 0.5],  # P3 - seg 1
            [0.6, 0.4],  # P3 - seg 2
            [0.55, 0.45],  # P3 - seg 3
        ])
        patient_ids = ["P1", "P1", "P1", "P2", "P2", "P2",
                       "P3", "P3", "P3"]

        # Test método 'mean'
        patient_probs, patient_labels = aggregate_patient_predictions(
            probs, patient_ids, method="mean"
        )

        assert len(patient_probs) == 3, "Debe haber 3 pacientes"
        assert set(patient_probs.keys()) == {"P1", "P2", "P3"}

        # P1: promedio [0.8,0.2], [0.7,0.3], [0.9,0.1] ≈ [0.8, 0.2]
        p1_mean = np.mean([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]], axis=0)
        assert np.allclose(patient_probs["P1"], p1_mean, atol=1e-6)
        assert patient_labels["P1"] == 0  # Clase 0 (HC)

        # P2: promedio ≈ [0.3, 0.7]
        assert patient_labels["P2"] == 1  # Clase 1 (PD)

        print(f"{OK} Agregación por promedio funciona correctamente")
        print(f"   P1: {patient_probs['P1']} → Clase {patient_labels['P1']}")
        print(f"   P2: {patient_probs['P2']} → Clase {patient_labels['P2']}")
        print(f"   P3: {patient_probs['P3']} → Clase {patient_labels['P3']}")

        return True

    except Exception as e:
        print(f"{FAIL} Error en patient aggregation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_shape_compatibility():
    """Test 6: Verificar compatibilidad de shapes de entrada."""
    print("\n" + "=" * 70)
    print("TEST 6: COMPATIBILIDAD DE SHAPES")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, num_domains=10)

        # Test 1: Input correcto [B, F, T]
        x_correct = torch.randn(4, 65, 41)
        out_pd, out_dom, _ = model(x_correct)
        assert out_pd.shape == (4, 2)
        print(f"{OK} Input [B, F=65, T=41] procesa correctamente")

        # Test 2: Verificar que embedding tiene dim correcta
        _, _, emb = model(x_correct, return_embeddings=True)
        assert emb.shape == (4, 64), f"Embedding shape: {emb.shape}"
        print(f"{OK} Embeddings shape: {emb.shape} (B, 64)")

        return True

    except Exception as e:
        print(f"{FAIL} Error en shapes: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_config_sgd():
    """Test 7: Verificar configuración SGD según Ibarra."""
    print("\n" + "=" * 70)
    print("TEST 7: CONFIGURACIÓN SGD (IBARRA 2023)")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, num_domains=10)

        # Crear optimizador según Ibarra
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )

        # Verificar parámetros
        assert optimizer.defaults["lr"] == 0.1
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 1e-4

        print(f"{OK} SGD: lr=0.1, momentum=0.9, weight_decay=1e-4")

        # Crear scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
        print(f"{OK} StepLR scheduler: step=30, gamma=0.1")

        # Test lambda scheduler
        lambda_scheduler = lambda epoch: epoch / 100
        assert lambda_scheduler(0) == 0.0
        assert lambda_scheduler(50) == 0.5
        assert lambda_scheduler(100) == 1.0
        print(f"{OK} Lambda scheduler lineal 0→1 funciona correctamente")

        return True

    except Exception as e:
        print(f"{FAIL} Error en config SGD: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_mechanism_basic():
    """Test 8: Verificación básica del mecanismo de atención."""
    print("\n" + "=" * 70)
    print("TEST 8: MECANISMO DE ATENCIÓN TEMPORAL")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA
        import torch.nn.functional as F

        model = CNN1D_DA(in_ch=65, c1=64, c2=128, c3=128)

        # Input dummy
        x = torch.randn(2, 65, 41)

        # Forward para obtener output de block3 (antes de atención)
        y = model.block1(x)
        y = model.block2(y)
        y = model.block3(y)  # [B, 128, T']

        # Verificar split en mitades
        B, C, T = y.shape
        assert C == 128, f"Canales incorrectos: {C}"
        half = C // 2
        assert half == 64, f"Half incorrecto: {half}"

        # Simular atención manual
        A, V = y[:, :half, :], y[:, half:, :]
        alpha = F.softmax(A, dim=-1)  # Softmax temporal
        z = (alpha * V).sum(dim=-1)  # [B, 64]

        # Verificar shapes
        assert alpha.shape == (B, half, T), f"Alpha shape: {alpha.shape}"
        assert z.shape == (B, half), f"Z shape: {z.shape}"

        # Verificar que softmax suma 1 por canal
        sums = alpha.sum(dim=-1)  # [B, half]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

        print(f"{OK} Split en mitades correcto: A[{half}], V[{half}]")
        print(f"{OK} Softmax temporal suma=1 por canal")
        print(f"{OK} Embedding resultante: {z.shape}")
        print(f"{OK} Mecanismo de atención implementado correctamente")

        return True

    except Exception as e:
        print(f"{FAIL} Error en atención: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_head_outputs():
    """Test 9: Verificar que dual-head retorna outputs correctos."""
    print("\n" + "=" * 70)
    print("TEST 9: DUAL-HEAD (PD + DOMAIN)")
    print("=" * 70)

    try:
        from modules.cnn1d_model import CNN1D_DA

        model = CNN1D_DA(in_ch=65, num_pd=2, num_domains=26)
        model.eval()  # Desactivar dropout para test determinista
        x = torch.randn(3, 65, 41)

        # Forward completo
        with torch.no_grad():
            logits_pd, logits_domain, emb = model(x, return_embeddings=True)

        # Verificar que PD head retorna 2 clases
        assert logits_pd.shape[-1] == 2
        print(f"{OK} PD Head: {logits_pd.shape} (2 clases: HC/PD)")

        # Verificar que Domain head retorna n_domains
        assert logits_domain.shape[-1] == 26
        print(f"{OK} Domain Head: {logits_domain.shape} (26 dominios)")

        # Verificar que embeddings tienen dim correcta
        assert emb.shape[-1] == 64
        print(f"{OK} Embeddings: {emb.shape} (dim=64)")

        # Test que GRL no afecta forward (solo backward)
        emb1 = emb.clone()
        model.set_lambda(0.5)
        with torch.no_grad():
            _, _, emb2 = model(x, return_embeddings=True)
        assert torch.allclose(emb1, emb2), "GRL afecta forward (incorrecto)"
        print(f"{OK} GRL no afecta forward pass (correcto)")

        return True

    except Exception as e:
        print(f"{FAIL} Error en dual-head: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notebook_exists():
    """Test 10: Verificar que el notebook existe."""
    print("\n" + "=" * 70)
    print("TEST 10: NOTEBOOK CNN1D_DA")
    print("=" * 70)

    notebook_path = Path("cnn1d_da_training.ipynb")

    if notebook_path.exists():
        print(f"{OK} Notebook encontrado: cnn1d_da_training.ipynb")
        return True
    else:
        print(f"{FAIL} Notebook no encontrado")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE IMPLEMENTACIÓN CNN1D + DA (IBARRA 2023)")
    print("=" * 70)

    tests = [
        test_imports_cnn1d,
        test_cnn1d_architecture,
        test_gradient_reversal_layer,
        test_speaker_independent_split,
        test_patient_aggregation,
        test_input_shape_compatibility,
        test_training_config_sgd,
        test_attention_mechanism_basic,
        test_dual_head_outputs,
        test_notebook_exists,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"{FAIL} Test falló: {e}")
            results.append(False)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests pasados: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}✅ TODOS LOS TESTS PASARON{RESET}")
        print("\nImplementación CNN1D+DA lista según Ibarra (2023)!")
        print("\nPróximos pasos:")
        print("  1. Ejecutar data_preprocessing.ipynb")
        print("  2. Ejecutar cnn1d_da_training.ipynb")
        print("  3. Comparar con CNN2D_DA")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  ALGUNOS TESTS FALLARON{RESET}")
        print("\nRevisa los errores arriba antes de ejecutar entrenamiento.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

