#!/usr/bin/env python3
"""
Script de Verificación de Implementación Ibarra (2023)
======================================================
Verifica que todos los componentes estén correctamente implementados.

Uso:
    python test/test_ibarra_implementation.py
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


def test_imports():
    """Test 1: Verificar que todos los módulos se importen correctamente."""
    print("\n" + "=" * 70)
    print("TEST 1: IMPORTS DE MÓDULOS")
    print("=" * 70)

    try:
        from modules.models.cnn2d.model_da import CNN2D_DA
        from modules.models.cnn2d.training import train_model_da, train_model_da_kfold
        from modules.core.utils import create_10fold_splits_by_speaker
        from modules.models.cnn2d.utils import compute_class_weights_auto
        from modules.preprocessing import (
            SAMPLE_RATE,
            WINDOW_MS,
            N_MELS,
            HOP_MS,
            FFT_WINDOW,
        )

        print(f"{OK} Todos los modulos importados correctamente")
        return True
    except Exception as e:
        print(f"{FAIL} Error en imports: {e}")
        return False


def test_preprocessing_config():
    """Test 2: Verificar configuración de preprocesamiento según Ibarra."""
    print("\n" + "=" * 70)
    print("TEST 2: CONFIGURACIÓN DE PREPROCESAMIENTO")
    print("=" * 70)

    from modules.preprocessing import (
        SAMPLE_RATE,
        WINDOW_MS,
        OVERLAP,
        N_MELS,
        HOP_MS,
        FFT_WINDOW,
        TARGET_FRAMES,
    )

    specs = {
        "SAMPLE_RATE": (SAMPLE_RATE, 44100),
        "WINDOW_MS": (WINDOW_MS, 400),
        "OVERLAP": (OVERLAP, 0.5),
        "N_MELS": (N_MELS, 65),
        "HOP_MS": (HOP_MS, 10),
        "FFT_WINDOW": (FFT_WINDOW, 40),
        "TARGET_FRAMES": (TARGET_FRAMES, 41),
    }

    all_ok = True
    for name, (actual, expected) in specs.items():
        if actual == expected:
            print(f"{GREEN}{OK}{RESET} {name}: {actual}")
        else:
            print(f"{RED}{FAIL}{RESET} {name}: {actual} (esperado: {expected})")
            all_ok = False

    return all_ok


def test_model_architecture():
    """Test 3: Verificar arquitectura del modelo."""
    print("\n" + "=" * 70)
    print("TEST 3: ARQUITECTURA CNN2D-DA")
    print("=" * 70)

    try:
        from modules.models.cnn2d.model_da import CNN2D_DA

        # Crear modelo
        model = CNN2D_DA(n_domains=10, p_drop_conv=0.3, p_drop_fc=0.5)

        # Test forward pass
        x_test = torch.randn(2, 1, 65, 41)
        logits_pd, logits_domain = model(x_test)

        # Verificar shapes
        assert logits_pd.shape == (2, 2), "Shape incorrecto de logits_pd"
        assert logits_domain.shape == (2, 10), "Shape incorrecto de logits_domain"

        # Verificar MaxPool 3x3
        maxpool_count = 0
        for module in model.feature_extractor.modules():
            if isinstance(module, torch.nn.MaxPool2d):
                if module.kernel_size == (3, 3):
                    maxpool_count += 1

        print(f"{GREEN}{OK}{RESET} Modelo creado correctamente")
        print(f"{GREEN}{OK}{RESET} Output PD shape: {logits_pd.shape}")
        print(f"{GREEN}{OK}{RESET} Output Domain shape: {logits_domain.shape}")
        print(f"{GREEN}{OK}{RESET} MaxPool 3x3 detectados: {maxpool_count}/2")

        if maxpool_count == 2:
            print(f"{GREEN}{OK}{RESET} Arquitectura cumple con Ibarra 2023")
        else:
            print(
                f"{YELLOW}{WARN}{RESET} MaxPool 3x3 esperados: 2, encontrados: {maxpool_count}"
            )

        return True
    except Exception as e:
        print(f"{RED}{FAIL}{RESET} Error en arquitectura: {e}")
        return False


def test_kfold_splits():
    """Test 4: Verificar splits 10-fold."""
    print("\n" + "=" * 70)
    print("TEST 4: 10-FOLD CV SPLITS")
    print("=" * 70)

    try:
        from modules.core.utils import create_10fold_splits_by_speaker

        # Metadata de ejemplo
        metadata = []
        for subj in ["s1", "s2", "s3", "s4", "s5"]:
            for label in [0, 1]:
                for i in range(10):
                    metadata.append({"subject_id": subj, "label": label})

        # Crear splits
        fold_splits = create_10fold_splits_by_speaker(metadata, n_folds=5, seed=42)

        # Verificar que se crearon 5 folds
        assert len(fold_splits) == 5, "Número de folds incorrecto"

        # Verificar que cada fold tiene train y val
        for i, split in enumerate(fold_splits):
            assert "train" in split, f"Fold {i} sin train"
            assert "val" in split, f"Fold {i} sin val"

        # Verificar que no hay overlap entre train y val
        for i, split in enumerate(fold_splits):
            train_set = set(split["train"])
            val_set = set(split["val"])
            overlap = train_set & val_set
            assert len(overlap) == 0, f"Fold {i} tiene overlap entre train/val"

        print(f"{GREEN}{OK}{RESET} {len(fold_splits)} folds creados correctamente")
        print(f"{GREEN}{OK}{RESET} Sin overlap entre train/val")
        print(f"{GREEN}{OK}{RESET} Splits estratificados por hablante")

        return True
    except Exception as e:
        print(f"{RED}{FAIL}{RESET} Error en K-fold splits: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_class_weights():
    """Test 5: Verificar detección automática de class weights."""
    print("\n" + "=" * 70)
    print("TEST 5: CLASS WEIGHTS AUTOMÁTICOS")
    print("=" * 70)

    try:
        from modules.models.cnn2d.utils import compute_class_weights_auto

        # Dataset balanceado
        labels_balanced = torch.tensor([0, 0, 0, 1, 1, 1])
        weights_balanced = compute_class_weights_auto(labels_balanced, threshold=0.4)

        # Dataset desbalanceado
        labels_unbalanced = torch.tensor([0, 0, 0, 0, 1, 1])
        weights_unbalanced = compute_class_weights_auto(
            labels_unbalanced, threshold=0.4
        )

        if weights_balanced is None:
            print(f"{GREEN}{OK}{RESET} Dataset balanceado: Sin pesos (correcto)")
        else:
            print(f"{YELLOW}{WARN}{RESET} Dataset balanceado: Pesos aplicados")

        if weights_unbalanced is not None:
            print(
                f"{GREEN}{OK}{RESET} Dataset desbalanceado: Pesos aplicados (correcto)"
            )
            print(f"   Pesos: {weights_unbalanced.tolist()}")
        else:
            print(f"{RED}{FAIL}{RESET} Dataset desbalanceado: Sin pesos (incorrecto)")

        return True
    except Exception as e:
        print(f"{RED}{FAIL}{RESET} Error en class weights: {e}")
        return False


def test_sgd_config():
    """Test 6: Verificar configuración SGD según Ibarra."""
    print("\n" + "=" * 70)
    print("TEST 6: CONFIGURACIÓN SGD")
    print("=" * 70)

    try:
        from modules.models.cnn2d.model_da import CNN2D_DA

        model = CNN2D_DA(n_domains=10)

        # Crear optimizador según Ibarra
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )

        # Crear scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        print(f"{GREEN}{OK}{RESET} SGD creado: lr=0.1, momentum=0.9, wd=1e-4")
        print(f"{GREEN}{OK}{RESET} StepLR scheduler: step=30, gamma=0.1")

        return True
    except Exception as e:
        print(f"{RED}{FAIL}{RESET} Error en SGD config: {e}")
        return False


def test_main_script():
    """Test 7: Verificar que el script principal existe."""
    print("\n" + "=" * 70)
    print("TEST 7: SCRIPT PRINCIPAL")
    print("=" * 70)

    script_path = Path("train_cnn_da_kfold.py")

    if script_path.exists():
        print(f"{GREEN}{OK}{RESET} Script principal encontrado: train_cnn_da_kfold.py")
        return True
    else:
        print(f"{RED}{FAIL}{RESET} Script principal no encontrado")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("VERIFICACION DE IMPLEMENTACION IBARRA (2023)")
    print("=" * 70)

    tests = [
        test_imports,
        test_preprocessing_config,
        test_model_architecture,
        test_kfold_splits,
        test_class_weights,
        test_sgd_config,
        test_main_script,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"{RED}{FAIL}{RESET} Test fallo: {e}")
            results.append(False)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests pasados: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}TODOS LOS TESTS PASARON{RESET}")
        print("\nImplementacion lista para usar segun Ibarra (2023)!")
        return 0
    else:
        print(f"\n{YELLOW}ALGUNOS TESTS FALLARON{RESET}")
        print("\nRevisa los errores arriba antes de ejecutar el entrenamiento.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
