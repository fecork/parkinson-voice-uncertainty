# ============================================================
# CELDA PARA REEMPLAZAR LA CREACIÓN DEL MODELO EN EL NOTEBOOK
# ============================================================

"""
INSTRUCCIONES:
1. Busca la celda que crea el modelo CNN2D en tu notebook
2. Reemplaza TODO el contenido de esa celda con este código
3. Ejecuta la celda
"""

# ============================================================
# CREAR MODELO CON HIPERPARÁMETROS SELECCIONADOS
# ============================================================

print("=" * 70)
print("CREANDO MODELO CON HIPERPARÁMETROS SELECCIONADOS")
print("=" * 70)

# Verificar que las variables estén definidas
if "BEST_PARAMS" not in globals():
    print("❌ Error: BEST_PARAMS no está definido.")
    print("   Ejecuta primero la celda del selector de hiperparámetros.")
    print("=" * 70)
else:
    print(f"✅ Usando hiperparámetros de: {HYPERPARAMETER_SOURCE}")

    # Crear modelo con los parámetros seleccionados
    best_model = CNN2D(
        n_classes=2,
        p_drop_conv=BEST_PARAMS["p_drop_conv"],
        p_drop_fc=BEST_PARAMS["p_drop_fc"],
        input_shape=(65, 41),
        filters_1=BEST_PARAMS["filters_1"],
        filters_2=BEST_PARAMS["filters_2"],
        kernel_size_1=BEST_PARAMS["kernel_size_1"],
        kernel_size_2=BEST_PARAMS["kernel_size_2"],
        dense_units=BEST_PARAMS["dense_units"],
    ).to(device)

    print(f"✅ Modelo creado con hiperparámetros de {HYPERPARAMETER_SOURCE}:")
    print(f"   - Filters 1: {BEST_PARAMS['filters_1']}")
    print(f"   - Filters 2: {BEST_PARAMS['filters_2']}")
    print(f"   - Kernel 1: {BEST_PARAMS['kernel_size_1']}")
    print(f"   - Kernel 2: {BEST_PARAMS['kernel_size_2']}")
    print(f"   - Dense units: {BEST_PARAMS['dense_units']}")
    print(f"   - Dropout conv: {BEST_PARAMS['p_drop_conv']}")
    print(f"   - Dropout fc: {BEST_PARAMS['p_drop_fc']}")

    # Mostrar arquitectura
    print_model_summary(best_model)

    print("=" * 70)
