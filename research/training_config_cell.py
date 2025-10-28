# ============================================================
# CONFIGURAR ENTRENAMIENTO CON HIPERPARÁMETROS SELECCIONADOS
# ============================================================

print("=" * 70)
print("CONFIGURANDO ENTRENAMIENTO CON HIPERPARÁMETROS SELECCIONADOS")
print("=" * 70)

# Verificar que las variables estén definidas
if "BEST_PARAMS" not in globals():
    print("❌ Error: BEST_PARAMS no está definido.")
    print("   Ejecuta primero la celda del selector de hiperparámetros.")
    print("=" * 70)
else:
    print(f"✅ Configurando entrenamiento con parámetros de: {HYPERPARAMETER_SOURCE}")

    # Configuración de entrenamiento usando los parámetros seleccionados
    FINAL_TRAINING_CONFIG = {
        "n_epochs": BEST_PARAMS["n_epochs"],
        "early_stopping_patience": BEST_PARAMS["early_stopping_patience"],
        "learning_rate": BEST_PARAMS["learning_rate"],
        "batch_size": BEST_PARAMS["batch_size"],
    }

    # Crear DataLoaders con el batch size seleccionado
    train_loader_final = DataLoader(
        train_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=True,
        num_workers=0,
    )
    val_loader_final = DataLoader(
        val_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=False,
        num_workers=0,
    )
    test_loader_final = DataLoader(
        test_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=False,
        num_workers=0,
    )

    # Optimizador SGD con momentum usando los parámetros seleccionados
    optimizer_final = optim.SGD(
        best_model.parameters(),
        lr=FINAL_TRAINING_CONFIG["learning_rate"],
        momentum=BEST_PARAMS["momentum"],
        weight_decay=BEST_PARAMS["weight_decay"],
        nesterov=True,  # Mejora sobre Ibarra
    )

    # Calcular class weights para balancear las clases
    if CLASS_WEIGHTS_CONFIG["enabled"]:
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        criterion_final = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"✅ Class weights habilitados: {class_weights.tolist()}")
    else:
        criterion_final = nn.CrossEntropyLoss()
        print("⚠️  Class weights deshabilitados")

    # Scheduler StepLR usando los parámetros seleccionados
    scheduler_final = torch.optim.lr_scheduler.StepLR(
        optimizer_final, step_size=BEST_PARAMS["step_size"], gamma=BEST_PARAMS["gamma"]
    )

    print(f"\n⚙️  Configuración final ({HYPERPARAMETER_SOURCE}):")
    print(f"   - Learning rate inicial: {FINAL_TRAINING_CONFIG['learning_rate']}")
    print(f"   - Momentum: {BEST_PARAMS['momentum']} (Nesterov: True)")
    print(f"   - Weight decay: {BEST_PARAMS['weight_decay']}")
    print(
        f"   - Scheduler: StepLR (step={BEST_PARAMS['step_size']}, gamma={BEST_PARAMS['gamma']})"
    )
    print(f"   - Batch size: {FINAL_TRAINING_CONFIG['batch_size']}")
    print(f"   - Épocas máximas: {FINAL_TRAINING_CONFIG['n_epochs']}")
    print(
        f"   - Early stopping patience: {FINAL_TRAINING_CONFIG['early_stopping_patience']}"
    )

    # Mostrar diferencias con Ibarra si estamos usando Optuna
    if not USE_IBARRA:
        print(f"\n📊 DIFERENCIAS CON IBARRA:")
        print(f"   - Ibarra batch_size: 64 → Optuna: {BEST_PARAMS['batch_size']}")
        print(f"   - Ibarra kernel_1: 6 → Optuna: {BEST_PARAMS['kernel_size_1']}")
        print(f"   - Ibarra filters_2: 64 → Optuna: {BEST_PARAMS['filters_2']}")
        print(f"   - Ibarra dense_units: 32 → Optuna: {BEST_PARAMS['dense_units']}")

    print("=" * 70)
