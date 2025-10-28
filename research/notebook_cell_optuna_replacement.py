# ============================================================
# CELDA PARA REEMPLAZAR LA SECCIÓN DE OPTUNA EN EL NOTEBOOK
# ============================================================

"""
INSTRUCCIONES:
1. Busca la celda que dice "OPTUNA OPTIMIZATION" en tu notebook
2. Reemplaza TODO el contenido de esa celda con este código
3. Ejecuta la celda
"""

# ============================================================
# OPTUNA OPTIMIZATION (CON SELECTOR DE HIPERPARÁMETROS)
# ============================================================

"""
Optimización automática de hiperparámetros usando Optuna.
Si USE_IBARRA_HYPERPARAMETERS = True, se saltará la optimización y usará parámetros de Ibarra.
Si USE_IBARRA_HYPERPARAMETERS = False, ejecutará la optimización de Optuna.
"""

# Verificar si las variables del selector están definidas
if "BEST_PARAMS" not in globals():
    print("❌ Error: Variables del selector no están definidas.")
    print("   Ejecuta primero la celda del selector de hiperparámetros.")
    print("=" * 70)
else:
    print("=" * 70)
    print("OPTUNA OPTIMIZATION (CON SELECTOR DE HIPERPARÁMETROS)")
    print("=" * 70)

    if USE_IBARRA:
        print("📚 MODO IBARRA: Saltando optimización de Optuna")
        print("   Usando parámetros exactos del paper de Ibarra 2023")
        print("   BEST_PARAMS ya contiene los parámetros de Ibarra")
        print("=" * 70)
    else:
        print("🔍 MODO OPTUNA: Ejecutando optimización automática")
        print("   Buscando mejores hiperparámetros con Optuna...")

        # ============================================================
        # CONFIGURACIÓN DE OPTUNA
        # ============================================================

        OPTUNA_CONFIG = {
            "enabled": True,
            "experiment_name": "cnn2d_optuna_optimization",
            "n_trials": 30,
            "n_epochs_per_trial": 10,
            "metric": "f1",
            "direction": "maximize",
            "pruning_enabled": True,
            "pruning_patience": 3,
            "pruning_min_trials": 2,
        }

        print(f"⚙️  Configuración de Optuna:")
        print(f"   - Número de trials: {OPTUNA_CONFIG['n_trials']}")
        print(f"   - Épocas por trial: {OPTUNA_CONFIG['n_epochs_per_trial']}")
        print(f"   - Métrica: {OPTUNA_CONFIG['metric']}")
        print(
            f"   - Pruning: {'Habilitado' if OPTUNA_CONFIG['pruning_enabled'] else 'Deshabilitado'}"
        )

        # ============================================================
        # FUNCIÓN OBJETIVO PARA OPTUNA
        # ============================================================

        def objective(trial):
            """Función objetivo para Optuna."""
            # Sugerir hiperparámetros
            params = {
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "p_drop_conv": trial.suggest_float("p_drop_conv", 0.1, 0.5),
                "p_drop_fc": trial.suggest_float("p_drop_fc", 0.1, 0.5),
                "filters_1": trial.suggest_categorical("filters_1", [32, 64, 128]),
                "filters_2": trial.suggest_categorical("filters_2", [32, 64, 128]),
                "kernel_size_1": trial.suggest_categorical(
                    "kernel_size_1", [3, 4, 5, 6]
                ),
                "kernel_size_2": trial.suggest_categorical(
                    "kernel_size_2", [7, 8, 9, 10]
                ),
                "dense_units": trial.suggest_categorical(
                    "dense_units", [16, 32, 64, 128]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
            }

            # Crear modelo con parámetros sugeridos
            model = CNN2D(
                n_classes=2,
                p_drop_conv=params["p_drop_conv"],
                p_drop_fc=params["p_drop_fc"],
                input_shape=(65, 41),
                filters_1=params["filters_1"],
                filters_2=params["filters_2"],
                kernel_size_1=params["kernel_size_1"],
                kernel_size_2=params["kernel_size_2"],
                dense_units=params["dense_units"],
            ).to(device)

            # Optimizador
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True,
            )

            # Criterion
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            # Scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )

            # DataLoaders
            train_loader = DataLoader(
                train_dataset, params["batch_size"], shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, params["batch_size"], shuffle=False, num_workers=0
            )

            # Entrenar modelo
            best_val_f1 = 0
            for epoch in range(OPTUNA_CONFIG["n_epochs_per_trial"]):
                # Entrenamiento
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()

                # Validación
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)

                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()

                        val_preds.extend(predicted.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())

                # Calcular F1 score
                val_f1 = f1_score(val_targets, val_preds, average="macro")

                # Actualizar mejor F1
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

                # Pruning
                if OPTUNA_CONFIG["pruning_enabled"]:
                    trial.report(val_f1, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                scheduler.step()

            return best_val_f1

        # ============================================================
        # EJECUTAR OPTIMIZACIÓN
        # ============================================================

        print(f"\n🚀 Iniciando optimización de Optuna...")

        # Crear estudio
        study = optuna.create_study(
            direction=OPTUNA_CONFIG["direction"],
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=OPTUNA_CONFIG["pruning_min_trials"],
                n_warmup_steps=OPTUNA_CONFIG["pruning_patience"],
            )
            if OPTUNA_CONFIG["pruning_enabled"]
            else None,
        )

        # Ejecutar optimización
        study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"])

        # Obtener mejores parámetros
        best_params = study.best_params
        best_value = study.best_value

        print(f"\n✅ Optimización completada!")
        print(f"   - Mejor F1 score: {best_value:.4f}")
        print(f"   - Mejores parámetros: {best_params}")

        # Actualizar BEST_PARAMS con los resultados de Optuna
        BEST_PARAMS.update(best_params)
        BEST_PARAMS["source"] = "optuna_optimized"
        HYPERPARAMETER_SOURCE = "Optuna Optimizado"

        # Guardar resultados
        results_dir = Path("results/cnn_optuna_optimization")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Guardar mejores parámetros
        with open(results_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # Guardar estudio completo
        with open(results_dir / "optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)

        print(f"💾 Resultados guardados en: {results_dir}")
        print("=" * 70)
