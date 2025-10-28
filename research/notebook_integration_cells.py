# ============================================================
# CELDAS PARA INTEGRAR EN EL NOTEBOOK
# ============================================================

"""
Estas son las celdas que puedes copiar y pegar en tu notebook
para reemplazar las celdas existentes y usar el nuevo sistema.
"""

# ============================================================
# CELDA 1: SELECTOR DE HIPERPARÁMETROS (REEMPLAZA LA CELDA DE CONFIGURACIÓN)
# ============================================================

"""
# ============================================================
# SELECTOR DE HIPERPARÁMETROS: IBARRA vs OPTUNA
# ============================================================

# 🔧 CONFIGURACIÓN PRINCIPAL - CAMBIA ESTE VALOR
USE_IBARRA_HYPERPARAMETERS = True  # True = Ibarra, False = Optuna

# ============================================================
# IMPORTAR SISTEMA DE CONFIGURACIÓN
# ============================================================

import sys
from pathlib import Path

# Agregar módulos al path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from modules.core.hyperparameter_config import HyperparameterManager, compare_hyperparameters

# ============================================================
# CARGAR HIPERPARÁMETROS SEGÚN CONFIGURACIÓN
# ============================================================

print("=" * 80)
print("🔧 SELECTOR DE HIPERPARÁMETROS")
print("=" * 80)

# Crear manager
manager = HyperparameterManager()

# Obtener hiperparámetros según configuración
if USE_IBARRA_HYPERPARAMETERS:
    print("📚 Usando hiperparámetros del PAPER DE IBARRA 2023")
    hyperparameters = manager.get_ibarra_hyperparameters()
    source = "Paper Ibarra 2023"
else:
    print("🔍 Usando mejores hiperparámetros de OPTUNA")
    hyperparameters = manager.get_optuna_hyperparameters()
    source = "Optuna Optimizado"

print(f"✅ Fuente: {source}")

# ============================================================
# MOSTRAR PARÁMETROS SELECCIONADOS
# ============================================================

print(f"\n📊 PARÁMETROS SELECCIONADOS:")
print("-" * 50)

# Parámetros de arquitectura
print("🏗️  ARQUITECTURA:")
print(f"   • kernel_size_1: {hyperparameters['kernel_size_1']}")
print(f"   • kernel_size_2: {hyperparameters['kernel_size_2']}")
print(f"   • filters_1: {hyperparameters['filters_1']}")
print(f"   • filters_2: {hyperparameters['filters_2']}")
print(f"   • dense_units: {hyperparameters['dense_units']}")
print(f"   • p_drop_conv: {hyperparameters['p_drop_conv']}")
print(f"   • p_drop_fc: {hyperparameters['p_drop_fc']}")

# Parámetros de entrenamiento
print("\n🚀 ENTRENAMIENTO:")
print(f"   • batch_size: {hyperparameters['batch_size']}")
print(f"   • learning_rate: {hyperparameters['learning_rate']}")
print(f"   • momentum: {hyperparameters['momentum']}")
print(f"   • weight_decay: {hyperparameters['weight_decay']}")
print(f"   • n_epochs: {hyperparameters['n_epochs']}")
print(f"   • early_stopping_patience: {hyperparameters['early_stopping_patience']}")

# Parámetros del scheduler
print("\n📈 SCHEDULER:")
print(f"   • step_size: {hyperparameters['step_size']}")
print(f"   • gamma: {hyperparameters['gamma']}")
print(f"   • optimizer: {hyperparameters['optimizer']}")

# ============================================================
# COMPARACIÓN (OPCIONAL)
# ============================================================

if not USE_IBARRA_HYPERPARAMETERS:
    print(f"\n📊 COMPARACIÓN CON IBARRA:")
    print("-" * 50)
    
    ibarra_params = manager.get_ibarra_hyperparameters()
    
    # Comparar parámetros clave
    key_params = ["kernel_size_1", "kernel_size_2", "filters_2", "dense_units", "batch_size", "learning_rate"]
    
    for param in key_params:
        ibarra_val = ibarra_params[param]
        optuna_val = hyperparameters[param]
        
        if ibarra_val != optuna_val:
            if isinstance(ibarra_val, (int, float)) and isinstance(optuna_val, (int, float)):
                diff = optuna_val - ibarra_val
                diff_str = f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+d}"
                print(f"   • {param}: Ibarra={ibarra_val} → Optuna={optuna_val} ({diff_str})")
            else:
                print(f"   • {param}: Ibarra={ibarra_val} → Optuna={optuna_val}")
        else:
            print(f"   • {param}: {ibarra_val} (igual)")

# ============================================================
# GUARDAR CONFIGURACIÓN
# ============================================================

# Guardar configuración actual
manager.save_config(use_ibarra=USE_IBARRA_HYPERPARAMETERS)

print(f"\n💾 Configuración guardada:")
print(f"   • Usar Ibarra: {USE_IBARRA_HYPERPARAMETERS}")
print(f"   • Archivo: config/hyperparameter_config.json")

# ============================================================
# PREPARAR VARIABLES PARA EL RESTO DEL NOTEBOOK
# ============================================================

# Crear variables globales que el resto del notebook puede usar
BEST_PARAMS = hyperparameters
USE_IBARRA = USE_IBARRA_HYPERPARAMETERS
HYPERPARAMETER_SOURCE = source

print(f"\n✅ Variables preparadas:")
print(f"   • BEST_PARAMS: Diccionario con hiperparámetros")
print(f"   • USE_IBARRA: {USE_IBARRA}")
print(f"   • HYPERPARAMETER_SOURCE: {HYPERPARAMETER_SOURCE}")

print("=" * 80)
print("🎯 LISTO PARA ENTRENAR CON LOS PARÁMETROS SELECCIONADOS")
print("=" * 80)
"""

# ============================================================
# CELDA 2: CREAR MODELO (REEMPLAZA LA CELDA DE CREACIÓN DE MODELO)
# ============================================================

"""
# ============================================================
# CREAR MODELO CON HIPERPARÁMETROS SELECCIONADOS
# ============================================================

print("=" * 70)
print("CREANDO MODELO CON HIPERPARÁMETROS SELECCIONADOS")
print("=" * 70)

# Verificar que las variables estén definidas
if 'BEST_PARAMS' not in globals():
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
"""

# ============================================================
# CELDA 3: CONFIGURAR ENTRENAMIENTO (REEMPLAZA LA CELDA DE CONFIGURACIÓN DE ENTRENAMIENTO)
# ============================================================

"""
# ============================================================
# CONFIGURAR ENTRENAMIENTO CON HIPERPARÁMETROS SELECCIONADOS
# ============================================================

print("=" * 70)
print("CONFIGURANDO ENTRENAMIENTO CON HIPERPARÁMETROS SELECCIONADOS")
print("=" * 70)

# Verificar que las variables estén definidas
if 'BEST_PARAMS' not in globals():
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
        "batch_size": BEST_PARAMS["batch_size"]
    }
    
    # Crear DataLoaders con el batch size seleccionado
    train_loader_final = DataLoader(
        train_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=True,
        num_workers=0
    )
    val_loader_final = DataLoader(
        val_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=False,
        num_workers=0
    )
    test_loader_final = DataLoader(
        test_dataset,
        BEST_PARAMS["batch_size"],  # Usar batch_size seleccionado
        shuffle=False,
        num_workers=0
    )
    
    # Optimizador SGD con momentum usando los parámetros seleccionados
    optimizer_final = optim.SGD(
        best_model.parameters(),
        lr=FINAL_TRAINING_CONFIG['learning_rate'],
        momentum=BEST_PARAMS["momentum"],
        weight_decay=BEST_PARAMS["weight_decay"],
        nesterov=True  # Mejora sobre Ibarra
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
        optimizer_final,
        step_size=BEST_PARAMS["step_size"],
        gamma=BEST_PARAMS["gamma"]
    )
    
    print(f"\n⚙️  Configuración final ({HYPERPARAMETER_SOURCE}):")
    print(f"   - Learning rate inicial: {FINAL_TRAINING_CONFIG['learning_rate']}")
    print(f"   - Momentum: {BEST_PARAMS['momentum']} (Nesterov: True)")
    print(f"   - Weight decay: {BEST_PARAMS['weight_decay']}")
    print(f"   - Scheduler: StepLR (step={BEST_PARAMS['step_size']}, gamma={BEST_PARAMS['gamma']})")
    print(f"   - Batch size: {FINAL_TRAINING_CONFIG['batch_size']}")
    print(f"   - Épocas máximas: {FINAL_TRAINING_CONFIG['n_epochs']}")
    print(f"   - Early stopping patience: {FINAL_TRAINING_CONFIG['early_stopping_patience']}")
    
    # Mostrar diferencias con Ibarra si estamos usando Optuna
    if not USE_IBARRA:
        print(f"\n📊 DIFERENCIAS CON IBARRA:")
        print(f"   - Ibarra batch_size: 64 → Optuna: {BEST_PARAMS['batch_size']}")
        print(f"   - Ibarra kernel_1: 6 → Optuna: {BEST_PARAMS['kernel_size_1']}")
        print(f"   - Ibarra filters_2: 64 → Optuna: {BEST_PARAMS['filters_2']}")
        print(f"   - Ibarra dense_units: 32 → Optuna: {BEST_PARAMS['dense_units']}")
    
    print("=" * 70)
"""

# ============================================================
# CELDA 4: COMPARACIÓN RÁPIDA (NUEVA CELDA OPCIONAL)
# ============================================================

"""
# ============================================================
# COMPARACIÓN RÁPIDA: IBARRA vs OPTUNA
# ============================================================

print("=" * 80)
print("📊 COMPARACIÓN RÁPIDA: IBARRA vs OPTUNA")
print("=" * 80)

try:
    compare_hyperparameters()
except Exception as e:
    print(f"Error en comparación: {e}")
    print("Asegúrate de que el sistema de hiperparámetros esté correctamente instalado.")

print("=" * 80)
"""
