# ============================================================
# CELDA PARA INSERTAR EN EL NOTEBOOK cnn2d_training.ipynb
# ============================================================

"""
INSTRUCCIONES:
1. Copia este código
2. Insértalo como una NUEVA CELDA después de la celda de configuración (Cell 3)
3. Ejecuta la celda
4. Cambia USE_IBARRA_HYPERPARAMETERS = True/False según quieras usar Ibarra o Optuna
"""

# ============================================================
# SELECTOR DE HIPERPARÁMETROS: IBARRA vs OPTUNA
# ============================================================

"""
Esta celda permite elegir entre usar los hiperparámetros exactos del paper de Ibarra
o los mejores hiperparámetros encontrados por Optuna.

USO:
1. Para usar parámetros de Ibarra: USE_IBARRA_HYPERPARAMETERS = True
2. Para usar parámetros de Optuna: USE_IBARRA_HYPERPARAMETERS = False
"""

# ============================================================
# CONFIGURACIÓN - CAMBIA ESTE VALOR SEGÚN LO QUE QUIERAS USAR
# ============================================================

# 🔧 CONFIGURACIÓN PRINCIPAL
USE_IBARRA_HYPERPARAMETERS = True  # True = Ibarra, False = Optuna

# ============================================================
# IMPORTAR SISTEMA DE CONFIGURACIÓN
# ============================================================

import sys
from pathlib import Path

# Agregar módulos al path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from modules.core.hyperparameter_config import (
    HyperparameterManager,
    compare_hyperparameters,
)

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
    key_params = [
        "kernel_size_1",
        "kernel_size_2",
        "filters_2",
        "dense_units",
        "batch_size",
        "learning_rate",
    ]

    for param in key_params:
        ibarra_val = ibarra_params[param]
        optuna_val = hyperparameters[param]

        if ibarra_val != optuna_val:
            if isinstance(ibarra_val, (int, float)) and isinstance(
                optuna_val, (int, float)
            ):
                diff = optuna_val - ibarra_val
                diff_str = f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+d}"
                print(
                    f"   • {param}: Ibarra={ibarra_val} → Optuna={optuna_val} ({diff_str})"
                )
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
