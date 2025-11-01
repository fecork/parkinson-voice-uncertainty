#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de configuración de hiperparámetros.
"""

import sys
from pathlib import Path

# Agregar módulos al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.hyperparameter_config import (
    HyperparameterManager,
    get_hyperparameters,
    compare_hyperparameters,
)


def example_ibarra_usage():
    """Ejemplo de uso con parámetros de Ibarra."""
    print("=" * 80)
    print("📚 EJEMPLO: USANDO PARÁMETROS DE IBARRA")
    print("=" * 80)

    # Opción 1: Usar función de conveniencia
    ibarra_params = get_hyperparameters(use_ibarra=True)

    print("Parámetros de Ibarra obtenidos:")
    for key, value in ibarra_params.items():
        print(f"  {key}: {value}")

    return ibarra_params


def example_optuna_usage():
    """Ejemplo de uso con parámetros de Optuna."""
    print("\n" + "=" * 80)
    print("🔍 EJEMPLO: USANDO PARÁMETROS DE OPTUNA")
    print("=" * 80)

    # Opción 2: Usar manager directamente
    manager = HyperparameterManager()

    # Buscar archivos de Optuna
    optuna_paths = [
        Path("checkpoints/cnn2d_optuna_best_params.json"),
        Path("results/cnn_optuna_optimization/best_params.json"),
        Path("checkpoints/cnn2d_optuna_trials.json"),
    ]

    optuna_path = None
    for path in optuna_paths:
        if path.exists():
            optuna_path = path
            break

    if optuna_path:
        print(f"Usando archivo de Optuna: {optuna_path}")
        optuna_params = manager.get_optuna_hyperparameters(optuna_path)
    else:
        print("No se encontró archivo de Optuna, usando Ibarra como fallback")
        optuna_params = manager.get_optuna_hyperparameters()

    print("Parámetros de Optuna obtenidos:")
    for key, value in optuna_params.items():
        print(f"  {key}: {value}")

    return optuna_params


def example_config_management():
    """Ejemplo de gestión de configuración."""
    print("\n" + "=" * 80)
    print("⚙️  EJEMPLO: GESTIÓN DE CONFIGURACIÓN")
    print("=" * 80)

    manager = HyperparameterManager()

    # Guardar configuración para usar Ibarra
    print("Guardando configuración para usar Ibarra...")
    manager.save_config(use_ibarra=True)

    # Cargar configuración
    print("Cargando configuración...")
    config = manager.load_config()
    print(f"Configuración cargada: {config}")

    # Cambiar a Optuna
    print("Cambiando a Optuna...")
    manager.save_config(use_ibarra=False)

    # Cargar nueva configuración
    config = manager.load_config()
    print(f"Nueva configuración: {config}")


def example_model_creation():
    """Ejemplo de creación de modelo con parámetros seleccionados."""
    print("\n" + "=" * 80)
    print("🏗️  EJEMPLO: CREACIÓN DE MODELO")
    print("=" * 80)

    # Simular selección de parámetros
    use_ibarra = True  # Cambiar a False para usar Optuna

    if use_ibarra:
        print("Usando parámetros de Ibarra...")
        params = get_hyperparameters(use_ibarra=True)
    else:
        print("Usando parámetros de Optuna...")
        params = get_hyperparameters(use_ibarra=False)

    # Simular creación de modelo (sin PyTorch para el ejemplo)
    print(f"\nCreando modelo CNN2D con:")
    print(f"  - kernel_size_1: {params['kernel_size_1']}")
    print(f"  - kernel_size_2: {params['kernel_size_2']}")
    print(f"  - filters_1: {params['filters_1']}")
    print(f"  - filters_2: {params['filters_2']}")
    print(f"  - dense_units: {params['dense_units']}")
    print(f"  - p_drop_conv: {params['p_drop_conv']}")
    print(f"  - p_drop_fc: {params['p_drop_fc']}")
    print(f"  - batch_size: {params['batch_size']}")
    print(f"  - learning_rate: {params['learning_rate']}")

    return params


def example_notebook_integration():
    """Ejemplo de cómo integrar en un notebook."""
    print("\n" + "=" * 80)
    print("📓 EJEMPLO: INTEGRACIÓN EN NOTEBOOK")
    print("=" * 80)

    print("""
# CELDA 1: Selector de hiperparámetros
from modules.core.hyperparameter_config import get_hyperparameters

# Configuración - CAMBIA ESTE VALOR
USE_IBARRA_HYPERPARAMETERS = True  # True = Ibarra, False = Optuna

# Obtener hiperparámetros
BEST_PARAMS = get_hyperparameters(use_ibarra=USE_IBARRA_HYPERPARAMETERS)
HYPERPARAMETER_SOURCE = "Ibarra 2023" if USE_IBARRA_HYPERPARAMETERS else "Optuna"

print(f"Usando parámetros de: {HYPERPARAMETER_SOURCE}")
print(f"Batch size: {BEST_PARAMS['batch_size']}")
print(f"Learning rate: {BEST_PARAMS['learning_rate']}")

# CELDA 2: Crear modelo
from modules.models.cnn2d.model import CNN2D

model = CNN2D(
    n_classes=2,
    kernel_size_1=BEST_PARAMS["kernel_size_1"],
    kernel_size_2=BEST_PARAMS["kernel_size_2"],
    filters_1=BEST_PARAMS["filters_1"],
    filters_2=BEST_PARAMS["filters_2"],
    dense_units=BEST_PARAMS["dense_units"],
    p_drop_conv=BEST_PARAMS["p_drop_conv"],
    p_drop_fc=BEST_PARAMS["p_drop_fc"],
)

# CELDA 3: Configurar entrenamiento
import torch.optim as optim

optimizer = optim.SGD(
    model.parameters(),
    lr=BEST_PARAMS["learning_rate"],
    momentum=BEST_PARAMS["momentum"],
    weight_decay=BEST_PARAMS["weight_decay"]
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=BEST_PARAMS["step_size"],
    gamma=BEST_PARAMS["gamma"]
)
    """)


def main():
    """Función principal con todos los ejemplos."""
    print("🚀 EJEMPLOS DE USO DEL SISTEMA DE HIPERPARÁMETROS")
    print("=" * 80)

    # Ejemplo 1: Usar Ibarra
    ibarra_params = example_ibarra_usage()

    # Ejemplo 2: Usar Optuna
    optuna_params = example_optuna_usage()

    # Ejemplo 3: Gestión de configuración
    example_config_management()

    # Ejemplo 4: Creación de modelo
    model_params = example_model_creation()

    # Ejemplo 5: Integración en notebook
    example_notebook_integration()

    # Comparación final
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 80)

    try:
        compare_hyperparameters()
    except Exception as e:
        print(f"Error en comparación: {e}")

    print("\n🎉 TODOS LOS EJEMPLOS COMPLETADOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
