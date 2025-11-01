#!/usr/bin/env python3
"""
Prueba de integración completa con Weights & Biases
==================================================

Script para verificar que toda la integración con wandb funciona correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.training_monitor import create_training_monitor, test_wandb_connection
from modules.core.experiment_config import WANDB_CONFIG, TRAINING_MONITOR_CONFIG
import torch
import numpy as np


def test_wandb_integration():
    """Probar integración completa con wandb."""
    print("=" * 70)
    print("PRUEBA DE INTEGRACIÓN COMPLETA CON WANDB")
    print("=" * 70)

    # 1. Probar conexión
    print("\n1️⃣ Probando conexión con wandb...")
    connection_success = test_wandb_connection(WANDB_CONFIG["api_key"])

    if not connection_success:
        print("❌ Error en conexión - No se puede continuar")
        return False

    # 2. Crear configuración de prueba
    print("\n2️⃣ Creando configuración de prueba...")
    test_config = {
        "experiment_name": "test_wandb_integration",
        "model_architecture": "CNN2D",
        "dataset": "Test Data",
        "test_param": 42,
    }

    # 3. Crear monitor de prueba
    print("\n3️⃣ Creando monitor de prueba...")
    try:
        monitor = create_training_monitor(
            config=test_config,
            experiment_name="test_integration",
            use_wandb=True,
            wandb_key=WANDB_CONFIG["api_key"],
            tags=["test", "integration"],
            notes="Prueba de integración con wandb",
        )
        print("✅ Monitor creado exitosamente")
    except Exception as e:
        print(f"❌ Error creando monitor: {e}")
        return False

    # 4. Simular entrenamiento
    print("\n4️⃣ Simulando entrenamiento...")
    try:
        for epoch in range(5):
            # Simular métricas
            train_loss = 1.0 - epoch * 0.1 + np.random.normal(0, 0.05)
            val_loss = 1.2 - epoch * 0.08 + np.random.normal(0, 0.05)
            train_f1 = 0.5 + epoch * 0.1 + np.random.normal(0, 0.02)
            val_f1 = 0.4 + epoch * 0.12 + np.random.normal(0, 0.02)

            # Loggear métricas
            monitor.log(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_f1=train_f1,
                val_f1=val_f1,
                learning_rate=0.001,
            )

            print(f"   Época {epoch + 1}: Train F1={train_f1:.3f}, Val F1={val_f1:.3f}")

        print("✅ Simulación de entrenamiento completada")
    except Exception as e:
        print(f"❌ Error en simulación: {e}")
        return False

    # 5. Probar plotting local
    print("\n5️⃣ Probando plotting local...")
    try:
        monitor.plot_local()
        print("✅ Plotting local funcionando")
    except Exception as e:
        print(f"❌ Error en plotting: {e}")
        return False

    # 6. Finalizar experimento
    print("\n6️⃣ Finalizando experimento...")
    try:
        monitor.finish()
        print("✅ Experimento finalizado correctamente")
    except Exception as e:
        print(f"❌ Error finalizando: {e}")
        return False

    print("\n🎉 ¡INTEGRACIÓN COMPLETA EXITOSA!")
    print("=" * 70)
    return True


def test_configuration():
    """Probar configuración de wandb."""
    print("\n🔧 CONFIGURACIÓN DE WANDB:")
    print(f"   - Proyecto: {WANDB_CONFIG['project_name']}")
    print(f"   - API Key: {'*' * 20}...{WANDB_CONFIG['api_key'][-4:]}")
    print(f"   - Tags: {WANDB_CONFIG['tags']}")
    print(f"   - Notas: {WANDB_CONFIG['notes']}")

    print("\n📊 CONFIGURACIÓN DE MONITOREO:")
    for key, value in TRAINING_MONITOR_CONFIG.items():
        print(f"   - {key}: {value}")


if __name__ == "__main__":
    test_configuration()
    success = test_wandb_integration()

    if success:
        print("\n✅ Todas las pruebas pasaron - wandb está listo para usar")
    else:
        print("\n❌ Algunas pruebas fallaron - revisa la configuración")
