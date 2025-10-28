#!/usr/bin/env python3
"""
Prueba de conexión con Weights & Biases
=======================================

Script para verificar que la conexión con wandb funciona correctamente.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.training_monitor import test_wandb_connection
from modules.core.experiment_config import WANDB_CONFIG


def test_wandb_setup():
    """Probar configuración completa de wandb."""
    print("=" * 70)
    print("PRUEBA DE CONEXIÓN CON WEIGHTS & BIASES")
    print("=" * 70)

    # Mostrar configuración
    print(f"📊 Configuración:")
    print(f"   - Proyecto: {WANDB_CONFIG['project_name']}")
    print(f"   - API Key: {'*' * 20}...{WANDB_CONFIG['api_key'][-4:]}")
    print(f"   - Tags: {WANDB_CONFIG['tags']}")
    print(f"   - Notas: {WANDB_CONFIG['notes']}")

    # Probar conexión
    print(f"\n🔗 Probando conexión...")
    success = test_wandb_connection(WANDB_CONFIG["api_key"])

    if success:
        print(f"\n✅ ¡Conexión exitosa! Puedes usar wandb en tu notebook.")
        print(f"   📊 Ve a https://wandb.ai para ver tus experimentos")
    else:
        print(f"\n❌ Error en la conexión. Verifica tu API key.")
        print(f"   🔑 Obtén tu API key en: https://wandb.ai/authorize")

    print("=" * 70)
    return success


if __name__ == "__main__":
    test_wandb_setup()
