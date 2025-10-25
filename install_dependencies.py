#!/usr/bin/env python3
"""
Script para instalar dependencias del proyecto Parkinson Voice Uncertainty.

Este script usa el módulo centralizado de gestión de dependencias para
verificar e instalar automáticamente todas las dependencias necesarias.

Uso:
    python install_dependencies.py

Opciones:
    --check-only: Solo verificar dependencias sin instalar
    --force: Forzar reinstalación de todas las dependencias
    --requirements: Ruta personalizada a requirements.txt
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio del proyecto al path para importar módulos
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core.dependency_manager import (
    install_dependencies, 
    check_dependencies, 
    get_environment_info,
    print_environment_info
)


def main():
    parser = argparse.ArgumentParser(description='Instalar dependencias del proyecto')
    parser.add_argument('--check-only', action='store_true', help='Solo verificar dependencias')
    parser.add_argument('--force', action='store_true', help='Forzar reinstalación')
    parser.add_argument('--requirements', help='Ruta personalizada a requirements.txt')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 INSTALADOR DE DEPENDENCIAS - PARKINSON VOICE UNCERTAINTY")
    print("="*70)
    
    # Mostrar información del entorno
    if args.verbose:
        print_environment_info()
    
    if args.check_only:
        # Solo verificar dependencias
        all_installed, missing = check_dependencies(verbose=True)
        if all_installed:
            print("\n✅ Todas las dependencias están instaladas")
            return 0
        else:
            print(f"\n⚠️  Faltan {len(missing)} dependencias: {missing}")
            return 1
    else:
        # Instalar dependencias
        success = install_dependencies(
            force_install=args.force,
            requirements_path=args.requirements,
            verbose=True
        )
        
        if success:
            print("\n✅ ¡Instalación completada exitosamente!")
            print("🚀 El proyecto está listo para ejecutar")
            return 0
        else:
            print("\n❌ Error durante la instalación")
            print("💡 Intenta instalar manualmente:")
            print("   pip install -r requirements.txt")
            return 1


if __name__ == "__main__":
    sys.exit(main())
