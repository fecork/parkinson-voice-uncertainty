"""
Plantilla de configuración para notebooks.

Copia este código al inicio de cualquier notebook para configurar
automáticamente el entorno y las dependencias.

Uso:
    1. Copia el código de abajo
    2. Pégalo en la primera celda de tu notebook
    3. Ejecuta la celda

El código detectará automáticamente si estás en Colab o local y
instalará las dependencias necesarias.
"""

# ============================================================
# CONFIGURAR ENTORNO Y DEPENDENCIAS (PLANTILLA)
# ============================================================

# Importar el gestor de dependencias centralizado
from modules.core.dependency_manager import setup_notebook_environment

# Configurar el entorno automáticamente
# Esto verifica e instala todas las dependencias necesarias
success = setup_notebook_environment(auto_install=True, verbose=True)

if not success:
    print("❌ Error configurando el entorno")
    print("💡 Intenta instalar manualmente: pip install -r requirements.txt")
    import sys
    sys.exit(1)

print("="*70)
