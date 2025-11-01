"""
Ejemplo de configuración de Google Colab con Git.

Este archivo demuestra cómo usar la función setup_colab_git()
para configurar el entorno de Colab de manera rápida y sencilla.

Copia la sección que necesites al inicio de cualquier notebook en Colab.
"""

# ============================================================================
# CONFIGURACION COMPLETA - Copia esto al inicio de tu notebook en Colab
# ============================================================================

# Opción 1: Configuración por defecto
# ----------------------------------------------------------------------------
# Usa los valores predeterminados:
# - computer_name="ZenBook"
# - project_dir="parkinson-voice-uncertainty"
# - branch="main"

# from modules.core.notebook_setup import setup_colab_git
# project_path = setup_colab_git()


# Opción 2: Configuración personalizada
# ----------------------------------------------------------------------------
# Especifica tus propios valores

# from modules.core.notebook_setup import setup_colab_git
# project_path = setup_colab_git(
#     computer_name="MiPC",
#     project_dir="parkinson-voice-uncertainty",
#     branch="dev"
# )


# Opción 3: Configuración compacta (una línea)
# ----------------------------------------------------------------------------

# from modules.core.notebook_setup import setup_colab_git
# setup_colab_git()


# ============================================================================
# DESPUES DE CONFIGURAR
# ============================================================================
# Una vez ejecutada setup_colab_git(), puedes usar el proyecto normalmente:

# Ejemplo 1: Cargar datos preprocesados
# from modules.data.cache_utils import load_from_cache
# healthy_data = load_from_cache("healthy_ibarra")
# parkinson_data = load_from_cache("parkinson_ibarra")

# Ejemplo 2: Entrenar un modelo
# from modules.models.cnn2d.model import CNN2D
# from modules.models.cnn2d.training import train_model
# model = CNN2D(n_classes=2)


# ============================================================================
# QUE HACE setup_colab_git()
# ============================================================================
# La función realiza automáticamente las siguientes acciones:
#
# 1. Monta Google Drive en /content/drive
# 2. Busca el proyecto en: /content/drive/Othercomputers/[PC]/[proyecto]
# 3. Configura Git (marca el repo como seguro)
# 4. Actualiza referencias remotas (fetch --all --prune)
# 5. Cambia a la rama especificada (crea si no existe localmente)
# 6. Actualiza el código (pull origin [branch])
# 7. Instala dependencias (pip install -r requirements.txt)
# 8. Cambia directorio de trabajo al proyecto
# 9. Activa autoreload para notebooks
# 10. Retorna la ruta completa del proyecto


# ============================================================================
# PARAMETROS
# ============================================================================
# computer_name (str): Nombre del PC tal como aparece en Google Drive
#                      Default: "ZenBook"
#
# project_dir (str): Nombre de la carpeta del repositorio
#                    Default: "parkinson-voice-uncertainty"
#
# branch (str): Rama de Git a utilizar
#               Default: "main"


# ============================================================================
# RETORNO
# ============================================================================
# str: Ruta completa al proyecto configurado
#      Ejemplo: "/content/drive/Othercomputers/ZenBook/parkinson-..."


# ============================================================================
# ERRORES COMUNES
# ============================================================================
# Error: "No se encuentra [PC] en Drive"
# Solución: Verifica el nombre exacto en Google Drive

# Error: "No se encuentra el repositorio"
# Solución: Verifica la ruta completa del proyecto

# Error: "Command failed: git checkout [branch]"
# Solución: La rama no existe en el remoto, verifica el nombre
