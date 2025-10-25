"""
Módulo centralizado para gestión de dependencias.

Este módulo maneja la instalación automática de dependencias para todos los notebooks
del proyecto, evitando duplicidad de código y proporcionando una interfaz unificada.

Uso:
    from modules.core.dependency_manager import install_dependencies, check_dependencies

    # Verificar dependencias
    if not check_dependencies():
        install_dependencies()
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Dependencias principales del proyecto
REQUIRED_MODULES = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "librosa": "Librosa",
    "soundfile": "SoundFile",
    "optuna": "Optuna",
    "jupyter": "Jupyter",
}

# Dependencias básicas para instalación manual
BASIC_DEPENDENCIES = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "librosa>=0.8.1",
    "soundfile>=0.10.3",
    "optuna>=3.0.0",
    "jupyter>=1.0.0",
    "tqdm>=4.62.0",
]


def is_colab() -> bool:
    """
    Detectar si estamos ejecutando en Google Colab.

    Returns:
        bool: True si estamos en Colab, False en caso contrario
    """
    try:
        import google.colab

        return True
    except ImportError:
        return False


def is_jupyter() -> bool:
    """
    Detectar si estamos ejecutando en Jupyter (local o Colab).

    Returns:
        bool: True si estamos en Jupyter, False en caso contrario
    """
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def check_dependencies(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Verificar qué dependencias están instaladas.

    Args:
        verbose: Si mostrar información detallada

    Returns:
        Tuple[bool, List[str]]: (todas_instaladas, dependencias_faltantes)
    """
    installed = []
    missing = []

    for module, name in REQUIRED_MODULES.items():
        try:
            __import__(module)
            installed.append(f"✅ {name}")
        except ImportError:
            missing.append(module)
            if verbose:
                print(f"❌ {name} - {module}")

    if verbose:
        print("🔍 Estado de dependencias:")
        for item in installed:
            print(f"   {item}")
        for module in missing:
            print(f"   ❌ {REQUIRED_MODULES[module]} - {module}")

    return len(missing) == 0, missing


def install_from_requirements(requirements_path: Optional[str] = None) -> bool:
    """
    Instalar dependencias desde requirements.txt.

    Args:
        requirements_path: Ruta al archivo requirements.txt (opcional)

    Returns:
        bool: True si la instalación fue exitosa, False en caso contrario
    """
    if requirements_path is None:
        requirements_path = "requirements.txt"

    requirements_file = Path(requirements_path)

    if not requirements_file.exists():
        print(f"❌ {requirements_path} no encontrado")
        return False

    try:
        print(f"📦 Instalando desde {requirements_path}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✅ Dependencias instaladas desde requirements.txt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando desde {requirements_path}: {e}")
        return False


def install_basic_dependencies() -> bool:
    """
    Instalar dependencias básicas una por una.

    Returns:
        bool: True si todas las instalaciones fueron exitosas, False en caso contrario
    """
    print("📦 Instalando dependencias básicas...")
    failed_installations = []

    for dep in BASIC_DEPENDENCIES:
        try:
            print(f"   Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Error instalando {dep}: {e}")
            failed_installations.append(dep)
            continue

    if failed_installations:
        print(
            f"⚠️  Fallaron {len(failed_installations)} instalaciones: {failed_installations}"
        )
        return False
    else:
        print("✅ Instalación de dependencias básicas completada")
        return True


def install_dependencies(
    force_install: bool = False,
    requirements_path: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """
    Instalar dependencias automáticamente.

    Args:
        force_install: Forzar instalación aunque las dependencias estén presentes
        requirements_path: Ruta personalizada al requirements.txt
        verbose: Mostrar información detallada

    Returns:
        bool: True si la instalación fue exitosa, False en caso contrario
    """
    if verbose:
        print("=" * 70)
        print("🔧 GESTIÓN DE DEPENDENCIAS - PARKINSON VOICE UNCERTAINTY")
        print("=" * 70)

    # Verificar dependencias actuales
    all_installed, missing = check_dependencies(verbose=verbose)

    if all_installed and not force_install:
        if verbose:
            print("\n✅ Todas las dependencias están instaladas")
        return True

    if not force_install and not missing:
        if verbose:
            print("\n✅ No hay dependencias faltantes")
        return True

    # Detectar entorno
    if is_colab():
        if verbose:
            print("\n🔍 Entorno: Google Colab")
        env_type = "colab"
    else:
        if verbose:
            print("\n🔍 Entorno: Local")
        env_type = "local"

    # Instalar dependencias faltantes
    if verbose:
        print(f"\n🚀 Instalando {len(missing)} dependencias faltantes...")

    # Intentar instalar desde requirements.txt primero
    success = install_from_requirements(requirements_path)

    if not success:
        if verbose:
            print(
                "⚠️  Falló instalación desde requirements.txt, intentando dependencias básicas..."
            )
        success = install_basic_dependencies()

    # Verificar instalación final
    if verbose:
        print("\n🔍 Verificando instalación final...")

    all_installed, missing = check_dependencies(verbose=False)

    if all_installed:
        if verbose:
            print("\n✅ ¡Todas las dependencias instaladas correctamente!")
            print("🚀 El proyecto está listo para ejecutar")
        return True
    else:
        if verbose:
            print(f"\n⚠️  Aún faltan {len(missing)} dependencias: {missing}")
            print("💡 Intenta instalar manualmente:")
            print("   pip install -r requirements.txt")
        return False


def get_environment_info() -> Dict[str, str]:
    """
    Obtener información del entorno de ejecución.

    Returns:
        Dict con información del entorno
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "is_colab": str(is_colab()),
        "is_jupyter": str(is_jupyter()),
        "working_directory": str(Path.cwd()),
    }

    # Información adicional si está disponible
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        info["torch_version"] = "No instalado"
        info["cuda_available"] = "N/A"

    return info


def print_environment_info():
    """Imprimir información del entorno de ejecución."""
    info = get_environment_info()

    print("🔍 Información del entorno:")
    for key, value in info.items():
        print(f"   {key}: {value}")


def setup_notebook_environment(auto_install: bool = True, verbose: bool = True) -> bool:
    """
    Configurar el entorno para notebooks.

    Esta función debe ser llamada al inicio de cada notebook para asegurar
    que todas las dependencias estén disponibles.

    Args:
        auto_install: Instalar automáticamente dependencias faltantes
        verbose: Mostrar información detallada

    Returns:
        bool: True si el entorno está listo, False en caso contrario
    """
    if verbose:
        print("🚀 Configurando entorno para notebook...")
        print_environment_info()

    # Verificar dependencias
    all_installed, missing = check_dependencies(verbose=verbose)

    if all_installed:
        if verbose:
            print("\n✅ Entorno listo - todas las dependencias disponibles")
        return True

    if not auto_install:
        if verbose:
            print(f"\n⚠️  Faltan {len(missing)} dependencias: {missing}")
            print("💡 Ejecuta: install_dependencies() para instalarlas")
        return False

    # Instalar dependencias automáticamente
    if verbose:
        print(f"\n🔧 Instalando {len(missing)} dependencias faltantes...")

    success = install_dependencies(verbose=verbose)

    if success:
        if verbose:
            print("\n✅ Entorno configurado correctamente")
        return True
    else:
        if verbose:
            print("\n❌ Error configurando entorno")
        return False


# Función de conveniencia para notebooks
def setup():
    """
    Función de conveniencia para configurar el entorno.

    Esta es la función principal que deben llamar los notebooks.
    """
    return setup_notebook_environment(auto_install=True, verbose=True)


if __name__ == "__main__":
    # Si se ejecuta como script, instalar dependencias
    import argparse

    parser = argparse.ArgumentParser(description="Gestión de dependencias del proyecto")
    parser.add_argument(
        "--check-only", action="store_true", help="Solo verificar dependencias"
    )
    parser.add_argument("--force", action="store_true", help="Forzar reinstalación")
    parser.add_argument("--requirements", help="Ruta personalizada a requirements.txt")

    args = parser.parse_args()

    if args.check_only:
        all_installed, missing = check_dependencies()
        sys.exit(0 if all_installed else 1)
    else:
        success = install_dependencies(
            force_install=args.force, requirements_path=args.requirements
        )
        sys.exit(0 if success else 1)
