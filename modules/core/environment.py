"""
Módulo para detectar el entorno de ejecución (Local/Colab) y configurar rutas.

Este módulo proporciona funciones para:
- Detectar automáticamente si el código corre en Google Colab o en local
- Configurar rutas del proyecto según el entorno detectado
- Proporcionar acceso consistente a directorios del proyecto

Uso:
    from modules.core.environment import setup_environment

    ENV, PATHS = setup_environment()
    print(f"Entorno: {ENV}")
    print(f"Cache: {PATHS['cache_original']}")
"""

from pathlib import Path
from typing import Dict, Optional


def find_project_root(start_path: Path = None) -> Path:
    """
    Encuentra la raíz del proyecto buscando archivos característicos.

    Busca hacia arriba desde el directorio actual hasta encontrar
    la raíz del proyecto (identificada por archivos como requirements.txt,
    README.md, o la carpeta modules/).

    Args:
        start_path: Directorio desde donde empezar. Si None, usa cwd.

    Returns:
        Path: Ruta a la raíz del proyecto

    Raises:
        FileNotFoundError: Si no se puede encontrar la raíz del proyecto
    """
    if start_path is None:
        start_path = Path.cwd()

    # Archivos/carpetas que identifican la raíz del proyecto
    markers = [
        "requirements.txt",
        "Pipfile",
        ".git",
        "modules",
        "setup.py",
        "pyproject.toml",
    ]

    current = start_path.resolve()

    # Buscar hacia arriba hasta encontrar un marcador
    for _ in range(10):  # Límite de 10 niveles hacia arriba
        for marker in markers:
            if (current / marker).exists():
                return current

        # Subir un nivel
        parent = current.parent
        if parent == current:  # Llegamos a la raíz del sistema
            break
        current = parent

    # Si no encontramos marcadores, usar directorio actual
    return Path.cwd()


def detect_environment() -> str:
    """
    Detecta si el código está corriendo en Google Colab o en local.

    Returns:
        str: 'colab' si está en Google Colab, 'local' si está en local

    Examples:
        >>> env = detect_environment()
        >>> print(env)
        'local'
    """
    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        return "local"


def get_project_paths(
    environment: Optional[str] = None, colab_base: Optional[str] = None
) -> Dict[str, Path]:
    """
    Retorna las rutas principales del proyecto según el entorno.

    Args:
        environment: 'colab' o 'local'. Si es None, detecta automático.
        colab_base: Ruta base en Colab. Si es None, usa default.

    Returns:
        dict: Diccionario con las rutas principales del proyecto:
            - base: Directorio raíz del proyecto
            - cache_original: Directorio de cache original
            - cache_augmented: Directorio de cache augmentado
            - cache_sequences: Directorio de cache de secuencias
            - results: Directorio de resultados
            - data: Directorio de datos
            - notebooks: Directorio de notebooks
            - modules: Directorio de módulos

    Examples:
        >>> paths = get_project_paths('local')
        >>> print(paths['cache_original'])
        cache/original
    """
    if environment is None:
        environment = detect_environment()

    if environment == "colab":
        if colab_base is None:
            # Ruta por defecto en Colab (Google Drive)
            default_colab = "/content/drive/Othercomputers/ZenBook"
            colab_base = f"{default_colab}/parkinson-voice-uncertainty"
        base_path = Path(colab_base)
    else:
        # En local, encontrar la raíz del proyecto automáticamente
        base_path = find_project_root()

    return {
        "base": base_path,
        "cache_original": base_path / "cache" / "original",
        "cache_augmented": base_path / "cache" / "augmented",
        "cache_sequences": base_path / "cache" / "sequences",
        "config": base_path / "config",
        "results": base_path / "results",
        "data": base_path / "data",
        "notebooks": base_path / "notebooks",
        "modules": base_path / "modules",
        "pipelines": base_path / "pipelines",
        "research": base_path / "research",
        "test": base_path / "test",
    }


def setup_environment(verbose: bool = True, colab_base: Optional[str] = None) -> tuple:
    """
    Configura entorno detectando Colab/local y retornando rutas.

    Esta es la función principal que se debe usar en los notebooks.
    Detecta el entorno automáticamente y configura las rutas.

    En local, busca automáticamente la raíz del proyecto (por ejemplo,
    si ejecutas desde research/, encuentra la raíz correctamente).

    Args:
        verbose: Si True, imprime info sobre el entorno detectado
        colab_base: Ruta base custom para Colab. Si None, usa default.

    Returns:
        tuple: (environment, paths) donde:
            - environment (str): 'colab' o 'local'
            - paths (dict): Diccionario con todas las rutas del proyecto

    Examples:
        >>> ENV, PATHS = setup_environment()
        ==================================================================
        CONFIGURACIÓN DE ENTORNO
        ==================================================================
        Entorno detectado: LOCAL
        Ruta base: /path/to/parkinson-voice-uncertainty
        Cache original: /path/to/parkinson-voice-uncertainty/cache/original
        Cache augmented: /path/to/parkinson-voice-uncertainty/cache/augmented
        ==================================================================

        >>> # Usar las rutas (funciona desde cualquier subdirectorio)
        >>> cache_path = PATHS['cache_original'] / "healthy_ibarra.pkl"
    """
    env = detect_environment()
    paths = get_project_paths(env, colab_base)

    if verbose:
        print("=" * 70)
        print("CONFIGURACIÓN DE ENTORNO")
        print("=" * 70)
        print(f"Entorno detectado: {env.upper()}")
        print(f"Ruta base: {paths['base']}")
        print(f"Cache original: {paths['cache_original']}")
        print(f"Cache augmented: {paths['cache_augmented']}")
        if env == "colab":
            print("\nMODO COLAB: Usando rutas de Google Drive")
        else:
            print("\nMODO LOCAL: Usando rutas relativas")
        print("=" * 70)

    return env, paths


def get_colab_drive_paths() -> Dict[str, str]:
    """
    Retorna posibles rutas de Google Drive en Colab.

    Útil para detectar diferentes montajes de Google Drive.

    Returns:
        dict: Diccionario con rutas comunes de Google Drive en Colab
    """
    return {
        "my_drive": "/content/drive/MyDrive",
        "other_computers": "/content/drive/Othercomputers",
        "shared_drives": "/content/drive/Shareddrives",
    }


def mount_google_drive(verbose: bool = True) -> bool:
    """
    Monta Google Drive en Colab si no está montado.

    Args:
        verbose: Si True, imprime información sobre el montaje

    Returns:
        bool: True si montó exitosamente o ya estaba montado, False si no
    """
    try:
        from google.colab import drive
        import os

        drive_path = "/content/drive"

        if not os.path.exists(drive_path):
            if verbose:
                print("Montando Google Drive...")
            drive.mount("/content/drive")
            if verbose:
                print("Google Drive montado exitosamente")
            return True
        else:
            if verbose:
                print("Google Drive ya está montado")
            return True

    except ImportError:
        if verbose:
            print("No estás en Google Colab, no se puede montar Drive")
        return False
    except Exception as e:
        if verbose:
            print(f"Error al montar Google Drive: {e}")
        return False
