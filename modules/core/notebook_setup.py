"""
Plantilla de configuración para notebooks.

Este módulo proporciona funciones simples que todos los notebooks
pueden usar para configurar su entorno automáticamente.

Uso en cualquier notebook:
    from modules.core.notebook_setup import setup_notebook
    ENV, PATHS = setup_notebook()
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Dict

from modules.core.dependency_manager import setup_notebook_environment


def add_project_to_path() -> Path:
    """
    Agrega la raíz del proyecto al sys.path.

    Busca hacia arriba desde el directorio actual hasta encontrar
    la raíz del proyecto (identificada por la carpeta modules/).

    Returns:
        Path: Ruta a la raíz del proyecto

    Raises:
        FileNotFoundError: Si no se encuentra la raíz del proyecto
    """
    current_dir = Path.cwd()

    # Buscar hacia arriba hasta encontrar la raíz
    for _ in range(10):
        if (current_dir / "modules").exists():
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            return current_dir

        parent = current_dir.parent
        if parent == current_dir:  # Llegamos a la raíz del sistema
            break
        current_dir = parent

    raise FileNotFoundError(
        "No se pudo encontrar la raíz del proyecto. "
        "Asegúrate de que la carpeta 'modules/' existe."
    )


def setup_notebook(verbose: bool = True) -> Tuple[str, Dict[str, Path]]:
    """
    Configuración completa para notebooks (path + environment + rutas).

    Esta es la función principal que deben llamar los notebooks al inicio.
    Configura:
    - Agrega la raíz del proyecto al sys.path
    - Detecta si está en Local o Colab
    - Configura las rutas del proyecto

    Args:
        verbose: Si True, muestra información detallada

    Returns:
        tuple: (environment, paths) donde:
            - environment (str): 'local' o 'colab'
            - paths (dict): Diccionario con rutas del proyecto

    Example:
        ```python
        # Al inicio de cualquier notebook
        from modules.core.notebook_setup import setup_notebook

        ENV, PATHS = setup_notebook()

        # Usar las rutas
        cache_path = PATHS['cache_original'] / "data.pkl"
        ```
    """
    # Agregar proyecto al path
    try:
        project_root = add_project_to_path()
        if verbose:
            print(f"Raíz del proyecto agregada al path: {project_root}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    # Importar aquí después de agregar al path
    from modules.core.environment import setup_environment as env_setup

    # Configurar entorno y rutas
    return env_setup(verbose=verbose)


def setup_environment(verbose: bool = True) -> bool:
    """
    Configurar el entorno para cualquier notebook.

    Esta es la función principal que deben llamar todos los notebooks
    al inicio para asegurar que el entorno esté configurado correctamente.

    Args:
        verbose: Mostrar información detallada durante la configuración

    Returns:
        bool: True si el entorno está listo, False en caso contrario

    Example:
        ```python
        # Al inicio de cualquier notebook
        from modules.core.notebook_setup import setup_environment

        if not setup_environment():
            print("Error configurando entorno")
            exit(1)
        ```
    """
    return setup_notebook_environment(auto_install=True, verbose=verbose)


def quick_setup() -> bool:
    """
    Configuración rápida sin información detallada.

    Returns:
        bool: True si el entorno está listo, False en caso contrario
    """
    return setup_notebook_environment(auto_install=True, verbose=False)


def setup():
    """Alias para setup_environment() para mayor simplicidad."""
    return setup_environment()


def setup_colab_git(
    computer_name: str = "ZenBook",
    project_dir: str = "parkinson-voice-uncertainty",
    branch: str = "main",
) -> str:
    """
    Configurar entorno de Google Colab con Drive, Git y dependencias.

    Esta función monta Google Drive, configura el repositorio Git,
    actualiza la rama especificada e instala las dependencias del
    proyecto.

    Args:
        computer_name: Nombre del computador tal como aparece en Drive
        project_dir: Nombre de la carpeta del repositorio
        branch: Rama de Git a utilizar

    Returns:
        str: Ruta completa al proyecto configurado

    Raises:
        AssertionError: Si no se encuentra el computador o proyecto
        RuntimeError: Si algún comando Git crítico falla

    Example:
        ```python
        from modules.core.notebook_setup import setup_colab_git

        # Configuración por defecto
        project_path = setup_colab_git()

        # Configuración personalizada
        project_path = setup_colab_git(
            computer_name="MiPC",
            project_dir="mi-proyecto",
            branch="dev"
        )
        ```
    """
    # Montar Google Drive
    print("Montando Google Drive...")
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except ImportError:
        raise RuntimeError(
            "Este script requiere Google Colab. "
            "Para entornos locales usa setup_environment()"
        )

    # Construir rutas
    base_path = "/content/drive/Othercomputers"
    project_path = os.path.join(base_path, computer_name, project_dir)

    # Función auxiliar para ejecutar comandos
    def _run_cmd(*args, check=False):
        """Ejecutar comando de shell con logging."""
        print("$", " ".join(args))
        result = subprocess.run(
            args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        print(result.stdout)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(args)}")
        return result.returncode

    # Verificaciones de rutas
    computer_path = os.path.join(base_path, computer_name)
    assert os.path.isdir(computer_path), (
        f"No se encuentra {computer_name} en {base_path}. "
        f"Revisa el nombre exacto en Drive."
    )
    assert os.path.isdir(project_path), (
        f"No se encuentra el repositorio en: {project_path}"
    )

    # Agregar al path de Python para imports
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
        print(f"Agregado al sys.path: {project_path}")

    # Configurar Git - marcar repositorio como seguro
    print("\nConfigurando Git...")
    _run_cmd("git", "config", "--global", "--add", "safe.directory", project_path)

    # Actualizar referencias remotas
    print("\nActualizando referencias remotas...")
    _run_cmd("git", "-C", project_path, "fetch", "--all", "--prune")

    # Mostrar rama actual
    print("\nRama actual:")
    _run_cmd("git", "-C", project_path, "branch", "--show-current")

    # Checkout de la rama especificada
    print(f"\nCambiando a rama: {branch}")
    return_code = _run_cmd("git", "-C", project_path, "checkout", branch)
    if return_code != 0:
        msg = f"Rama {branch} no existe localmente, creando desde origin/{branch}"
        print(msg)
        _run_cmd(
            "git", "-C", project_path, "checkout", "-b", branch, f"origin/{branch}"
        )

    # Actualizar contenido de la rama
    print(f"\nActualizando rama {branch}...")
    _run_cmd("git", "-C", project_path, "pull", "origin", branch)

    # Instalar dependencias
    requirements_path = os.path.join(project_path, "requirements.txt")
    if os.path.exists(requirements_path):
        print("\nInstalando dependencias...")
        current_dir = os.getcwd()
        os.chdir("/content")
        _run_cmd("python", "-m", "pip", "install", "-q", "-r", requirements_path)
        os.chdir(current_dir)
    else:
        print("\nNo se encontró requirements.txt, omitiendo instalación")

    # Cambiar al directorio del proyecto
    os.chdir(project_path)
    print(f"\nDirectorio de trabajo: {os.getcwd()}")

    # Activar autoreload
    print("\nActivando autoreload...")
    try:
        get_ipython().run_line_magic("load_ext", "autoreload")  # noqa
        get_ipython().run_line_magic("autoreload", "2")  # noqa
        print("Autoreload activo")
    except Exception as e:
        print(f"No se activó autoreload (no es grave): {e}")

    # Resumen final
    print("\n" + "=" * 70)
    print("CONFIGURACION COMPLETADA")
    print("=" * 70)
    print(f"Proyecto: {project_path}")
    print("Rama: ", end="")
    _run_cmd("git", "-C", project_path, "branch", "--show-current")
    print("=" * 70)

    return project_path
