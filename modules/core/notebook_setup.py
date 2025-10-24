"""
Plantilla de configuración para notebooks.

Este módulo proporciona una función simple que todos los notebooks
pueden usar para configurar su entorno automáticamente.

Uso en cualquier notebook:
    from modules.core.notebook_setup import setup_environment
    setup_environment()
"""

from modules.core.dependency_manager import setup_notebook_environment


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
            print("❌ Error configurando entorno")
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


# Función de conveniencia para importación directa
def setup():
    """Alias para setup_environment() para mayor simplicidad."""
    return setup_environment()
