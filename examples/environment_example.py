"""
Ejemplo de uso del módulo de detección de entorno.

Este script demuestra cómo usar el módulo environment para hacer
que tu código funcione tanto en local como en Google Colab sin cambios.

El módulo automáticamente:
- Detecta si está corriendo en Local o Colab
- Configura las rutas apropiadas
- Proporciona acceso consistente a directorios del proyecto
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core.environment import (
    setup_environment,
    detect_environment,
    mount_google_drive,
)


def ejemplo_basico():
    """Ejemplo básico: configuración automática."""
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Configuración Automática")
    print("=" * 70)

    # Configurar entorno automáticamente
    ENV, PATHS = setup_environment(verbose=True)

    print(f"\nEl código detectó que está corriendo en: {ENV}")
    print(f"Ruta base del proyecto: {PATHS['base']}")


def ejemplo_uso_rutas():
    """Ejemplo: uso de rutas dinámicas en código."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Uso de Rutas Dinámicas")
    print("=" * 70)

    # Configurar entorno
    ENV, PATHS = setup_environment(verbose=False)

    # Usar rutas en tu código
    cache_healthy = PATHS["cache_original"] / "healthy_ibarra.pkl"
    cache_parkinson = PATHS["cache_augmented"] / "augmented_dataset.pkl"
    results_dir = PATHS["results"] / "mi_experimento"

    print(f"\nRutas configuradas para {ENV.upper()}:")
    print(f"  Cache healthy: {cache_healthy}")
    print(f"  Cache parkinson: {cache_parkinson}")
    print(f"  Directorio de resultados: {results_dir}")

    # Verificar si los archivos existen
    if cache_healthy.exists():
        print("\n  Cache healthy encontrado!")
    else:
        print("\n  Cache healthy no existe (normal en primer uso)")


def ejemplo_crear_directorios():
    """Ejemplo: crear directorios automáticamente."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Crear Directorios Automáticamente")
    print("=" * 70)

    ENV, PATHS = setup_environment(verbose=False)

    # Crear directorio de resultados
    mi_experimento = PATHS["results"] / "ejemplo_experimento"
    mi_experimento.mkdir(parents=True, exist_ok=True)

    print(f"\nDirectorio creado: {mi_experimento}")
    print(f"Existe: {mi_experimento.exists()}")


def ejemplo_colab_especifico():
    """Ejemplo: código específico para Colab."""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Funcionalidad Específica de Colab")
    print("=" * 70)

    # Detectar entorno
    env = detect_environment()

    if env == "colab":
        print("\nEstamos en Colab - montando Google Drive...")
        if mount_google_drive(verbose=True):
            ENV, PATHS = setup_environment(verbose=False)
            print("Google Drive montado y rutas configuradas")
        else:
            print("Error al montar Google Drive")
    else:
        print("\nEstamos en local - no es necesario montar Drive")
        ENV, PATHS = setup_environment(verbose=False)
        print("Rutas locales configuradas")


def ejemplo_ruta_personalizada():
    """Ejemplo: usar ruta base personalizada en Colab."""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Ruta Base Personalizada en Colab")
    print("=" * 70)

    from modules.core.environment import get_project_paths

    # Simular diferentes ubicaciones en Colab
    custom_paths = get_project_paths(
        "colab", colab_base="/content/drive/MyDrive/mi_proyecto_personalizado"
    )

    print("\nRutas con base personalizada:")
    print(f"  Base: {custom_paths['base']}")
    print(f"  Cache: {custom_paths['cache_original']}")
    print(f"  Results: {custom_paths['results']}")


def ejemplo_notebook():
    """Ejemplo: código típico en un notebook."""
    print("\n" + "=" * 70)
    print("EJEMPLO 6: Código Típico en Notebook")
    print("=" * 70)

    # En tu notebook, simplemente haz esto al inicio:
    from modules.core.environment import setup_environment

    ENV, PATHS = setup_environment(verbose=True)

    # Luego usa PATHS en todo tu código
    print("\nEjemplo de código en notebook:")
    print("```python")
    print("# Cargar datos")
    print("cache_path = PATHS['cache_original'] / 'healthy_ibarra.pkl'")
    print("data = load_data(str(cache_path))")
    print("")
    print("# Guardar resultados")
    print("save_dir = PATHS['results'] / 'mi_modelo'")
    print("save_dir.mkdir(parents=True, exist_ok=True)")
    print("model.save(save_dir / 'model.pth')")
    print("```")


def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "=" * 70)
    print("EJEMPLOS DE USO: modules.core.environment")
    print("=" * 70)
    print("\nEste módulo te permite escribir código que funciona tanto")
    print("en local como en Google Colab sin necesidad de cambios.")

    try:
        ejemplo_basico()
        ejemplo_uso_rutas()
        ejemplo_crear_directorios()
        ejemplo_colab_especifico()
        ejemplo_ruta_personalizada()
        ejemplo_notebook()

        print("\n" + "=" * 70)
        print("EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("=" * 70)
        print("\nPara usar en tus notebooks:")
        print("  from modules.core.environment import setup_environment")
        print("  ENV, PATHS = setup_environment(verbose=True)")

    except Exception as e:
        print(f"\nError ejecutando ejemplos: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
