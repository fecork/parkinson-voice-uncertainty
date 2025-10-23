"""
Cache Utilities Module
======================
Funciones para gestionar el cache de data augmentation.

Utilidades:
    - show_cache_info: Ver información de archivos en cache
    - clear_cache: Limpiar archivos de cache
"""

import os
import glob


def show_cache_info(cache_dir: str = "./cache") -> None:
    """
    Mostrar información sobre archivos en cache.

    Args:
        cache_dir: Directorio de cache a inspeccionar

    Example:
        >>> show_cache_info("./cache")
        📁 CACHE DIRECTORY: ./cache
           Archivos: 2

           1. augmented_dataset_abc123.pkl
              • Tamaño: 8.5 MB
        ...
    """
    if not os.path.exists(cache_dir):
        print(f"⚠️  No existe el directorio de cache: {cache_dir}")
        return

    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))

    if not cache_files:
        print(f"📁 Cache vacío: {cache_dir}")
        return

    print(f"📁 CACHE DIRECTORY: {cache_dir}")
    print(f"   Archivos: {len(cache_files)}\n")

    total_size = 0
    for i, cf in enumerate(cache_files, 1):
        size_mb = os.path.getsize(cf) / (1024 * 1024)
        total_size += size_mb
        print(f"   {i}. {os.path.basename(cf)}")
        print(f"      • Tamaño: {size_mb:.1f} MB")

    print(f"\n   📊 Total: {total_size:.1f} MB")


def clear_cache(cache_dir: str = "./cache", confirm: bool = True) -> None:
    """
    Limpiar todos los archivos de cache.

    Args:
        cache_dir: Directorio de cache a limpiar
        confirm: Si True, solo muestra advertencia. Si False, ejecuta limpieza.

    Example:
        >>> clear_cache("./cache", confirm=False)
        🗑️  Eliminado: augmented_dataset_abc123.pkl
        ✅ Cache limpiado: 1 archivos eliminados

    Warning:
        Esta operación es irreversible. Los archivos eliminados no se pueden
        recuperar. Se regenerarán automáticamente en la siguiente ejecución
        con data augmentation.
    """
    if confirm:
        print("⚠️  Esta acción eliminará todos los archivos de cache.")
        print("   Para limpiar cache ejecuta: clear_cache(confirm=False)")
        return

    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))

    if not cache_files:
        print("✅ Cache ya está vacío")
        return

    for cf in cache_files:
        try:
            os.remove(cf)
            print(f"   🗑️  Eliminado: {os.path.basename(cf)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"✅ Cache limpiado: {len(cache_files)} archivos eliminados")


def get_cache_stats(cache_dir: str = "./cache") -> dict:
    """
    Obtener estadísticas del cache sin imprimir.

    Args:
        cache_dir: Directorio de cache a inspeccionar

    Returns:
        dict: Diccionario con estadísticas del cache:
            - exists (bool): Si el directorio existe
            - num_files (int): Número de archivos
            - total_size_mb (float): Tamaño total en MB
            - files (list): Lista de archivos con sus tamaños

    Example:
        >>> stats = get_cache_stats("./cache")
        >>> print(f"Cache tiene {stats['num_files']} archivos")
    """
    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "num_files": 0,
            "total_size_mb": 0.0,
            "files": [],
        }

    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
    files_info = []
    total_size = 0

    for cf in cache_files:
        size_mb = os.path.getsize(cf) / (1024 * 1024)
        total_size += size_mb
        files_info.append({"name": os.path.basename(cf), "size_mb": size_mb})

    return {
        "exists": True,
        "num_files": len(cache_files),
        "total_size_mb": round(total_size, 2),
        "files": files_info,
    }


def print_cache_config(cache_dir: str, use_cache: bool, force_regenerate: bool) -> None:
    """
    Imprimir configuración actual del cache de forma visual.

    Args:
        cache_dir: Directorio de cache configurado
        use_cache: Si el cache está activado
        force_regenerate: Si se forzará regeneración

    Example:
        >>> print_cache_config("./cache", True, False)
        💾 CONFIGURACIÓN DE CACHE:
           • Estado: ✅ ACTIVADO
           • Directorio: ./cache
           • Regenerar: ❌ NO
    """
    print("💾 CONFIGURACIÓN DE CACHE:")
    status = "✅ ACTIVADO" if use_cache else "❌ DESACTIVADO"
    print(f"   • Estado: {status}")
    print(f"   • Directorio: {cache_dir}")
    regen = "⚠️  SÍ (ignorará cache)" if force_regenerate else "❌ NO"
    print(f"   • Regenerar: {regen}")

    if use_cache and os.path.exists(cache_dir):
        stats = get_cache_stats(cache_dir)
        if stats["num_files"] > 0:
            num = stats["num_files"]
            size = stats["total_size_mb"]
            print(f"   • Archivos existentes: {num} ({size:.1f} MB)")
        else:
            print("   • Cache vacío (se generará en primera ejecución)")
