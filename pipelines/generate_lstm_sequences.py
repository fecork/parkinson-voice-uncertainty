#!/usr/bin/env python3
"""
Script de Generación de Secuencias LSTM
=======================================
Genera secuencias de espectrogramas desde cache original para LSTM.
Procesa ambos datasets (healthy y parkinson) y guarda en cache/sequences/.

Uso:
    python pipelines/generate_lstm_sequences.py --n_frames 7
    python pipelines/generate_lstm_sequences.py --all_frames
"""

import argparse
import sys
from pathlib import Path

# Agregar directorio raíz al path para importar modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos propios
from modules.core.dataset import load_spectrograms_cache
from modules.core.sequence_dataset import (
    group_spectrograms_to_sequences,
    save_sequence_cache,
    print_sequence_stats,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Generar secuencias LSTM desde cache original"
    )

    # Datos
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directorio base de cache",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/sequences",
        help="Directorio de salida para secuencias",
    )

    # Configuración de secuencias
    parser.add_argument(
        "--n_frames",
        type=int,
        default=7,
        help="Número de frames por secuencia (default: 7)",
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Generar secuencias para todos los valores [3, 5, 7, 9]",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=3,
        help="Número mínimo de frames (descartar si menos)",
    )

    # Procesamiento
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Re-normalizar secuencias (por defecto usa normalización aplicada)",
    )

    return parser.parse_args()


# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================


def generate_sequences_for_dataset(
    dataset_name: str,
    cache_path: str,
    n_frames: int,
    min_frames: int,
    normalize: bool,
    output_dir: Path,
) -> None:
    """
    Genera secuencias para un dataset específico.

    Args:
        dataset_name: Nombre del dataset ('healthy' o 'parkinson')
        cache_path: Path al cache original
        n_frames: Número de frames por secuencia
        min_frames: Mínimo de frames requeridos
        normalize: Si re-normalizar secuencias
        output_dir: Directorio de salida
    """
    print("\n" + "=" * 70)
    print(f"PROCESANDO {dataset_name.upper()} - {n_frames} FRAMES")
    print("=" * 70)

    # Cargar dataset desde cache original
    print(f"\n📁 Cargando {dataset_name} desde cache original...")
    print(f"   Cache: {cache_path}")

    dataset = load_spectrograms_cache(cache_path)
    if dataset is None:
        print(f"❌ ERROR: No se pudo cargar {cache_path}")
        return

    print(f"✅ Dataset cargado: {len(dataset)} espectrogramas")

    # Generar secuencias
    print(f"\n🔄 Generando secuencias de {n_frames} frames...")
    sequences, lengths, metadata = group_spectrograms_to_sequences(
        dataset=dataset,
        n_frames=n_frames,
        min_frames=min_frames,
        normalize=normalize,
    )

    # Guardar cache de secuencias
    output_filename = f"{dataset_name}_n{n_frames}.pkl"
    output_path = output_dir / output_filename

    save_sequence_cache(
        sequences=sequences,
        lengths=lengths,
        metadata=metadata,
        cache_path=str(output_path),
    )

    # Imprimir estadísticas
    print_sequence_stats(
        sequences=sequences,
        lengths=lengths,
        label=f"{dataset_name.capitalize()} n={n_frames}",
    )

    print(f"\n✅ Secuencias guardadas en: {output_path}")


def generate_all_frame_sizes(
    cache_dir: str,
    output_dir: Path,
    min_frames: int,
    normalize: bool,
) -> None:
    """Genera secuencias para todos los valores de n_frames."""
    frame_sizes = [3, 5, 7, 9]

    print("\n🚀 GENERANDO SECUENCIAS PARA TODOS LOS TAMAÑOS")
    print(f"   Frame sizes: {frame_sizes}")
    print(f"   Output dir: {output_dir}")

    for n_frames in frame_sizes:
        # Procesar healthy
        healthy_cache = Path(cache_dir) / "original" / "healthy_ibarra.pkl"
        if healthy_cache.exists():
            generate_sequences_for_dataset(
                dataset_name="healthy",
                cache_path=str(healthy_cache),
                n_frames=n_frames,
                min_frames=min_frames,
                normalize=normalize,
                output_dir=output_dir,
            )
        else:
            print(f"⚠️  Cache no encontrado: {healthy_cache}")

        # Procesar parkinson
        parkinson_cache = Path(cache_dir) / "original" / "parkinson_ibarra.pkl"
        if parkinson_cache.exists():
            generate_sequences_for_dataset(
                dataset_name="parkinson",
                cache_path=str(parkinson_cache),
                n_frames=n_frames,
                min_frames=min_frames,
                normalize=normalize,
                output_dir=output_dir,
            )
        else:
            print(f"⚠️  Cache no encontrado: {parkinson_cache}")


# ============================================================
# MAIN
# ============================================================


def main():
    """Pipeline principal de generación de secuencias."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("GENERACIÓN DE SECUENCIAS LSTM DESDE CACHE ORIGINAL")
    print("=" * 70)
    print("\n🔧 Configuración:")
    print(f"  - Cache dir: {args.cache_dir}")
    print(f"  - Output dir: {args.output_dir}")
    print(f"  - Min frames: {args.min_frames}")
    print(f"  - Normalize: {args.normalize}")

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_frames:
        print(f"  - Modo: TODOS los frame sizes [3, 5, 7, 9]")
        generate_all_frame_sizes(
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            min_frames=args.min_frames,
            normalize=args.normalize,
        )
    else:
        print(f"  - Modo: UN frame size ({args.n_frames})")

        # Procesar healthy
        healthy_cache = Path(args.cache_dir) / "original" / "healthy_ibarra.pkl"
        if healthy_cache.exists():
            generate_sequences_for_dataset(
                dataset_name="healthy",
                cache_path=str(healthy_cache),
                n_frames=args.n_frames,
                min_frames=args.min_frames,
                normalize=args.normalize,
                output_dir=output_dir,
            )
        else:
            print(f"❌ ERROR: Cache no encontrado: {healthy_cache}")
            sys.exit(1)

        # Procesar parkinson
        parkinson_cache = Path(args.cache_dir) / "original" / "parkinson_ibarra.pkl"
        if parkinson_cache.exists():
            generate_sequences_for_dataset(
                dataset_name="parkinson",
                cache_path=str(parkinson_cache),
                n_frames=args.n_frames,
                min_frames=args.min_frames,
                normalize=args.normalize,
                output_dir=output_dir,
            )
        else:
            print(f"❌ ERROR: Cache no encontrado: {parkinson_cache}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ GENERACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)

    # Listar archivos generados
    print(f"\n📁 Archivos generados en {output_dir}:")
    for file_path in sorted(output_dir.glob("*.pkl")):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   - {file_path.name} ({size_mb:.1f} MB)")

    print("\n💡 Próximos pasos:")
    print("   1. Entrenar modelo: abrir lstm_da_training.ipynb")
    print("   2. K-fold completo: python pipelines/train_lstm_da_kfold.py")
    print("   3. Verificar secuencias: revisar estadísticas arriba")


if __name__ == "__main__":
    main()
