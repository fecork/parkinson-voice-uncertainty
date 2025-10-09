#!/usr/bin/env python3
"""
🎯 MUESTREADOR DE DATOS SALUDABLES
===================================

Script para tomar una muestra aleatoria de sujetos saludables y copiar
sus archivos de vocales para balancear el dataset de Parkinson.

Propósito:
- Tomar N sujetos aleatorios de data/healthy/
- Copiar sus archivos .nsp de vocales a vowels_healthy/
- Generar cantidad similar de espectrogramas que el dataset Parkinson

Uso:
    python sample_healthy_data.py --target-spectrograms 1219
    python sample_healthy_data.py --num-subjects 10
    python sample_healthy_data.py --seed 42
"""

import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict
import json

# ============================================================================
# CONFIGURACIÓN DEFAULT
# ============================================================================
DEFAULT_SOURCE_DIR = "./data/healthy"
DEFAULT_OUTPUT_DIR = "./vowels_healthy"
DEFAULT_TARGET_SPECTROGRAMS = 1219  # Similar al dataset Parkinson con augmentation
DEFAULT_SEED = 42
DEFAULT_FILE_EXTENSION = "nsp"  # nsp o egg

# Constantes del pipeline (deben coincidir con el notebook)
AUGMENTATION_FACTOR = 10.1  # Factor de multiplicación
FILES_PER_SUBJECT = 13      # Archivos de vocales por sujeto


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_healthy_subjects(source_dir: Path) -> List[int]:
    """
    Obtiene lista de IDs de sujetos saludables disponibles.

    Returns:
        Lista de IDs de sujetos (enteros)
    """
    subjects = []
    for item in source_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            vowels_dir = item / "vowels"
            if vowels_dir.exists():
                subjects.append(int(item.name))

    return sorted(subjects)


def count_vowel_files(subject_dir: Path, extension: str = "nsp") -> int:
    """
    Cuenta archivos de vocales de un sujeto.

    Args:
        subject_dir: Directorio del sujeto
        extension: Extensión del archivo (nsp o egg)

    Returns:
        Número de archivos encontrados
    """
    vowels_dir = subject_dir / "vowels"
    if not vowels_dir.exists():
        return 0

    return len(list(vowels_dir.glob(f"*.{extension}")))


def calculate_required_subjects(
    target_spectrograms: int,
    files_per_subject: int = FILES_PER_SUBJECT,
    augmentation_factor: float = AUGMENTATION_FACTOR
) -> int:
    """
    Calcula cuántos sujetos se necesitan para target espectrogramas.

    Args:
        target_spectrograms: Número objetivo de espectrogramas
        files_per_subject: Archivos de vocales por sujeto
        augmentation_factor: Factor de multiplicación

    Returns:
        Número de sujetos necesarios
    """
    spectrograms_per_subject = files_per_subject * augmentation_factor
    required = int(target_spectrograms / spectrograms_per_subject) + 1
    return required


def sample_subjects(
    all_subjects: List[int],
    num_subjects: int,
    seed: int = DEFAULT_SEED
) -> List[int]:
    """
    Toma muestra aleatoria de sujetos.

    Args:
        all_subjects: Lista de todos los sujetos disponibles
        num_subjects: Número de sujetos a seleccionar
        seed: Semilla para reproducibilidad

    Returns:
        Lista de IDs de sujetos seleccionados
    """
    random.seed(seed)

    if num_subjects > len(all_subjects):
        msg = f"Solo hay {len(all_subjects)} sujetos, se tomarán todos"
        print(f"⚠️  {msg}")
        return all_subjects

    return sorted(random.sample(all_subjects, num_subjects))


def copy_vowel_files(
    source_dir: Path,
    output_dir: Path,
    subject_ids: List[int],
    extension: str = "nsp",
    dry_run: bool = False
) -> Dict[str, any]:
    """
    Copia archivos de vocales de los sujetos seleccionados.
    
    Args:
        source_dir: Directorio fuente (data/healthy)
        output_dir: Directorio de salida (vowels_healthy)
        subject_ids: Lista de IDs de sujetos a copiar
        extension: Extensión del archivo (nsp o egg)
        dry_run: Si True, solo simula sin copiar
        
    Returns:
        Diccionario con estadísticas de la operación
    """
    stats = {
        "subjects_processed": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "errors": [],
        "copied_files": []
    }
    
    # Crear directorio de salida
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject_id in subject_ids:
        subject_dir = source_dir / str(subject_id) / "vowels"
        
        if not subject_dir.exists():
            stats["errors"].append(f"Sujeto {subject_id}: no existe carpeta vowels")
            continue
        
        # Buscar archivos de vocales
        vowel_files = list(subject_dir.glob(f"*.{extension}"))
        
        if not vowel_files:
            stats["errors"].append(f"Sujeto {subject_id}: no se encontraron archivos .{extension}")
            continue
        
        # Copiar archivos
        for src_file in vowel_files:
            dst_file = output_dir / src_file.name
            
            try:
                if not dry_run:
                    shutil.copy2(src_file, dst_file)
                
                stats["files_copied"] += 1
                stats["copied_files"].append({
                    "subject_id": subject_id,
                    "filename": src_file.name,
                    "source": str(src_file),
                    "destination": str(dst_file)
                })
                
            except Exception as e:
                stats["errors"].append(f"Error copiando {src_file.name}: {str(e)}")
                stats["files_skipped"] += 1
        
        stats["subjects_processed"] += 1
    
    return stats


def save_metadata(
    output_dir: Path,
    subject_ids: List[int],
    stats: Dict,
    config: Dict
):
    """
    Guarda metadata de la muestra en archivo JSON.
    
    Args:
        output_dir: Directorio de salida
        subject_ids: Lista de IDs seleccionados
        stats: Estadísticas de la operación
        config: Configuración usada
    """
    metadata = {
        "config": config,
        "subject_ids": subject_ids,
        "num_subjects": len(subject_ids),
        "stats": {
            "subjects_processed": stats["subjects_processed"],
            "files_copied": stats["files_copied"],
            "files_skipped": stats["files_skipped"],
            "num_errors": len(stats["errors"])
        },
        "errors": stats["errors"],
        "expected_spectrograms": len(subject_ids) * FILES_PER_SUBJECT * AUGMENTATION_FACTOR
    }
    
    metadata_file = output_dir / "sampling_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Metadata guardada en: {metadata_file}")


def print_report(
    all_subjects: List[int],
    selected_subjects: List[int],
    stats: Dict,
    config: Dict
):
    """
    Imprime reporte detallado de la operación.
    """
    print("\n" + "="*70)
    print("📊 REPORTE DE MUESTREO DE DATOS SALUDABLES")
    print("="*70)
    
    print(f"\n📁 DIRECTORIO FUENTE: {config['source_dir']}")
    print(f"📁 DIRECTORIO SALIDA: {config['output_dir']}")
    print(f"🔧 EXTENSIÓN: .{config['extension']}")
    print(f"🎲 SEED: {config['seed']}")
    
    print(f"\n📈 SUJETOS:")
    print(f"   • Disponibles: {len(all_subjects)}")
    print(f"   • Seleccionados: {len(selected_subjects)}")
    print(f"   • Procesados: {stats['subjects_processed']}")
    
    print(f"\n📄 ARCHIVOS:")
    print(f"   • Copiados: {stats['files_copied']}")
    print(f"   • Omitidos: {stats['files_skipped']}")
    print(f"   • Esperados por sujeto: {FILES_PER_SUBJECT}")
    
    expected_specs = len(selected_subjects) * FILES_PER_SUBJECT * AUGMENTATION_FACTOR
    print(f"\n🎯 ESPECTROGRAMAS ESTIMADOS:")
    print(f"   • Con augmentation (~{AUGMENTATION_FACTOR}x): {expected_specs:.0f}")
    print(f"   • Target objetivo: {config.get('target_spectrograms', 'N/A')}")
    
    if stats["errors"]:
        print(f"\n⚠️  ERRORES ({len(stats['errors'])}):")
        for error in stats["errors"][:5]:  # Mostrar solo primeros 5
            print(f"   • {error}")
        if len(stats["errors"]) > 5:
            print(f"   • ... y {len(stats['errors']) - 5} más")
    else:
        print("\n✅ Sin errores")
    
    print("\n" + "="*70)
    
    if config.get('dry_run'):
        print("🔍 MODO DRY RUN - No se copiaron archivos realmente")
        print("="*70)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Muestreo de datos saludables para balancear dataset Parkinson",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source-dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directorio fuente con datos saludables (default: {DEFAULT_SOURCE_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--target-spectrograms",
        type=int,
        default=None,
        help=f"Número objetivo de espectrogramas (default: {DEFAULT_TARGET_SPECTROGRAMS})"
    )
    
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Número de sujetos a seleccionar (sobreescribe --target-spectrograms)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Semilla aleatoria (default: {DEFAULT_SEED})"
    )
    
    parser.add_argument(
        "--extension",
        type=str,
        choices=["nsp", "egg"],
        default=DEFAULT_FILE_EXTENSION,
        help=f"Extensión de archivos a copiar (default: {DEFAULT_FILE_EXTENSION})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin copiar archivos"
    )
    
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="Solo listar sujetos disponibles y salir"
    )
    
    args = parser.parse_args()
    
    # Convertir rutas a Path
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    # Verificar que existe el directorio fuente
    if not source_dir.exists():
        print(f"❌ Error: No existe el directorio {source_dir}")
        sys.exit(1)
    
    # Obtener todos los sujetos disponibles
    print("🔍 Buscando sujetos saludables...")
    all_subjects = get_healthy_subjects(source_dir)
    
    if not all_subjects:
        print(f"❌ Error: No se encontraron sujetos en {source_dir}")
        sys.exit(1)
    
    print(f"✅ Encontrados {len(all_subjects)} sujetos (IDs: {min(all_subjects)} - {max(all_subjects)})")
    
    # Si solo queremos listar
    if args.list_subjects:
        print(f"\n📋 SUJETOS DISPONIBLES ({len(all_subjects)}):")
        for i in range(0, len(all_subjects), 20):
            batch = all_subjects[i:i+20]
            print(f"   {', '.join(map(str, batch))}")
        sys.exit(0)
    
    # Calcular número de sujetos necesarios
    if args.num_subjects is not None:
        num_subjects = args.num_subjects
        print(f"🎯 Sujetos especificados manualmente: {num_subjects}")
    else:
        target_specs = args.target_spectrograms or DEFAULT_TARGET_SPECTROGRAMS
        num_subjects = calculate_required_subjects(target_specs)
        print(f"🎯 Target de espectrogramas: {target_specs}")
        print(f"📊 Sujetos necesarios (calculado): {num_subjects}")
    
    # Validar
    if num_subjects > len(all_subjects):
        print(f"⚠️  Solo hay {len(all_subjects)} sujetos disponibles")
        num_subjects = len(all_subjects)
    
    # Tomar muestra
    print(f"🎲 Seleccionando {num_subjects} sujetos aleatorios (seed={args.seed})...")
    selected_subjects = sample_subjects(all_subjects, num_subjects, args.seed)
    print(f"✅ Sujetos seleccionados: {selected_subjects[:10]}{'...' if len(selected_subjects) > 10 else ''}")
    
    # Copiar archivos
    if args.dry_run:
        print("\n🔍 MODO DRY RUN - Simulando copia de archivos...")
    else:
        print(f"\n📦 Copiando archivos .{args.extension} de vocales...")
    
    stats = copy_vowel_files(
        source_dir=source_dir,
        output_dir=output_dir,
        subject_ids=selected_subjects,
        extension=args.extension,
        dry_run=args.dry_run
    )
    
    # Configuración para el reporte
    config = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_spectrograms": args.target_spectrograms or DEFAULT_TARGET_SPECTROGRAMS,
        "num_subjects": num_subjects,
        "seed": args.seed,
        "extension": args.extension,
        "dry_run": args.dry_run
    }
    
    # Imprimir reporte
    print_report(all_subjects, selected_subjects, stats, config)
    
    # Guardar metadata
    if not args.dry_run:
        save_metadata(output_dir, selected_subjects, stats, config)
    
    print("\n✨ ¡Proceso completado!")
    print(f"\n💡 Próximo paso: Ejecutar el pipeline de augmentation en el notebook")
    print(f"   usando los archivos de: {output_dir}/")


if __name__ == "__main__":
    main()

