#!/usr/bin/env python3
"""
🔍 VERIFICADOR DE MUESTREO
==========================

Script para verificar y analizar el resultado del muestreo de datos saludables.

Uso:
    python verify_sampling.py
    python verify_sampling.py --directory vowels_healthy
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def analyze_directory(directory: Path):
    """
    Analiza el contenido del directorio de salida.
    """
    if not directory.exists():
        print(f"❌ Error: No existe el directorio {directory}")
        return
    
    # Leer metadata si existe
    metadata_file = directory / "sampling_metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Contar archivos
    nsp_files = list(directory.glob("*.nsp"))
    egg_files = list(directory.glob("*.egg"))
    total_files = len(nsp_files) + len(egg_files)
    
    # Analizar nombres de archivos
    subjects = set()
    vowels = Counter()
    extensions = Counter()
    
    for file in directory.iterdir():
        if file.suffix in ['.nsp', '.egg']:
            # Parsear nombre: subject_id-vocal.extension
            parts = file.stem.split('-')
            if len(parts) >= 2:
                subject_id = parts[0]
                vowel = '-'.join(parts[1:])
                subjects.add(subject_id)
                vowels[vowel] += 1
                extensions[file.suffix[1:]] += 1
    
    # Imprimir reporte
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN DE MUESTREO")
    print("="*70)
    
    print(f"\n📁 DIRECTORIO: {directory}")
    
    if metadata:
        print(f"\n📋 METADATA:")
        print(f"   • Seed: {metadata['config']['seed']}")
        print(f"   • Target espectrogramas: {metadata['config']['target_spectrograms']}")
        print(f"   • Sujetos seleccionados: {metadata['num_subjects']}")
        print(f"   • Archivos copiados: {metadata['stats']['files_copied']}")
        print(f"   • Espectrogramas esperados: {metadata['expected_spectrograms']:.0f}")
        print(f"\n   📝 IDs de sujetos:")
        ids = metadata['subject_ids']
        for i in range(0, len(ids), 10):
            batch = ids[i:i+10]
            print(f"      {', '.join(map(str, batch))}")
    else:
        print("\n⚠️  No se encontró archivo de metadata")
    
    print(f"\n📊 ARCHIVOS ENCONTRADOS:")
    print(f"   • Total: {total_files}")
    print(f"   • .nsp: {len(nsp_files)}")
    print(f"   • .egg: {len(egg_files)}")
    
    print(f"\n👥 SUJETOS ÚNICOS: {len(subjects)}")
    if len(subjects) <= 20:
        print(f"   IDs: {', '.join(sorted(subjects, key=int))}")
    else:
        print(f"   Primeros 10: {', '.join(list(sorted(subjects, key=int))[:10])}")
    
    print(f"\n🎵 DISTRIBUCIÓN DE VOCALES:")
    for vowel, count in sorted(vowels.items()):
        print(f"   • {vowel}: {count} archivos")
    
    # Validaciones
    print(f"\n✅ VALIDACIONES:")
    
    expected_files_per_subject = 13
    expected_total = len(subjects) * expected_files_per_subject
    
    if total_files == expected_total:
        print(f"   ✓ Número de archivos correcto ({total_files})")
    else:
        print(f"   ⚠️  Número de archivos inesperado: {total_files} (esperado: {expected_total})")
    
    if len(vowels) == 13:
        print(f"   ✓ Todas las vocales presentes (13)")
    else:
        print(f"   ⚠️  Vocales incompletas: {len(vowels)} (esperado: 13)")
    
    # Verificar que todos los sujetos tienen las mismas vocales
    files_per_subject = Counter()
    for file in directory.iterdir():
        if file.suffix in ['.nsp', '.egg']:
            parts = file.stem.split('-')
            if len(parts) >= 2:
                subject_id = parts[0]
                files_per_subject[subject_id] += 1
    
    all_complete = all(count == expected_files_per_subject for count in files_per_subject.values())
    if all_complete:
        print(f"   ✓ Todos los sujetos tienen {expected_files_per_subject} archivos")
    else:
        print(f"   ⚠️  Algunos sujetos tienen archivos incompletos:")
        for subject_id, count in files_per_subject.items():
            if count != expected_files_per_subject:
                print(f"      • Sujeto {subject_id}: {count} archivos")
    
    if metadata:
        if metadata['stats']['num_errors'] == 0:
            print(f"   ✓ Sin errores reportados")
        else:
            print(f"   ⚠️  {metadata['stats']['num_errors']} errores reportados")
    
    print("\n" + "="*70)
    
    # Resumen final
    if all_complete and total_files == expected_total and len(vowels) == 13:
        print("✅ MUESTREO COMPLETADO CORRECTAMENTE")
    else:
        print("⚠️  VERIFICAR POSIBLES PROBLEMAS")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verificar resultado del muestreo de datos saludables"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        default="./vowels_healthy",
        help="Directorio a verificar (default: ./vowels_healthy)"
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    analyze_directory(directory)


if __name__ == "__main__":
    main()

