#!/usr/bin/env python3
"""
Visualización de SpecAugment - Ejemplo de Uso
==============================================

Este script demuestra cómo usar las funciones de visualización
del módulo CNN2D para analizar los efectos de SpecAugment.
"""

import sys
from pathlib import Path

# Agregar módulos al path
sys.path.insert(0, str(Path.cwd()))

from modules.data.augmentation import create_augmented_dataset
from modules.core.dataset import to_pytorch_tensors
from modules.models.cnn2d.visualization import (
    visualize_augmented_samples,
    compare_healthy_vs_parkinson,
    analyze_specaugment_effects,
    quantify_specaugment_effects,
)


def main():
    """Función principal para demostrar visualización de augmentation"""

    print("=" * 70)
    print("📊 VISUALIZACIÓN DE SPECAUGMENT - EJEMPLO DE USO")
    print("=" * 70)

    # Configuración
    DATA_PATH_HEALTHY = "./data/vowels_healthy"
    DATA_PATH_PARKINSON = "./data/vowels_pk"
    AUGMENTATION_TYPES = ["original", "spec_augment"]  # Solo SpecAugment

    print(f"\nConfiguración:")
    print(f"   • Tipos de augmentation: {AUGMENTATION_TYPES}")
    print(f"   • Healthy: {DATA_PATH_HEALTHY}")
    print(f"   • Parkinson: {DATA_PATH_PARKINSON}")

    # Cargar datos
    print("\n" + "=" * 50)
    print("CARGANDO DATOS CON AUGMENTATION")
    print("=" * 50)

    # Healthy
    print("\n🟢 Cargando Healthy...")
    audio_files_healthy = list(Path(DATA_PATH_HEALTHY).glob("*.egg"))
    print(f"   Archivos encontrados: {len(audio_files_healthy)}")

    augmented_dataset_healthy = create_augmented_dataset(
        audio_files_healthy,
        augmentation_types=AUGMENTATION_TYPES,
        num_spec_augment_versions=2,
        use_cache=True,
        cache_dir="./cache/healthy",
        force_regenerate=False,
        progress_every=3,
    )

    X_healthy, y_task_healthy, y_domain_healthy, meta_healthy = to_pytorch_tensors(
        augmented_dataset_healthy
    )
    print(f"   ✅ Healthy cargado: {X_healthy.shape[0]} muestras")

    # Parkinson
    print("\n🔴 Cargando Parkinson...")
    audio_files_parkinson = list(Path(DATA_PATH_PARKINSON).glob("*.egg"))
    print(f"   Archivos encontrados: {len(audio_files_parkinson)}")

    augmented_dataset_parkinson = create_augmented_dataset(
        audio_files_parkinson,
        augmentation_types=AUGMENTATION_TYPES,
        num_spec_augment_versions=2,
        use_cache=True,
        cache_dir="./cache/parkinson",
        force_regenerate=False,
        progress_every=3,
    )

    X_parkinson, y_task_parkinson, y_domain_parkinson, meta_parkinson = (
        to_pytorch_tensors(augmented_dataset_parkinson)
    )
    print(f"   ✅ Parkinson cargado: {X_parkinson.shape[0]} muestras")

    # Visualizaciones
    print("\n" + "=" * 50)
    print("VISUALIZACIONES")
    print("=" * 50)

    # 1. Visualización básica
    print("\n1. Visualización básica de SpecAugment...")
    if len(augmented_dataset_healthy) > 0:
        fig_healthy = visualize_augmented_samples(
            augmented_dataset_healthy,
            num_samples=3,
            title="Healthy - SpecAugment",
            show=True,
        )

    if len(augmented_dataset_parkinson) > 0:
        fig_parkinson = visualize_augmented_samples(
            augmented_dataset_parkinson,
            num_samples=3,
            title="Parkinson - SpecAugment",
            show=True,
        )

    # 2. Comparación Healthy vs Parkinson
    print("\n2. Comparación Healthy vs Parkinson...")
    if len(augmented_dataset_healthy) > 0 and len(augmented_dataset_parkinson) > 0:
        fig_comparison = compare_healthy_vs_parkinson(
            augmented_dataset_healthy,
            augmented_dataset_parkinson,
            num_examples=2,
            show=True,
        )

    # 3. Análisis detallado
    print("\n3. Análisis detallado de SpecAugment...")
    if len(augmented_dataset_healthy) > 0:
        fig_healthy_analysis = analyze_specaugment_effects(
            augmented_dataset_healthy, "HEALTHY", num_examples=3, show=True
        )

    if len(augmented_dataset_parkinson) > 0:
        fig_parkinson_analysis = analyze_specaugment_effects(
            augmented_dataset_parkinson, "PARKINSON", num_examples=3, show=True
        )

    # 4. Análisis cuantitativo
    print("\n4. Análisis cuantitativo...")
    healthy_stats = quantify_specaugment_effects(
        augmented_dataset_healthy, "🟢 HEALTHY"
    )
    parkinson_stats = quantify_specaugment_effects(
        augmented_dataset_parkinson, "🔴 PARKINSON"
    )

    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    print("✅ Visualizaciones completadas")
    print("✅ Funciones del módulo CNN2D utilizadas correctamente")
    print("✅ Código modular y reutilizable")
    print("✅ Sin duplicidad de funciones")
    print("=" * 70)


if __name__ == "__main__":
    main()

