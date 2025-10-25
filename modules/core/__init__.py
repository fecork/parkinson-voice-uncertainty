"""
Core Module for Hyperparameter Optimization
===========================================
Módulo central para optimización de hiperparámetros con Talos.

Este módulo proporciona funcionalidades reutilizables para:
- Búsqueda de hiperparámetros con Talos
- Evaluación de modelos
- Análisis de resultados
- Integración con diferentes arquitecturas

Módulos:
- talos_optimization: Funciones base para optimización
- model_evaluation: Evaluación y comparación de modelos
- preprocessing: Preprocesamiento de audio
- dataset: Manejo de datasets
- visualization: Visualización de datos
"""

from .talos_optimization import (
    TalosOptimizer,
    get_default_search_params,
    create_talos_model_wrapper,
    evaluate_best_model,
    analyze_hyperparameter_importance,
    print_optimization_summary,
)

from .model_evaluation import (
    ModelEvaluator,
    compare_models,
    save_model_results,
)

# Import preprocessing module
from . import preprocessing

# Import dataset module functions
from .dataset import (
    process_dataset,
    process_dataset_with_checkpoint,
    process_dataset_parallel,
    process_dataset_parallel_with_checkpoint,
    to_pytorch_tensors,
    save_spectrograms_cache,
    load_spectrograms_cache,
    VowelSegmentsDataset,
    SampleMeta,
)

# Import visualization module functions
from .visualization import (
    visualize_audio_and_spectrograms,
    plot_spectrogram_comparison,
    plot_waveform,
    plot_mel_spectrogram,
    plot_label_distribution,
    plot_sample_spectrograms_grid,
    compare_original_vs_augmented,
    compare_audio_waveforms,
    plot_training_history,
    plot_confusion_matrix,
    create_audio_player,
    save_figure,
)

__all__ = [
    "TalosOptimizer",
    "get_default_search_params",
    "create_talos_model_wrapper",
    "evaluate_best_model",
    "analyze_hyperparameter_importance",
    "print_optimization_summary",
    "ModelEvaluator",
    "compare_models",
    "save_model_results",
    "preprocessing",
    # Dataset functions
    "process_dataset",
    "process_dataset_with_checkpoint",
    "process_dataset_parallel",
    "process_dataset_parallel_with_checkpoint",
    "to_pytorch_tensors",
    "save_spectrograms_cache",
    "load_spectrograms_cache",
    "VowelSegmentsDataset",
    "SampleMeta",
    # Visualization functions
    "visualize_audio_and_spectrograms",
    "plot_spectrogram_comparison",
    "plot_waveform",
    "plot_mel_spectrogram",
    "plot_label_distribution",
    "plot_sample_spectrograms_grid",
    "compare_original_vs_augmented",
    "compare_audio_waveforms",
    "plot_training_history",
    "plot_confusion_matrix",
    "create_audio_player",
    "save_figure",
]
