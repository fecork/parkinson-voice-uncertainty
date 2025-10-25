"""
Core Module for Hyperparameter Optimization
===========================================
Módulo central para optimización de hiperparámetros con Optuna.

Este módulo proporciona funcionalidades reutilizables para:
- Búsqueda de hiperparámetros con Optuna
- Evaluación de modelos
- Análisis de resultados
- Integración con diferentes arquitecturas

Módulos:
- optuna_optimization: Funciones base para optimización
- model_evaluation: Evaluación y comparación de modelos
- preprocessing: Preprocesamiento de audio
- dataset: Manejo de datasets
- visualization: Visualización de datos
"""

from .optuna_optimization import (
    OptunaOptimizer,
    OptunaModelWrapper,
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

# Import utility functions
from .utils import (
    create_10fold_splits_by_speaker,
)

__all__ = [
    "OptunaOptimizer",
    "OptunaModelWrapper",
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
    # Utility functions
    "create_10fold_splits_by_speaker",
]
