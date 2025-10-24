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
- hyperparameter_analysis: Análisis de importancia de hiperparámetros
"""

from .talos_optimization import (
    TalosOptimizer,
    get_default_search_params,
    create_talos_model_wrapper,
    evaluate_best_model,
    analyze_hyperparameter_importance,
    print_optimization_summary,
)

from .model_evaluation import ModelEvaluator, compare_models, save_model_results

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
]
