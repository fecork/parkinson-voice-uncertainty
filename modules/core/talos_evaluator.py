"""
Evaluador simple de resultados de Talos.

Este módulo proporciona funciones para evaluar rápidamente:
- Qué opciones se consideraron
- Si el proceso fue correcto
- Resumen de resultados
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List


def evaluate_talos_process(results_dir: str) -> Dict:
    """
    Evaluar si el proceso de Talos fue correcto y mostrar opciones consideradas.

    Args:
        results_dir: Directorio con resultados de Talos

    Returns:
        Dict con evaluación del proceso
    """
    results_path = Path(results_dir)

    # Verificar archivos necesarios
    required_files = ["talos_scan_results.csv", "best_params.json"]
    missing_files = [f for f in required_files if not (results_path / f).exists()]

    if missing_files:
        return {
            "status": "ERROR",
            "message": f"Archivos faltantes: {missing_files}",
            "process_correct": False,
        }

    # Cargar resultados
    results_df = pd.read_csv(results_path / "talos_scan_results.csv")
    with open(results_path / "best_params.json", "r") as f:
        best_params = json.load(f)

    # Evaluar el proceso
    evaluation = {
        "status": "SUCCESS",
        "process_correct": True,
        "total_configurations": len(results_df),
        "best_f1_score": results_df["f1"].max(),
        "mean_f1_score": results_df["f1"].mean(),
        "f1_std": results_df["f1"].std(),
        "best_params": best_params,
        "hyperparameters_tested": {},
        "performance_distribution": {},
        "issues": [],
    }

    # Analizar hiperparámetros probados
    hyperparams = [
        col
        for col in results_df.columns
        if col
        not in ["f1", "accuracy", "precision", "recall", "val_loss", "train_loss"]
    ]

    for param in hyperparams:
        if param in results_df.columns:
            unique_values = sorted(results_df[param].unique())
            evaluation["hyperparameters_tested"][param] = {
                "values_tested": unique_values,
                "count": len(unique_values),
                "best_value": results_df.loc[results_df["f1"].idxmax(), param],
            }

    # Analizar distribución de rendimiento
    f1_scores = results_df["f1"]
    evaluation["performance_distribution"] = {
        "excellent": (f1_scores > 0.9).sum(),
        "good": ((f1_scores > 0.8) & (f1_scores <= 0.9)).sum(),
        "fair": ((f1_scores > 0.7) & (f1_scores <= 0.8)).sum(),
        "poor": (f1_scores <= 0.7).sum(),
    }

    # Detectar posibles problemas
    if len(results_df) < 10:
        evaluation["issues"].append("Pocas configuraciones evaluadas (< 10)")

    if results_df["f1"].std() < 0.01:
        evaluation["issues"].append(
            "Muy poca variación en F1-scores (posible problema)"
        )

    if results_df["f1"].max() < 0.5:
        evaluation["issues"].append("Mejor F1-score muy bajo (< 0.5)")

    # Verificar si hay configuraciones duplicadas
    duplicate_configs = results_df.duplicated(subset=hyperparams).sum()
    if duplicate_configs > 0:
        evaluation["issues"].append(
            f"Configuraciones duplicadas encontradas: {duplicate_configs}"
        )

    return evaluation


def print_evaluation_summary(evaluation: Dict):
    """
    Imprimir resumen de la evaluación.

    Args:
        evaluation: Resultado de evaluate_talos_process()
    """
    print("=" * 70)
    print("EVALUACION DEL PROCESO DE OPTIMIZACION TALOS")
    print("=" * 70)

    if evaluation["status"] == "ERROR":
        print(f"ERROR: {evaluation['message']}")
        return

    print(f"Estado: {evaluation['status']}")
    print(f"Total configuraciones evaluadas: {evaluation['total_configurations']}")
    print(f"Mejor F1-score: {evaluation['best_f1_score']:.4f}")
    print(
        f"F1-score promedio: {evaluation['mean_f1_score']:.4f} ± {evaluation['f1_std']:.4f}"
    )

    print(f"\nHIPERPARAMETROS PROBADOS:")
    for param, info in evaluation["hyperparameters_tested"].items():
        print(f"   {param}:")
        print(f"     - Valores probados: {info['values_tested']}")
        print(f"     - Total valores: {info['count']}")
        print(f"     - Mejor valor: {info['best_value']}")

    print(f"\nDISTRIBUCION DE RENDIMIENTO:")
    dist = evaluation["performance_distribution"]
    print(f"   Excelente (F1 > 0.9): {dist['excellent']} configuraciones")
    print(f"   Bueno (F1 0.8-0.9): {dist['good']} configuraciones")
    print(f"   Regular (F1 0.7-0.8): {dist['fair']} configuraciones")
    print(f"   Malo (F1 <= 0.7): {dist['poor']} configuraciones")

    print(f"\nMEJORES HIPERPARAMETROS:")
    for param, value in evaluation["best_params"].items():
        if param not in [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "val_loss",
            "train_loss",
        ]:
            print(f"   {param}: {value}")

    if evaluation["issues"]:
        print(f"\nPROBLEMAS DETECTADOS:")
        for issue in evaluation["issues"]:
            print(f"   - {issue}")
    else:
        print(f"\nNo se detectaron problemas en el proceso")

    print("=" * 70)


def show_all_configurations(results_dir: str, top_n: int = 20) -> pd.DataFrame:
    """
    Mostrar todas las configuraciones consideradas.

    Args:
        results_dir: Directorio con resultados de Talos
        top_n: Número de mejores configuraciones a mostrar

    Returns:
        DataFrame con configuraciones ordenadas por F1-score
    """
    results_path = Path(results_dir)
    results_df = pd.read_csv(results_path / "talos_scan_results.csv")

    # Ordenar por F1-score descendente
    sorted_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

    # Agregar ranking
    sorted_df["rank"] = range(1, len(sorted_df) + 1)

    # Mostrar solo las columnas relevantes que existen
    available_metric_cols = ["rank", "f1", "accuracy"]
    if "precision" in sorted_df.columns:
        available_metric_cols.append("precision")
    if "recall" in sorted_df.columns:
        available_metric_cols.append("recall")

    hyperparam_cols = [
        col
        for col in sorted_df.columns
        if col
        not in [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "val_loss",
            "train_loss",
            "rank",
        ]
    ]

    display_df = sorted_df[available_metric_cols + hyperparam_cols]

    print(
        f"TODAS LAS CONFIGURACIONES CONSIDERADAS (Top {min(top_n, len(display_df))}):"
    )
    print("=" * 100)

    # Mostrar top N configuraciones
    top_configs = display_df.head(top_n)
    print(top_configs.to_string(index=False))

    print(f"\nResumen:")
    print(f"   - Total configuraciones: {len(sorted_df)}")
    print(f"   - Mejor F1-score: {sorted_df['f1'].max():.4f}")
    print(f"   - F1-score promedio: {sorted_df['f1'].mean():.4f}")
    print(
        f"   - Rango F1-score: {sorted_df['f1'].min():.4f} - {sorted_df['f1'].max():.4f}"
    )

    return sorted_df


def quick_evaluate(results_dir: str):
    """
    Evaluación rápida del proceso de Talos.

    Args:
        results_dir: Directorio con resultados de Talos
    """
    # Evaluar proceso
    evaluation = evaluate_talos_process(results_dir)

    # Imprimir resumen
    print_evaluation_summary(evaluation)

    # Mostrar configuraciones
    print(f"\nCONFIGURACIONES CONSIDERADAS:")
    show_all_configurations(results_dir, top_n=10)

    return evaluation


# Función de conveniencia para notebooks
def check_talos_results(results_dir: str):
    """
    Función de conveniencia para verificar resultados de Talos.

    Args:
        results_dir: Directorio con resultados de Talos

    Returns:
        Dict con evaluación del proceso
    """
    return quick_evaluate(results_dir)
