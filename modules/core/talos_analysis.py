"""
Análisis simple de resultados de Talos.

Este módulo proporciona funciones para analizar y mostrar todas las
configuraciones probadas durante la optimización de hiperparámetros.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional


def load_talos_results(results_dir: str) -> Dict:
    """
    Cargar resultados de optimización Talos.
    
    Args:
        results_dir: Directorio con resultados de Talos
        
    Returns:
        Dict con resultados cargados
    """
    results_path = Path(results_dir)
    
    # Cargar DataFrame con todas las configuraciones
    results_df = pd.read_csv(results_path / "talos_scan_results.csv")
    
    # Cargar mejores parámetros
    with open(results_path / "best_params.json", 'r') as f:
        best_params = json.load(f)
    
    return {
        "results_df": results_df,
        "best_params": best_params,
        "results_dir": results_path
    }


def analyze_all_configurations(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizar todas las configuraciones probadas.
    
    Args:
        results_df: DataFrame con resultados de Talos
        
    Returns:
        DataFrame con análisis detallado
    """
    # Crear copia para análisis
    analysis_df = results_df.copy()
    
    # Ordenar por F1-score descendente
    analysis_df = analysis_df.sort_values('f1', ascending=False).reset_index(drop=True)
    
    # Agregar ranking
    analysis_df['rank'] = range(1, len(analysis_df) + 1)
    
    # Calcular diferencias con el mejor
    best_f1 = analysis_df['f1'].iloc[0]
    analysis_df['f1_diff_from_best'] = best_f1 - analysis_df['f1']
    analysis_df['f1_percentage'] = (analysis_df['f1'] / best_f1) * 100
    
    return analysis_df


def generate_configuration_report(results_df: pd.DataFrame, best_params: Dict) -> str:
    """
    Generar reporte de texto con todas las configuraciones.
    
    Args:
        results_df: DataFrame con resultados de Talos
        best_params: Mejores parámetros encontrados
        
    Returns:
        String con reporte completo
    """
    report = []
    report.append("="*80)
    report.append("REPORTE COMPLETO DE OPTIMIZACIÓN TALOS")
    report.append("="*80)
    report.append("")
    
    # Estadísticas generales
    report.append("📊 ESTADÍSTICAS GENERALES:")
    report.append("-" * 40)
    report.append(f"Total configuraciones evaluadas: {len(results_df)}")
    report.append(f"Mejor F1-score: {results_df['f1'].max():.4f}")
    report.append(f"F1-score promedio: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    report.append(f"Mejor accuracy: {results_df['accuracy'].max():.4f}")
    report.append(f"Accuracy promedio: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    report.append("")
    
    # Mejores parámetros
    report.append("🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    report.append("-" * 40)
    for param, value in best_params.items():
        if param not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']:
            report.append(f"{param}: {value}")
    report.append("")
    
    # Top 10 configuraciones
    report.append("📈 TOP 10 CONFIGURACIONES:")
    report.append("-" * 40)
    top_10 = results_df.nlargest(10, 'f1')
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        report.append(f"{i:2d}. F1: {row['f1']:.4f} | "
                     f"Acc: {row['accuracy']:.4f} | "
                     f"Batch: {row['batch_size']} | "
                     f"LR: {row['learning_rate']} | "
                     f"Filters: {row['filters_1']}/{row['filters_2']} | "
                     f"Dense: {row['dense_units']}")
    report.append("")
    
    # Análisis por hiperparámetro
    hyperparams = [col for col in results_df.columns 
                   if col not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']]
    
    report.append("🔍 ANÁLISIS POR HIPERPARÁMETRO:")
    report.append("-" * 40)
    
    for param in hyperparams:
        if param in results_df.columns:
            unique_values = results_df[param].unique()
            report.append(f"\n{param.upper()}:")
            report.append(f"  Valores probados: {sorted(unique_values)}")
            
            # Encontrar mejor valor para este parámetro
            best_value = results_df.loc[results_df['f1'].idxmax(), param]
            report.append(f"  Mejor valor: {best_value}")
            
            # Estadísticas por valor
            if len(unique_values) <= 10:  # Solo si no hay demasiados valores únicos
                for value in sorted(unique_values):
                    subset = results_df[results_df[param] == value]
                    if len(subset) > 0:
                        report.append(f"    {value}: F1={subset['f1'].mean():.4f}±{subset['f1'].std():.4f} "
                                    f"(n={len(subset)})")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)


def create_detailed_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crear DataFrame detallado con todas las configuraciones.
    
    Args:
        results_df: DataFrame con resultados de Talos
        
    Returns:
        DataFrame detallado con análisis
    """
    # Crear copia para análisis
    detailed_df = results_df.copy()
    
    # Ordenar por F1-score descendente
    detailed_df = detailed_df.sort_values('f1', ascending=False).reset_index(drop=True)
    
    # Agregar ranking
    detailed_df['rank'] = range(1, len(detailed_df) + 1)
    
    # Calcular diferencias con el mejor
    best_f1 = detailed_df['f1'].iloc[0]
    detailed_df['f1_diff_from_best'] = best_f1 - detailed_df['f1']
    detailed_df['f1_percentage'] = (detailed_df['f1'] / best_f1) * 100
    
    # Agregar columnas de análisis
    detailed_df['performance_tier'] = pd.cut(
        detailed_df['f1'], 
        bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0], 
        labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Excelente']
    )
    
    # Reordenar columnas para mejor visualización
    metric_cols = ['rank', 'f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']
    hyperparam_cols = [col for col in detailed_df.columns 
                      if col not in metric_cols and col not in ['f1_diff_from_best', 'f1_percentage', 'performance_tier']]
    analysis_cols = ['f1_diff_from_best', 'f1_percentage', 'performance_tier']
    
    column_order = metric_cols + hyperparam_cols + analysis_cols
    detailed_df = detailed_df[column_order]
    
    return detailed_df


def save_analysis_results(
    results_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Guardar análisis completo en archivos de texto y CSV.
    
    Args:
        results_dir: Directorio con resultados de Talos
        output_dir: Directorio de salida (opcional)
        
    Returns:
        Dict con rutas de archivos generados
    """
    # Cargar resultados
    results = load_talos_results(results_dir)
    results_df = results["results_df"]
    best_params = results["best_params"]
    
    if output_dir is None:
        output_dir = Path(results_dir) / "analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    generated_files = {}
    
    # 1. Reporte completo de texto
    report_text = generate_configuration_report(results_df, best_params)
    report_path = output_dir / "complete_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    generated_files["report"] = str(report_path)
    
    # 2. DataFrame detallado
    detailed_df = create_detailed_dataframe(results_df)
    detailed_csv_path = output_dir / "detailed_configurations.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)
    generated_files["detailed_csv"] = str(detailed_csv_path)
    
    # 3. Solo top configuraciones
    top_20 = detailed_df.head(20)
    top_csv_path = output_dir / "top_20_configurations.csv"
    top_20.to_csv(top_csv_path, index=False)
    generated_files["top_20_csv"] = str(top_csv_path)
    
    # 4. Resumen estadístico
    summary_path = output_dir / "statistical_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN ESTADÍSTICO DE OPTIMIZACIÓN\n")
        f.write("="*50 + "\n\n")
        
        # Estadísticas generales
        f.write("ESTADÍSTICAS GENERALES:\n")
        f.write(f"Total configuraciones: {len(results_df)}\n")
        f.write(f"Mejor F1-score: {results_df['f1'].max():.4f}\n")
        f.write(f"F1-score promedio: {results_df['f1'].mean():.4f}\n")
        f.write(f"F1-score mediano: {results_df['f1'].median():.4f}\n")
        f.write(f"Desviación estándar: {results_df['f1'].std():.4f}\n\n")
        
        # Distribución de rendimiento
        f.write("DISTRIBUCIÓN DE RENDIMIENTO:\n")
        f.write(f"F1 > 0.9: {(results_df['f1'] > 0.9).sum()} configuraciones\n")
        f.write(f"F1 > 0.8: {(results_df['f1'] > 0.8).sum()} configuraciones\n")
        f.write(f"F1 > 0.7: {(results_df['f1'] > 0.7).sum()} configuraciones\n")
        f.write(f"F1 > 0.6: {(results_df['f1'] > 0.6).sum()} configuraciones\n\n")
        
        # Mejores parámetros
        f.write("MEJORES HIPERPARÁMETROS:\n")
        for param, value in best_params.items():
            if param not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']:
                f.write(f"{param}: {value}\n")
    
    generated_files["summary"] = str(summary_path)
    
    print(f"✅ Análisis completo guardado en: {output_dir}")
    print(f"📊 Archivos generados:")
    for name, path in generated_files.items():
        print(f"   - {name}: {path}")
    
    return generated_files


def print_quick_summary(results_dir: str):
    """
    Imprimir resumen rápido de los resultados.
    
    Args:
        results_dir: Directorio con resultados de Talos
    """
    results = load_talos_results(results_dir)
    results_df = results["results_df"]
    best_params = results["best_params"]
    
    print("="*70)
    print("RESUMEN RÁPIDO DE OPTIMIZACIÓN TALOS")
    print("="*70)
    print(f"📊 Total configuraciones evaluadas: {len(results_df)}")
    print(f"🏆 Mejor F1-score: {results_df['f1'].max():.4f}")
    print(f"📈 F1-score promedio: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"🎯 Mejor accuracy: {results_df['accuracy'].max():.4f}")
    print("")
    print("🏆 MEJORES HIPERPARÁMETROS:")
    for param, value in best_params.items():
        if param not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']:
            print(f"   {param}: {value}")
    print("")
    print("📈 TOP 5 CONFIGURACIONES:")
    top_5 = results_df.nlargest(5, 'f1')
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. F1: {row['f1']:.4f} | Acc: {row['accuracy']:.4f} | "
              f"Batch: {row['batch_size']} | LR: {row['learning_rate']}")
    print("="*70)


# Función de conveniencia para notebooks
def analyze_talos_results(results_dir: str, save_analysis: bool = True):
    """
    Función de conveniencia para analizar resultados de Talos.
    
    Args:
        results_dir: Directorio con resultados de Talos
        save_analysis: Si guardar análisis en archivos
        
    Returns:
        Dict con resultados del análisis
    """
    # Cargar resultados
    results = load_talos_results(results_dir)
    results_df = results["results_df"]
    best_params = results["best_params"]
    
    # Crear DataFrame detallado
    detailed_df = create_detailed_dataframe(results_df)
    
    # Imprimir resumen
    print_quick_summary(results_dir)
    
    if save_analysis:
        # Guardar análisis completo
        files = save_analysis_results(results_dir)
        print(f"\n💾 Análisis guardado en archivos:")
        for name, path in files.items():
            print(f"   - {name}: {path}")
    
    return {
        "detailed_df": detailed_df,
        "best_params": best_params,
        "summary_stats": {
            "total_configs": len(results_df),
            "best_f1": results_df['f1'].max(),
            "mean_f1": results_df['f1'].mean(),
            "std_f1": results_df['f1'].std(),
            "best_accuracy": results_df['accuracy'].max()
        }
    }
