"""
Módulo de visualización para resultados de optimización con Talos.

Este módulo proporciona funciones para visualizar:
- Todas las configuraciones probadas
- Comparación de hiperparámetros
- Análisis de importancia de parámetros
- Gráficas de evolución de métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    
    # Cargar resumen de optimización
    summary_path = results_path / "optimization_summary.txt"
    summary_text = ""
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_text = f.read()
    
    return {
        "results_df": results_df,
        "best_params": best_params,
        "summary": summary_text,
        "results_dir": results_path
    }


def plot_hyperparameter_importance(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    top_n: int = 10
) -> plt.Figure:
    """
    Visualizar importancia de hiperparámetros.
    
    Args:
        results_df: DataFrame con resultados de Talos
        save_path: Ruta para guardar la gráfica
        top_n: Número de parámetros más importantes a mostrar
        
    Returns:
        Figura de matplotlib
    """
    # Calcular correlaciones con F1-score
    hyperparams = [col for col in results_df.columns 
                   if col not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']]
    
    correlations = {}
    for param in hyperparams:
        if results_df[param].dtype in ['object', 'category']:
            # Para parámetros categóricos, usar correlación de Spearman
            corr = results_df[param].astype('category').cat.codes.corr(results_df['f1'])
        else:
            # Para parámetros numéricos, usar correlación de Pearson
            corr = results_df[param].corr(results_df['f1'])
        correlations[param] = abs(corr)
    
    # Ordenar por importancia
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    params, importances = zip(*sorted_params)
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    
    bars = ax.barh(params, importances, color=colors)
    ax.set_xlabel('Importancia (|Correlación con F1-score|)')
    ax.set_title('Importancia de Hiperparámetros en Optimización Talos')
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, importance in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_hyperparameter_combinations(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualizar combinaciones de hiperparámetros más importantes.
    
    Args:
        results_df: DataFrame con resultados de Talos
        save_path: Ruta para guardar la gráfica
        
    Returns:
        Figura de matplotlib
    """
    # Seleccionar top 10 configuraciones
    top_10 = results_df.nlargest(10, 'f1')
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Top 10 Configuraciones de Hiperparámetros', fontsize=16)
    
    # 1. F1-score vs Batch Size
    axes[0, 0].scatter(top_10['batch_size'], top_10['f1'], 
                      c=top_10['f1'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('F1-Score vs Batch Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1-score vs Learning Rate
    axes[0, 1].scatter(top_10['learning_rate'], top_10['f1'], 
                      c=top_10['f1'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score vs Learning Rate')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1-score vs Filters
    axes[1, 0].scatter(top_10['filters_1'], top_10['f1'], 
                      c=top_10['f1'], cmap='viridis', s=100, alpha=0.7, label='Filters 1')
    axes[1, 0].scatter(top_10['filters_2'], top_10['f1'], 
                      c=top_10['f1'], cmap='plasma', s=100, alpha=0.7, label='Filters 2')
    axes[1, 0].set_xlabel('Number of Filters')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score vs Number of Filters')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. F1-score vs Dense Units
    axes[1, 1].scatter(top_10['dense_units'], top_10['f1'], 
                      c=top_10['f1'], cmap='viridis', s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Dense Units')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score vs Dense Units')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_optimization_evolution(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualizar evolución de la optimización.
    
    Args:
        results_df: DataFrame con resultados de Talos
        save_path: Ruta para guardar la gráfica
        
    Returns:
        Figura de matplotlib
    """
    # Ordenar por orden de evaluación (asumiendo que están en orden cronológico)
    results_sorted = results_df.sort_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evolución de la Optimización de Hiperparámetros', fontsize=16)
    
    # 1. F1-score a lo largo del tiempo
    axes[0, 0].plot(results_sorted.index, results_sorted['f1'], 'b-', alpha=0.7, label='F1-Score')
    axes[0, 0].plot(results_sorted.index, results_sorted['f1'].cummax(), 'r-', linewidth=2, label='Mejor F1-Score')
    axes[0, 0].set_xlabel('Configuración #')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('Evolución del F1-Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy a lo largo del tiempo
    axes[0, 1].plot(results_sorted.index, results_sorted['accuracy'], 'g-', alpha=0.7, label='Accuracy')
    axes[0, 1].plot(results_sorted.index, results_sorted['accuracy'].cummax(), 'r-', linewidth=2, label='Mejor Accuracy')
    axes[0, 1].set_xlabel('Configuración #')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Evolución de la Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución de F1-scores
    axes[1, 0].hist(results_sorted['f1'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(results_sorted['f1'].max(), color='red', linestyle='--', linewidth=2, label='Mejor F1-Score')
    axes[1, 0].axvline(results_sorted['f1'].mean(), color='orange', linestyle='--', linewidth=2, label='F1-Score Promedio')
    axes[1, 0].set_xlabel('F1-Score')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de F1-Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Métricas vs Pérdida
    scatter = axes[1, 1].scatter(results_sorted['val_loss'], results_sorted['f1'], 
                                c=results_sorted['f1'], cmap='viridis', s=50, alpha=0.7)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score vs Validation Loss')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='F1-Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_dashboard(
    results_df: pd.DataFrame,
    best_params: Dict,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Crear dashboard interactivo con Plotly.
    
    Args:
        results_df: DataFrame con resultados de Talos
        best_params: Mejores parámetros encontrados
        save_path: Ruta para guardar el HTML
        
    Returns:
        Figura de Plotly
    """
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('F1-Score vs Batch Size', 'F1-Score vs Learning Rate',
                       'F1-Score vs Filters', 'Distribución de F1-Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. F1-Score vs Batch Size
    fig.add_trace(
        go.Scatter(x=results_df['batch_size'], y=results_df['f1'],
                  mode='markers', name='Configuraciones',
                  marker=dict(size=8, color=results_df['f1'], 
                             colorscale='Viridis', showscale=False)),
        row=1, col=1
    )
    
    # 2. F1-Score vs Learning Rate
    fig.add_trace(
        go.Scatter(x=results_df['learning_rate'], y=results_df['f1'],
                  mode='markers', name='Configuraciones',
                  marker=dict(size=8, color=results_df['f1'], 
                             colorscale='Viridis', showscale=False)),
        row=1, col=2
    )
    
    # 3. F1-Score vs Filters
    fig.add_trace(
        go.Scatter(x=results_df['filters_1'], y=results_df['f1'],
                  mode='markers', name='Filters 1',
                  marker=dict(size=8, color=results_df['f1'], 
                             colorscale='Viridis', showscale=False)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=results_df['filters_2'], y=results_df['f1'],
                  mode='markers', name='Filters 2',
                  marker=dict(size=8, color=results_df['f1'], 
                             colorscale='Plasma', showscale=False)),
        row=2, col=1
    )
    
    # 4. Histograma de F1-Scores
    fig.add_trace(
        go.Histogram(x=results_df['f1'], name='Distribución F1-Score',
                    marker_color='skyblue'),
        row=2, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title=f'Dashboard de Optimización Talos - Mejor F1-Score: {results_df["f1"].max():.4f}',
        showlegend=True,
        height=800
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Batch Size", row=1, col=1)
    fig.update_yaxes(title_text="F1-Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Learning Rate", row=1, col=2, type="log")
    fig.update_yaxes(title_text="F1-Score", row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Filters", row=2, col=1)
    fig.update_yaxes(title_text="F1-Score", row=2, col=1)
    
    fig.update_xaxes(title_text="F1-Score", row=2, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def generate_comprehensive_report(
    results_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Generar reporte completo de visualizaciones.
    
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
        output_dir = Path(results_dir) / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    generated_files = {}
    
    # 1. Importancia de hiperparámetros
    fig1 = plot_hyperparameter_importance(results_df)
    importance_path = output_dir / "hyperparameter_importance.png"
    fig1.savefig(importance_path, dpi=300, bbox_inches='tight')
    generated_files["importance"] = str(importance_path)
    plt.close(fig1)
    
    # 2. Combinaciones de hiperparámetros
    fig2 = plot_hyperparameter_combinations(results_df)
    combinations_path = output_dir / "hyperparameter_combinations.png"
    fig2.savefig(combinations_path, dpi=300, bbox_inches='tight')
    generated_files["combinations"] = str(combinations_path)
    plt.close(fig2)
    
    # 3. Evolución de optimización
    fig3 = plot_optimization_evolution(results_df)
    evolution_path = output_dir / "optimization_evolution.png"
    fig3.savefig(evolution_path, dpi=300, bbox_inches='tight')
    generated_files["evolution"] = str(evolution_path)
    plt.close(fig3)
    
    # 4. Dashboard interactivo
    dashboard = create_interactive_dashboard(results_df, best_params)
    dashboard_path = output_dir / "interactive_dashboard.html"
    dashboard.write_html(dashboard_path)
    generated_files["dashboard"] = str(dashboard_path)
    
    # 5. Resumen estadístico
    summary_path = output_dir / "optimization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("RESUMEN DE OPTIMIZACIÓN TALOS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total configuraciones evaluadas: {len(results_df)}\n")
        f.write(f"Mejor F1-score: {results_df['f1'].max():.4f}\n")
        f.write(f"F1-score promedio: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}\n")
        f.write(f"Mejor accuracy: {results_df['accuracy'].max():.4f}\n")
        f.write(f"Accuracy promedio: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}\n\n")
        
        f.write("MEJORES HIPERPARÁMETROS:\n")
        f.write("-"*30 + "\n")
        for param, value in best_params.items():
            if param not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']:
                f.write(f"{param}: {value}\n")
        
        f.write("\nTOP 5 CONFIGURACIONES:\n")
        f.write("-"*30 + "\n")
        top_5 = results_df.nlargest(5, 'f1')
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{i}. F1: {row['f1']:.4f} | Acc: {row['accuracy']:.4f} | "
                   f"Batch: {row['batch_size']} | LR: {row['learning_rate']}\n")
    
    generated_files["summary"] = str(summary_path)
    
    print(f"✅ Reporte completo generado en: {output_dir}")
    print(f"📊 Archivos generados:")
    for name, path in generated_files.items():
        print(f"   - {name}: {path}")
    
    return generated_files


# Función de conveniencia para notebooks
def visualize_talos_results(results_dir: str):
    """
    Función de conveniencia para visualizar resultados de Talos.
    
    Args:
        results_dir: Directorio con resultados de Talos
    """
    return generate_comprehensive_report(results_dir)
