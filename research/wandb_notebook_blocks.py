#!/usr/bin/env python3
"""
Bloques de código para el notebook con Weights & Biases
======================================================

Códigos listos para copiar y pegar en el notebook de CNN2D.
"""

# ============================================================
# BLOQUE 1: CONFIGURACIÓN DE WEIGHTS & BIASES
# ============================================================
WANDB_CONFIGURATION_BLOCK = """
# ============================================================
# CONFIGURACIÓN DE WEIGHTS & BIASES
# ============================================================

print("="*70)
print("CONFIGURANDO WEIGHTS & BIASES")
print("="*70)

# Importar configuración centralizada
from modules.core.experiment_config import WANDB_CONFIG, TRAINING_MONITOR_CONFIG

# Configuración del experimento actual
EXPERIMENT_CONFIG = {
    "experiment_name": "cnn2d_optuna_final_training",
    "use_wandb": True,
    "plot_every": 5,  # Cada 5 épocas
    "save_plots": True,
    "model_architecture": "CNN2D",
    "dataset": "Parkinson Voice",
    "optimization": "Optuna",
    "best_params": best_params  # Se define después de Optuna
}

print(f"✅ Configuración de wandb:")
print(f"   - Proyecto: {WANDB_CONFIG['project_name']}")
print(f"   - Experimento: {EXPERIMENT_CONFIG['experiment_name']}")
print(f"   - API Key: {'*' * 20}...{WANDB_CONFIG['api_key'][-4:]}")
print(f"   - Tags: {WANDB_CONFIG['tags']}")
print(f"   - Monitoreo cada: {EXPERIMENT_CONFIG['plot_every']} épocas")
print("="*70)
"""

# ============================================================
# BLOQUE 2: IMPORTAR MÓDULOS DE MONITOREO
# ============================================================
WANDB_IMPORTS_BLOCK = """
# ============================================================
# IMPORTAR MÓDULOS DE MONITOREO
# ============================================================

# Importar módulos de monitoreo
from modules.core.training_monitor import create_training_monitor, test_wandb_connection
from modules.core.experiment_config import WANDB_CONFIG, TRAINING_MONITOR_CONFIG

# Probar conexión con wandb
print("🔗 Probando conexión con Weights & Biases...")
connection_success = test_wandb_connection(WANDB_CONFIG['api_key'])

if connection_success:
    print("✅ Conexión exitosa - Listo para monitorear entrenamiento")
else:
    print("⚠️  Error en conexión - Continuando sin wandb")
    EXPERIMENT_CONFIG['use_wandb'] = False

print("="*70)
"""

# ============================================================
# BLOQUE 3: CREAR MONITOR DE ENTRENAMIENTO
# ============================================================
WANDB_MONITOR_CREATION_BLOCK = """
# ============================================================
# CREAR MONITOR DE ENTRENAMIENTO
# ============================================================

print("="*70)
print("CREANDO MONITOR DE ENTRENAMIENTO")
print("="*70)

# Crear monitor de entrenamiento
monitor = create_training_monitor(
    config=EXPERIMENT_CONFIG,
    experiment_name=EXPERIMENT_CONFIG["experiment_name"],
    use_wandb=EXPERIMENT_CONFIG["use_wandb"],
    wandb_key=WANDB_CONFIG["api_key"],
    tags=WANDB_CONFIG["tags"],
    notes=WANDB_CONFIG["notes"]
)

# Registrar el modelo en wandb
if EXPERIMENT_CONFIG["use_wandb"]:
    monitor.log_model(best_model, input_shape=(1, 65, 41))
    print("✅ Modelo registrado en wandb")

print(f"📊 Monitor configurado:")
print(f"   - Proyecto: {monitor.project_name}")
print(f"   - Experimento: {monitor.experiment_name}")
print(f"   - Wandb habilitado: {monitor.use_wandb}")
print(f"   - Plot cada: {monitor.plot_every} épocas")
print("="*70)
"""

# ============================================================
# BLOQUE 4: FUNCIÓN DE ENTRENAMIENTO CON MONITOREO
# ============================================================
WANDB_TRAINING_FUNCTION_BLOCK = '''
# ============================================================
# FUNCIÓN DE ENTRENAMIENTO CON MONITOREO
# ============================================================

def train_with_wandb_monitoring(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=100):
    """
    Entrenar modelo con monitoreo en tiempo real usando wandb.
    """
    print("="*70)
    print("INICIANDO ENTRENAMIENTO CON MONITOREO WANDB")
    print("="*70)
    
    # Importar funciones de entrenamiento
    from modules.models.cnn2d.training import train_one_epoch, evaluate
    
    model.train()
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Entrenar una época
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluar
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Loggear métricas a wandb y local
        monitor.log(
            epoch=epoch + 1,
            train_loss=train_metrics['loss'],
            train_f1=train_metrics['f1'],
            train_accuracy=train_metrics['accuracy'],
            train_precision=train_metrics['precision'],
            train_recall=train_metrics['recall'],
            val_loss=val_metrics['loss'],
            val_f1=val_metrics['f1'],
            val_accuracy=val_metrics['accuracy'],
            val_precision=val_metrics['precision'],
            val_recall=val_metrics['recall'],
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        # Plotear localmente cada N épocas
        if monitor.should_plot(epoch + 1):
            monitor.plot_local()
        
        # Early stopping basado en val_f1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), optuna_results_dir / 'best_model_wandb.pth')
        else:
            patience_counter += 1
        
        # Aplicar scheduler
        if scheduler:
            scheduler.step()
        
        # Early stopping
        if patience_counter >= FINAL_TRAINING_CONFIG['early_stopping_patience']:
            print(f"\\n⚠️  Early stopping en época {epoch + 1}")
            print(f"    Mejor val_f1: {best_val_f1:.4f}")
            break
        
        # Imprimir progreso
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época {epoch + 1:3d}/{epochs} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Finalizar monitoreo
    monitor.finish()
    monitor.print_summary()
    
    return {
        "model": model,
        "best_val_f1": best_val_f1,
        "final_epoch": epoch + 1
    }

print("✅ Función de entrenamiento con wandb creada")
print("="*70)
'''

# ============================================================
# BLOQUE 5: ENTRENAR CON MONITOREO
# ============================================================
WANDB_TRAINING_EXECUTION_BLOCK = """
# ============================================================
# ENTRENAR MODELO CON MONITOREO WANDB
# ============================================================

print("="*70)
print("ENTRENANDO MODELO CON MONITOREO WANDB")
print("="*70)

# Ejecutar entrenamiento con monitoreo
training_results = train_with_wandb_monitoring(
    model=best_model,
    train_loader=train_loader_final,
    val_loader=val_loader_final,
    optimizer=optimizer_final,
    criterion=criterion_final,
    scheduler=scheduler_final,
    epochs=FINAL_TRAINING_CONFIG['n_epochs']
)

# Extraer resultados
final_model = training_results["model"]
best_val_f1 = training_results["best_val_f1"]
final_epoch = training_results["final_epoch"]

print(f"\\n🎉 Entrenamiento completado:")
print(f"   - Mejor val_f1: {best_val_f1:.4f}")
print(f"   - Épocas entrenadas: {final_epoch}")
print(f"   - Modelo guardado: best_model_wandb.pth")
print("="*70)
"""

# ============================================================
# BLOQUE 6: EVALUACIÓN FINAL CON WANDB
# ============================================================
WANDB_EVALUATION_BLOCK = """
# ============================================================
# EVALUACIÓN FINAL CON WANDB
# ============================================================

print("="*70)
print("EVALUACIÓN FINAL CON WANDB")
print("="*70)

# Evaluar modelo final en test set
from modules.models.cnn2d.training import detailed_evaluation, print_evaluation_report

final_test_metrics = detailed_evaluation(
    model=final_model,
    loader=test_loader_final,
    device=device,
    class_names=["Healthy", "Parkinson"]
)

# Imprimir reporte
print_evaluation_report(final_test_metrics, class_names=["Healthy", "Parkinson"])

# Loggear métricas finales a wandb
if EXPERIMENT_CONFIG["use_wandb"]:
    monitor.log(
        epoch=final_epoch,
        test_accuracy=final_test_metrics["accuracy"],
        test_f1_macro=final_test_metrics["f1_macro"],
        test_precision_macro=final_test_metrics["classification_report"]["macro avg"]["precision"],
        test_recall_macro=final_test_metrics["classification_report"]["macro avg"]["recall"],
        test_f1_weighted=final_test_metrics["classification_report"]["weighted avg"]["f1-score"]
    )
    print("✅ Métricas finales loggeadas a wandb")

# Guardar métricas finales
final_metrics_path = optuna_results_dir / "test_metrics_wandb.json"
final_metrics_to_save = {
    "accuracy": float(final_test_metrics["accuracy"]),
    "f1_macro": float(final_test_metrics["f1_macro"]),
    "precision_macro": float(final_test_metrics["classification_report"]["macro avg"]["precision"]),
    "recall_macro": float(final_test_metrics["classification_report"]["macro avg"]["recall"]),
    "f1_weighted": float(final_test_metrics["classification_report"]["weighted avg"]["f1-score"]),
    "confusion_matrix": final_test_metrics["confusion_matrix"].tolist(),
    "best_hyperparameters": best_params,
    "training_config": FINAL_TRAINING_CONFIG,
    "final_epoch": final_epoch,
    "best_val_f1": best_val_f1,
    "wandb_enabled": EXPERIMENT_CONFIG["use_wandb"]
}

import json
with open(final_metrics_path, "w") as f:
    json.dump(final_metrics_to_save, f, indent=2)

print(f"\\n💾 Métricas finales guardadas en: {final_metrics_path}")
print("="*70)
"""

# ============================================================
# BLOQUE 7: RESUMEN FINAL CON WANDB
# ============================================================
WANDB_SUMMARY_BLOCK = """
# ============================================================
# RESUMEN FINAL CON WANDB
# ============================================================

print("="*70)
print("RESUMEN FINAL CON WANDB")
print("="*70)

print(f"\\n🔍 PROCESO DE OPTIMIZACIÓN:")
print(f"   - Configuraciones evaluadas: {len(results_df)}")
print(f"   - Mejor F1-score en validación: {results_df['f1'].max():.4f}")
print(f"   - F1-score promedio: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")

print(f"\\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
for param, value in best_params.items():
    if param not in ['f1', 'accuracy', 'precision', 'recall', 'val_loss', 'train_loss']:
        print(f"   - {param}: {value}")

print(f"\\n📊 RESULTADOS FINALES EN TEST SET:")
print(f"   - Accuracy:  {final_test_metrics['accuracy']:.4f}")
print(f"   - Precision: {final_test_metrics['classification_report']['macro avg']['precision']:.4f}")
print(f"   - Recall:    {final_test_metrics['classification_report']['macro avg']['recall']:.4f}")
print(f"   - F1-Score:  {final_test_metrics['f1_macro']:.4f}")

if EXPERIMENT_CONFIG["use_wandb"]:
    print(f"\\n📊 VISUALIZACIÓN EN WANDB:")
    print(f"   - Proyecto: {WANDB_CONFIG['project_name']}")
    print(f"   - Experimento: {EXPERIMENT_CONFIG['experiment_name']}")
    print(f"   - URL: https://wandb.ai/{WANDB_CONFIG['project_name']}")
    print(f"   - Métricas en tiempo real disponibles")

print(f"\\n💾 ARCHIVOS GUARDADOS:")
print(f"   - best_model_wandb.pth           # Modelo final optimizado")
print(f"   - test_metrics_wandb.json        # Métricas en test set")
print(f"   - training_progress_optuna.png   # Gráfica de entrenamiento local")
print(f"   - confusion_matrix_optuna.png    # Matriz de confusión")

print("="*70)
print("ENTRENAMIENTO CON WANDB COMPLETADO EXITOSAMENTE")
print("="*70)
"""


# ============================================================
# FUNCIÓN PARA MOSTRAR TODOS LOS BLOQUES
# ============================================================
def show_all_blocks():
    """Mostrar todos los bloques de código para el notebook."""
    blocks = [
        ("BLOQUE 1: Configuración de Weights & Biases", WANDB_CONFIGURATION_BLOCK),
        ("BLOQUE 2: Importar módulos de monitoreo", WANDB_IMPORTS_BLOCK),
        ("BLOQUE 3: Crear monitor de entrenamiento", WANDB_MONITOR_CREATION_BLOCK),
        (
            "BLOQUE 4: Función de entrenamiento con monitoreo",
            WANDB_TRAINING_FUNCTION_BLOCK,
        ),
        ("BLOQUE 5: Entrenar con monitoreo", WANDB_TRAINING_EXECUTION_BLOCK),
        ("BLOQUE 6: Evaluación final con wandb", WANDB_EVALUATION_BLOCK),
        ("BLOQUE 7: Resumen final con wandb", WANDB_SUMMARY_BLOCK),
    ]

    for title, block in blocks:
        print(f"\\n{'=' * 70}")
        print(f"{title}")
        print(f"{'=' * 70}")
        print(block.strip())
        print(f"\\n{'=' * 70}")


if __name__ == "__main__":
    show_all_blocks()
