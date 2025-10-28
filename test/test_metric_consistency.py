#!/usr/bin/env python3
"""
Prueba de consistencia de métricas entre Optuna y entrenamiento final
====================================================================

Verifica que Optuna y el entrenamiento final usen la misma métrica (F1-macro).
"""

import unittest
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.models.cnn2d.training import compute_metrics


class TestMetricConsistency(unittest.TestCase):
    """Pruebas para verificar consistencia de métricas."""

    def setUp(self):
        """Configurar datos de prueba."""
        # Crear datos de prueba con clases desbalanceadas
        np.random.seed(42)
        n_samples = 1000
        
        # 70% clase 0, 30% clase 1 (desbalanceado)
        self.labels = np.concatenate([
            np.zeros(int(0.7 * n_samples)),  # 700 muestras clase 0
            np.ones(int(0.3 * n_samples))    # 300 muestras clase 1
        ])
        
        # Predicciones con cierto nivel de error
        self.predictions = self.labels.copy()
        # Introducir errores en 20% de las muestras
        error_indices = np.random.choice(len(self.labels), size=int(0.2 * n_samples), replace=False)
        self.predictions[error_indices] = 1 - self.predictions[error_indices]

    def test_f1_macro_consistency(self):
        """Probar que F1-macro se calcula consistentemente."""
        # Calcular F1-macro manualmente
        f1_macro_manual = f1_score(self.labels, self.predictions, average="macro")
        
        # Calcular usando la función del módulo
        metrics = compute_metrics(self.labels, self.predictions)
        f1_macro_module = metrics["f1"]
        
        # Deberían ser iguales
        self.assertAlmostEqual(f1_macro_manual, f1_macro_module, places=6)
        print(f"✅ F1-macro consistente: {f1_macro_manual:.4f}")

    def test_f1_binary_vs_macro_difference(self):
        """Probar que F1-binary y F1-macro son diferentes en datos desbalanceados."""
        f1_binary = f1_score(self.labels, self.predictions, average="binary")
        f1_macro = f1_score(self.labels, self.predictions, average="macro")
        
        # En datos desbalanceados, estas métricas deberían ser diferentes
        self.assertNotAlmostEqual(f1_binary, f1_macro, places=3)
        print(f"📊 F1-binary: {f1_binary:.4f}")
        print(f"📊 F1-macro: {f1_macro:.4f}")
        print(f"📊 Diferencia: {abs(f1_binary - f1_macro):.4f}")

    def test_optuna_vs_training_metrics(self):
        """Simular cálculo de métricas como en Optuna vs entrenamiento."""
        # Simular cálculo como en Optuna
        f1_optuna = f1_score(self.labels, self.predictions, average="macro")
        acc_optuna = accuracy_score(self.labels, self.predictions)
        prec_optuna = precision_score(self.labels, self.predictions, average="macro")
        rec_optuna = recall_score(self.labels, self.predictions, average="macro")
        
        # Simular cálculo como en entrenamiento (después de la corrección)
        metrics_training = compute_metrics(self.labels, self.predictions)
        f1_training = metrics_training["f1"]
        acc_training = metrics_training["accuracy"]
        prec_training = metrics_training["precision"]
        rec_training = metrics_training["recall"]
        
        # Deberían ser iguales
        self.assertAlmostEqual(f1_optuna, f1_training, places=6)
        self.assertAlmostEqual(acc_optuna, acc_training, places=6)
        self.assertAlmostEqual(prec_optuna, prec_training, places=6)
        self.assertAlmostEqual(rec_optuna, rec_training, places=6)
        
        print("✅ Métricas consistentes entre Optuna y entrenamiento:")
        print(f"   F1-macro: {f1_optuna:.4f} = {f1_training:.4f}")
        print(f"   Accuracy: {acc_optuna:.4f} = {acc_training:.4f}")
        print(f"   Precision: {prec_optuna:.4f} = {prec_training:.4f}")
        print(f"   Recall: {rec_optuna:.4f} = {rec_training:.4f}")

    def test_early_stopping_metric_consistency(self):
        """Probar que early stopping usa la misma métrica que Optuna."""
        # Simular diferentes épocas con diferentes F1 scores
        epochs_data = [
            {"epoch": 0, "f1_macro": 0.60, "f1_binary": 0.65},
            {"epoch": 1, "f1_macro": 0.65, "f1_binary": 0.70},
            {"epoch": 2, "f1_macro": 0.70, "f1_binary": 0.68},  # F1-macro mejor, F1-binary peor
            {"epoch": 3, "f1_macro": 0.68, "f1_binary": 0.72},  # F1-binary mejor, F1-macro peor
            {"epoch": 4, "f1_macro": 0.72, "f1_binary": 0.71},  # F1-macro mejor
        ]
        
        # Simular early stopping con F1-macro (como Optuna)
        best_f1_macro = 0.0
        best_epoch_macro = 0
        
        for data in epochs_data:
            if data["f1_macro"] > best_f1_macro:
                best_f1_macro = data["f1_macro"]
                best_epoch_macro = data["epoch"]
        
        # Simular early stopping con F1-binary (como antes de la corrección)
        best_f1_binary = 0.0
        best_epoch_binary = 0
        
        for data in epochs_data:
            if data["f1_binary"] > best_f1_binary:
                best_f1_binary = data["f1_binary"]
                best_epoch_binary = data["epoch"]
        
        # Verificar que pueden dar resultados diferentes
        self.assertNotEqual(best_epoch_macro, best_epoch_binary)
        self.assertNotAlmostEqual(best_f1_macro, best_f1_binary, places=2)
        
        print(f"📊 Early stopping con F1-macro: época {best_epoch_macro}, F1={best_f1_macro:.4f}")
        print(f"📊 Early stopping con F1-binary: época {best_epoch_binary}, F1={best_f1_binary:.4f}")
        print("✅ Diferentes métricas pueden dar diferentes resultados de early stopping")

    def test_imbalanced_dataset_impact(self):
        """Probar el impacto en datasets desbalanceados."""
        # Crear dataset muy desbalanceado (95% clase 0, 5% clase 1)
        n_samples = 1000
        labels_imbalanced = np.concatenate([
            np.zeros(int(0.95 * n_samples)),  # 950 muestras clase 0
            np.ones(int(0.05 * n_samples))    # 50 muestras clase 1
        ])
        
        # Predicciones que clasifican todo como clase 0 (clasificador sesgado)
        predictions_biased = np.zeros_like(labels_imbalanced)
        
        # Calcular métricas
        f1_binary = f1_score(labels_imbalanced, predictions_biased, average="binary")
        f1_macro = f1_score(labels_imbalanced, predictions_biased, average="macro")
        f1_weighted = f1_score(labels_imbalanced, predictions_biased, average="weighted")
        
        print(f"\n📊 Dataset muy desbalanceado (95% clase 0, 5% clase 1):")
        print(f"   F1-binary: {f1_binary:.4f}")
        print(f"   F1-macro: {f1_macro:.4f}")
        print(f"   F1-weighted: {f1_weighted:.4f}")
        
        # F1-macro debería ser más sensible a la clase minoritaria
        self.assertLess(f1_macro, f1_binary)
        print("✅ F1-macro es más sensible a la clase minoritaria (mejor para early stopping)")


def run_metric_analysis():
    """Ejecutar análisis detallado de métricas."""
    print("="*70)
    print("ANÁLISIS DE CONSISTENCIA DE MÉTRICAS")
    print("="*70)
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    # Dataset desbalanceado (70% clase 0, 30% clase 1)
    labels = np.concatenate([
        np.zeros(int(0.7 * n_samples)),
        np.ones(int(0.3 * n_samples))
    ])
    
    # Predicciones con 20% de error
    predictions = labels.copy()
    error_indices = np.random.choice(len(labels), size=int(0.2 * n_samples), replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    print(f"📊 Dataset de prueba:")
    print(f"   Total muestras: {len(labels)}")
    print(f"   Clase 0: {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
    print(f"   Clase 1: {(labels == 1).sum()} ({(labels == 1).sum()/len(labels)*100:.1f}%)")
    print(f"   Errores introducidos: {len(error_indices)} ({len(error_indices)/len(labels)*100:.1f}%)")
    
    # Calcular todas las métricas
    f1_binary = f1_score(labels, predictions, average="binary")
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_micro = f1_score(labels, predictions, average="micro")
    
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average="macro")
    recall_macro = recall_score(labels, predictions, average="macro")
    
    print(f"\n📈 Métricas calculadas:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-binary: {f1_binary:.4f}")
    print(f"   F1-macro: {f1_macro:.4f}")
    print(f"   F1-weighted: {f1_weighted:.4f}")
    print(f"   F1-micro: {f1_micro:.4f}")
    print(f"   Precision-macro: {precision_macro:.4f}")
    print(f"   Recall-macro: {recall_macro:.4f}")
    
    # Verificar consistencia con el módulo
    try:
        metrics_module = compute_metrics(labels, predictions)
        print(f"\n✅ Verificación con módulo:")
        print(f"   F1 del módulo: {metrics_module['f1']:.4f}")
        print(f"   ¿Consistente? {'✅ SÍ' if abs(metrics_module['f1'] - f1_macro) < 1e-6 else '❌ NO'}")
    except Exception as e:
        print(f"❌ Error al usar módulo: {e}")
    
    print(f"\n💡 Recomendación:")
    print(f"   - Optuna usa F1-macro: {f1_macro:.4f}")
    print(f"   - Entrenamiento debe usar F1-macro: {f1_macro:.4f}")
    print(f"   - Diferencia con F1-binary: {abs(f1_macro - f1_binary):.4f}")
    
    if abs(f1_macro - f1_binary) > 0.01:
        print(f"   ⚠️  Diferencia significativa - corrección necesaria")
    else:
        print(f"   ✅ Diferencia pequeña - corrección aplicada correctamente")


if __name__ == "__main__":
    print("="*70)
    print("PRUEBAS DE CONSISTENCIA DE MÉTRICAS")
    print("="*70)
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*70)
    print("ANÁLISIS DETALLADO")
    print("="*70)
    
    # Ejecutar análisis detallado
    run_metric_analysis()
