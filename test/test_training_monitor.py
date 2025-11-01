#!/usr/bin/env python3
"""
Pruebas unitarias para el TrainingMonitor
========================================

Verifica que el sistema de monitoreo y visualización funcione correctamente.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.training_monitor import TrainingMonitor, create_training_monitor


class TestTrainingMonitor(unittest.TestCase):
    """Pruebas para la clase TrainingMonitor."""

    def setUp(self):
        """Configurar para cada prueba."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 10,
            "optimizer": "SGD"
        }

    def tearDown(self):
        """Limpiar después de cada prueba."""
        # Limpiar archivos temporales
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('modules.core.training_monitor.wandb')
    def test_training_monitor_initialization_with_wandb(self, mock_wandb):
        """Probar inicialización con Weights & Biases."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()

        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            config=self.config,
            use_wandb=True,
            wandb_key="test_key"
        )

        # Verificaciones
        self.assertEqual(monitor.project_name, "test_project")
        self.assertEqual(monitor.experiment_name, "test_experiment")
        self.assertEqual(monitor.config, self.config)
        self.assertTrue(monitor.use_wandb)
        
        # Verificar que wandb se configuró
        mock_wandb.login.assert_called_once_with(key="test_key")
        mock_wandb.init.assert_called_once()

    def test_training_monitor_initialization_without_wandb(self):
        """Probar inicialización sin Weights & Biases."""
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            config=self.config,
            use_wandb=False
        )

        # Verificaciones
        self.assertEqual(monitor.project_name, "test_project")
        self.assertEqual(monitor.experiment_name, "test_experiment")
        self.assertEqual(monitor.config, self.config)
        self.assertFalse(monitor.use_wandb)

    @patch('modules.core.training_monitor.wandb')
    def test_log_metrics(self, mock_wandb):
        """Probar logging de métricas."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()

        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            config=self.config,
            use_wandb=True
        )

        # Loggear métricas
        monitor.log(epoch=1, train_loss=0.5, val_loss=0.4, train_f1=0.8, val_f1=0.75)

        # Verificaciones
        self.assertEqual(len(monitor.metrics['epoch']), 1)
        self.assertEqual(monitor.metrics['epoch'][0], 1)
        self.assertEqual(monitor.metrics['train_loss'][0], 0.5)
        self.assertEqual(monitor.metrics['val_loss'][0], 0.4)
        self.assertEqual(monitor.metrics['train_f1'][0], 0.8)
        self.assertEqual(monitor.metrics['val_f1'][0], 0.75)

        # Verificar que wandb.log se llamó
        mock_wandb.log.assert_called_once_with({
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "train_f1": 0.8,
            "val_f1": 0.75
        })

    def test_log_metrics_without_wandb(self):
        """Probar logging de métricas sin wandb."""
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            config=self.config,
            use_wandb=False
        )

        # Loggear métricas
        monitor.log(epoch=1, train_loss=0.5, val_loss=0.4, train_f1=0.8, val_f1=0.75)

        # Verificaciones
        self.assertEqual(len(monitor.metrics['epoch']), 1)
        self.assertEqual(monitor.metrics['epoch'][0], 1)
        self.assertEqual(monitor.metrics['train_loss'][0], 0.5)
        self.assertEqual(monitor.metrics['val_loss'][0], 0.4)
        self.assertEqual(monitor.metrics['train_f1'][0], 0.8)
        self.assertEqual(monitor.metrics['val_f1'][0], 0.75)

    def test_should_plot(self):
        """Probar función should_plot."""
        monitor = TrainingMonitor(plot_every=5)

        # Casos de prueba
        self.assertTrue(monitor.should_plot(0))    # Época 0 siempre se plotea
        self.assertTrue(monitor.should_plot(5))    # Múltiplo de 5
        self.assertTrue(monitor.should_plot(10))   # Múltiplo de 5
        self.assertFalse(monitor.should_plot(1))   # No múltiplo de 5
        self.assertFalse(monitor.should_plot(3))   # No múltiplo de 5
        self.assertFalse(monitor.should_plot(7))   # No múltiplo de 5

    def test_get_best_metrics(self):
        """Probar obtención de mejores métricas."""
        monitor = TrainingMonitor()

        # Simular datos de entrenamiento
        for epoch in range(10):
            monitor.log(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.05,  # Decreciente
                val_loss=0.8 - epoch * 0.03,    # Decreciente
                train_f1=0.5 + epoch * 0.03,    # Creciente
                val_f1=0.4 + epoch * 0.04       # Creciente
            )

        # Obtener mejores métricas
        best = monitor.get_best_metrics()

        # Verificaciones
        self.assertIn('best_train_loss', best)
        self.assertIn('best_val_loss', best)
        self.assertIn('best_train_f1', best)
        self.assertIn('best_val_f1', best)

        # Los mejores losses deberían ser los últimos (más bajos)
        self.assertEqual(best['best_train_loss'], 0.55)  # 1.0 - 9 * 0.05
        self.assertEqual(best['best_val_loss'], 0.53)    # 0.8 - 9 * 0.03

        # Los mejores F1 deberían ser los últimos (más altos)
        self.assertEqual(best['best_train_f1'], 0.77)    # 0.5 + 9 * 0.03
        self.assertEqual(best['best_val_f1'], 0.76)      # 0.4 + 9 * 0.04

    @patch('modules.core.training_monitor.plt.show')
    @patch('modules.core.training_monitor.clear_output')
    def test_plot_local(self, mock_clear_output, mock_show):
        """Probar visualización local."""
        monitor = TrainingMonitor()

        # Simular datos
        for epoch in range(5):
            monitor.log(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.1,
                val_loss=0.8 - epoch * 0.08,
                train_f1=0.5 + epoch * 0.1,
                val_f1=0.4 + epoch * 0.12
            )

        # Probar plot local
        monitor.plot_local()

        # Verificaciones
        mock_clear_output.assert_called_once()
        mock_show.assert_called_once()

    @patch('modules.core.training_monitor.plt.savefig')
    @patch('modules.core.training_monitor.plt.show')
    @patch('modules.core.training_monitor.clear_output')
    def test_plot_local_with_save(self, mock_clear_output, mock_show, mock_savefig):
        """Probar visualización local con guardado."""
        monitor = TrainingMonitor()

        # Simular datos
        for epoch in range(3):
            monitor.log(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.2,
                val_loss=0.8 - epoch * 0.15
            )

        # Probar plot local con guardado
        save_path = Path(self.temp_dir) / "test_plot.png"
        monitor.plot_local(save_path=save_path)

        # Verificaciones
        mock_clear_output.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path, dpi=150, bbox_inches='tight')

    @patch('modules.core.training_monitor.wandb')
    def test_finish_with_wandb(self, mock_wandb):
        """Probar finalización con wandb."""
        monitor = TrainingMonitor(use_wandb=True)
        monitor.finish()

        # Verificar que wandb.finish se llamó
        mock_wandb.finish.assert_called_once()

    def test_finish_without_wandb(self):
        """Probar finalización sin wandb."""
        monitor = TrainingMonitor(use_wandb=False)
        
        # No debería lanzar excepción
        monitor.finish()

    def test_create_training_monitor(self):
        """Probar función de creación de monitor."""
        monitor = create_training_monitor(
            config=self.config,
            experiment_name="test_experiment",
            use_wandb=False
        )

        # Verificaciones
        self.assertIsInstance(monitor, TrainingMonitor)
        self.assertEqual(monitor.project_name, "parkinson-voice-uncertainty")
        self.assertEqual(monitor.experiment_name, "test_experiment")
        self.assertEqual(monitor.config, self.config)
        self.assertFalse(monitor.use_wandb)


class TestTrainingMonitorIntegration(unittest.TestCase):
    """Pruebas de integración para el TrainingMonitor."""

    def setUp(self):
        """Configurar para cada prueba."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpiar después de cada prueba."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('modules.core.training_monitor.wandb')
    def test_full_training_simulation(self, mock_wandb):
        """Simular un entrenamiento completo."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()

        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="full_training_test",
            config={"learning_rate": 0.001, "batch_size": 32},
            use_wandb=True,
            plot_every=3
        )

        # Simular entrenamiento de 10 épocas
        for epoch in range(10):
            # Simular métricas realistas
            train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.01)
            val_loss = 0.8 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.01)
            train_f1 = min(0.95, 0.5 + epoch * 0.05 + np.random.normal(0, 0.02))
            val_f1 = min(0.9, 0.4 + epoch * 0.06 + np.random.normal(0, 0.02))
            lr = 0.001 * (0.9 ** epoch)

            # Loggear métricas
            monitor.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_f1=train_f1,
                val_f1=val_f1,
                learning_rate=lr
            )

            # Verificar que se plotea en las épocas correctas
            if monitor.should_plot(epoch):
                with patch('modules.core.training_monitor.plt.show') as mock_show:
                    with patch('modules.core.training_monitor.clear_output') as mock_clear:
                        monitor.plot_local()
                        mock_show.assert_called_once()
                        mock_clear.assert_called_once()

        # Verificar que se loggearon todas las épocas
        self.assertEqual(len(monitor.metrics['epoch']), 10)
        self.assertEqual(mock_wandb.log.call_count, 10)

        # Verificar mejores métricas
        best = monitor.get_best_metrics()
        self.assertIn('best_train_f1', best)
        self.assertIn('best_val_f1', best)
        self.assertIn('best_train_loss', best)
        self.assertIn('best_val_loss', best)

        # Finalizar
        monitor.finish()
        mock_wandb.finish.assert_called_once()

    def test_error_handling(self):
        """Probar manejo de errores."""
        # Probar con wandb que falla
        with patch('modules.core.training_monitor.wandb') as mock_wandb:
            mock_wandb.login.side_effect = Exception("Connection error")
            
            # No debería lanzar excepción
            monitor = TrainingMonitor(use_wandb=True)
            self.assertFalse(monitor.use_wandb)  # Debería deshabilitarse

        # Probar logging sin wandb
        monitor = TrainingMonitor(use_wandb=False)
        monitor.log(epoch=1, test_metric=0.5)  # No debería lanzar excepción

    def test_empty_metrics_plot(self):
        """Probar plot con métricas vacías."""
        monitor = TrainingMonitor()
        
        with patch('modules.core.training_monitor.print') as mock_print:
            monitor.plot_local()
            mock_print.assert_called_with("No hay métricas para mostrar")


def run_visualization_test():
    """Función para probar visualización manualmente."""
    print("🧪 Probando visualización del TrainingMonitor...")
    
    # Crear monitor sin wandb para prueba rápida
    monitor = TrainingMonitor(
        project_name="test_project",
        experiment_name="visualization_test",
        config={"learning_rate": 0.001, "batch_size": 32},
        use_wandb=False,
        plot_every=2
    )
    
    # Simular datos de entrenamiento
    print("📊 Simulando datos de entrenamiento...")
    for epoch in range(10):
        train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.01)
        val_loss = 0.8 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.01)
        train_f1 = min(0.95, 0.5 + epoch * 0.05 + np.random.normal(0, 0.02))
        val_f1 = min(0.9, 0.4 + epoch * 0.06 + np.random.normal(0, 0.02))
        lr = 0.001 * (0.9 ** epoch)
        
        monitor.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_f1=train_f1,
            val_f1=val_f1,
            learning_rate=lr
        )
        
        print(f"   Época {epoch}: Loss={train_loss:.3f}, F1={val_f1:.3f}")
    
    # Mostrar gráficos
    print("📈 Generando visualizaciones...")
    try:
        monitor.plot_local()
        print("✅ Visualización generada correctamente")
    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return False
    
    # Mostrar resumen
    print("📋 Resumen de métricas:")
    monitor.print_summary()
    
    print("✅ Prueba de visualización completada")
    return True


if __name__ == "__main__":
    print("="*70)
    print("PRUEBAS UNITARIAS DEL TRAINING MONITOR")
    print("="*70)
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*70)
    print("PRUEBA MANUAL DE VISUALIZACIÓN")
    print("="*70)
    
    # Ejecutar prueba manual de visualización
    run_visualization_test()
