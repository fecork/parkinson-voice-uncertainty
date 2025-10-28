#!/usr/bin/env python3
"""
Prueba de conexión con Weights & Biases
======================================

Verifica que la integración con wandb funcione correctamente.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.training_monitor import TrainingMonitor
from modules.core.experiment_config import WANDB_CONFIG


class TestWandBConnection(unittest.TestCase):
    """Pruebas para la conexión con Weights & Biases."""

    def test_wandb_import(self):
        """Probar que wandb se puede importar."""
        try:
            import wandb
            print(f"✅ wandb importado correctamente: {wandb.__version__}")
            self.assertTrue(True)
        except ImportError as e:
            print(f"❌ Error importando wandb: {e}")
            self.fail("wandb no está disponible")

    def test_wandb_config(self):
        """Probar configuración de wandb."""
        self.assertIn("project_name", WANDB_CONFIG)
        self.assertIn("enabled", WANDB_CONFIG)
        self.assertIn("api_key", WANDB_CONFIG)
        
        self.assertEqual(WANDB_CONFIG["project_name"], "parkinson-voice-uncertainty")
        self.assertTrue(WANDB_CONFIG["enabled"])
        self.assertIsNotNone(WANDB_CONFIG["api_key"])

    @patch('modules.core.training_monitor.wandb')
    def test_wandb_login_success(self, mock_wandb):
        """Probar login exitoso a wandb."""
        # Configurar mock para login exitoso
        mock_wandb.login.return_value = True
        
        # Crear monitor con wandb
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            use_wandb=True,
            wandb_key="test_key"
        )
        
        # Verificar que se llamó login
        mock_wandb.login.assert_called_once_with(key="test_key")
        self.assertTrue(monitor.use_wandb)

    @patch('modules.core.training_monitor.wandb')
    def test_wandb_login_failure(self, mock_wandb):
        """Probar fallo en login a wandb."""
        # Configurar mock para fallo en login
        mock_wandb.login.side_effect = Exception("Authentication failed")
        
        # Crear monitor con wandb (debería deshabilitarse automáticamente)
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            use_wandb=True,
            wandb_key="invalid_key"
        )
        
        # Verificar que se deshabilitó wandb
        self.assertFalse(monitor.use_wandb)

    @patch('modules.core.training_monitor.wandb')
    def test_wandb_init_success(self, mock_wandb):
        """Probar inicialización exitosa de experimento."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()
        
        config = {"learning_rate": 0.001, "batch_size": 32}
        
        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            config=config,
            use_wandb=True,
            wandb_key="test_key"
        )
        
        # Verificar que se llamó init
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            name="test_experiment",
            config=config,
            reinit=True
        )

    @patch('modules.core.training_monitor.wandb')
    def test_wandb_logging(self, mock_wandb):
        """Probar logging de métricas a wandb."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.log = MagicMock()
        
        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            use_wandb=True,
            wandb_key="test_key"
        )
        
        # Loggear métricas
        monitor.log(epoch=1, train_loss=0.5, val_f1=0.8)
        
        # Verificar que se llamó log
        mock_wandb.log.assert_called_once_with({
            "epoch": 1,
            "train_loss": 0.5,
            "val_f1": 0.8
        })

    @patch('modules.core.training_monitor.wandb')
    def test_wandb_finish(self, mock_wandb):
        """Probar finalización de experimento."""
        # Configurar mocks
        mock_wandb.login.return_value = True
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.finish = MagicMock()
        
        # Crear monitor
        monitor = TrainingMonitor(
            project_name="test_project",
            experiment_name="test_experiment",
            use_wandb=True,
            wandb_key="test_key"
        )
        
        # Finalizar
        monitor.finish()
        
        # Verificar que se llamó finish
        mock_wandb.finish.assert_called_once()

    def test_wandb_error_handling(self):
        """Probar manejo de errores en wandb."""
        # Probar con wandb que falla en login
        with patch('modules.core.training_monitor.wandb') as mock_wandb:
            mock_wandb.login.side_effect = Exception("Network error")
            
            # No debería lanzar excepción
            monitor = TrainingMonitor(use_wandb=True)
            self.assertFalse(monitor.use_wandb)

        # Probar con wandb que falla en init
        with patch('modules.core.training_monitor.wandb') as mock_wandb:
            mock_wandb.login.return_value = True
            mock_wandb.init.side_effect = Exception("Project not found")
            
            # No debería lanzar excepción
            monitor = TrainingMonitor(use_wandb=True)
            self.assertFalse(monitor.use_wandb)

    def test_wandb_logging_error_handling(self):
        """Probar manejo de errores en logging."""
        with patch('modules.core.training_monitor.wandb') as mock_wandb:
            mock_wandb.login.return_value = True
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.log.side_effect = Exception("Logging failed")
            
            # Crear monitor
            monitor = TrainingMonitor(use_wandb=True)
            
            # Logging debería fallar silenciosamente
            with patch('builtins.print') as mock_print:
                monitor.log(epoch=1, test_metric=0.5)
                mock_print.assert_called()


def test_real_wandb_connection():
    """Probar conexión real con wandb (opcional)."""
    print("🔗 Probando conexión real con Weights & Biases...")
    
    try:
        import wandb
        
        # Probar login con la API key del proyecto
        try:
            wandb.login(key=WANDB_CONFIG["api_key"])
            print("✅ Login exitoso a Weights & Biases")
        except Exception as e:
            print(f"❌ Error en login: {e}")
            return False
        
        # Probar inicialización de experimento
        try:
            run = wandb.init(
                project=WANDB_CONFIG["project_name"],
                name="test_connection",
                config={"test": True},
                reinit=True
            )
            print("✅ Experimento inicializado correctamente")
            
            # Probar logging
            wandb.log({"test_metric": 0.5, "epoch": 1})
            print("✅ Logging de métricas exitoso")
            
            # Finalizar
            wandb.finish()
            print("✅ Experimento finalizado correctamente")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en experimento: {e}")
            return False
            
    except ImportError:
        print("❌ wandb no está instalado")
        return False


def run_visualization_demo():
    """Ejecutar demo de visualización."""
    print("\n" + "="*70)
    print("DEMO DE VISUALIZACIÓN CON WANDB")
    print("="*70)
    
    try:
        from modules.core.training_monitor import TrainingMonitor
        
        # Crear monitor con wandb
        monitor = TrainingMonitor(
            project_name="parkinson-voice-uncertainty",
            experiment_name="visualization_demo",
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 5,
                "optimizer": "SGD"
            },
            use_wandb=True,
            wandb_key=WANDB_CONFIG["api_key"],
            plot_every=1
        )
        
        print("📊 Simulando entrenamiento con monitoreo...")
        
        # Simular entrenamiento
        for epoch in range(5):
            import numpy as np
            
            # Métricas simuladas
            train_loss = 1.0 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.01)
            val_loss = 0.8 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.01)
            train_f1 = min(0.95, 0.5 + epoch * 0.1 + np.random.normal(0, 0.02))
            val_f1 = min(0.9, 0.4 + epoch * 0.12 + np.random.normal(0, 0.02))
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
            
            print(f"   Época {epoch}: Loss={train_loss:.3f}, F1={val_f1:.3f}")
            
            # Mostrar gráfico local
            if monitor.should_plot(epoch):
                try:
                    monitor.plot_local()
                except Exception as e:
                    print(f"   ⚠️  Error en visualización local: {e}")
        
        # Resumen final
        print("\n📋 Resumen del entrenamiento:")
        monitor.print_summary()
        
        # Finalizar
        monitor.finish()
        
        print(f"\n🔗 Ver resultados en: https://wandb.ai/{WANDB_CONFIG['project_name']}")
        print("✅ Demo completado exitosamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("PRUEBAS DE CONEXIÓN CON WEIGHTS & BIASES")
    print("="*70)
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*70)
    print("PRUEBA DE CONEXIÓN REAL (OPCIONAL)")
    print("="*70)
    
    # Preguntar si ejecutar prueba real
    try:
        response = input("¿Ejecutar prueba real con wandb? (y/N): ").strip().lower()
        if response in ['y', 'yes', 'sí', 'si']:
            test_real_wandb_connection()
    except KeyboardInterrupt:
        print("\n⏹️  Prueba cancelada por el usuario")
    
    print("\n" + "="*70)
    print("DEMO DE VISUALIZACIÓN")
    print("="*70)
    
    # Ejecutar demo
    run_visualization_demo()
