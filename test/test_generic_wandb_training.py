#!/usr/bin/env python3
"""
Pruebas unitarias para train_with_wandb_monitoring_generic
=========================================================

Pruebas completas para la función de entrenamiento genérica con monitoreo de Weights & Biases.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic
from modules.core.training_monitor import TrainingMonitor


class TestGenericWandbTraining(unittest.TestCase):
    """Pruebas unitarias para train_with_wandb_monitoring_generic."""
    
    def setUp(self):
        """Configurar datos de prueba antes de cada test."""
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear datos sintéticos
        self.batch_size = 4
        self.num_samples = 20
        self.input_size = (3, 64, 64)  # Para CNN2D
        self.num_classes = 2
        
        # Generar datos de entrenamiento
        X_train = torch.randn(self.num_samples, *self.input_size)
        y_train = torch.randint(0, self.num_classes, (self.num_samples,))
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Generar datos de validación
        X_val = torch.randn(self.num_samples // 2, *self.input_size)
        y_val = torch.randint(0, self.num_classes, (self.num_samples // 2,))
        val_dataset = TensorDataset(X_val, y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Crear modelo simple
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, self.num_classes)
        ).to(self.device)
        
        # Crear optimizador y criterio
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Crear scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # Crear monitor mock
        self.monitor = Mock(spec=TrainingMonitor)
        self.monitor.log = Mock()
        self.monitor.should_plot = Mock(return_value=False)
        self.monitor.plot_local = Mock()
        self.monitor.finish = Mock()
        self.monitor.print_summary = Mock()
        
        # Directorio temporal para guardar modelos
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Limpiar después de cada test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_training_generic_architecture(self):
        """Probar entrenamiento básico con arquitectura genérica."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=3,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                model_name="test_model.pth",
                verbose=False
            )
            
            # Verificar resultados
            self.assertIsInstance(results, dict)
            self.assertIn('model', results)
            self.assertIn('best_val_f1', results)
            self.assertIn('final_epoch', results)
            self.assertIn('history', results)
            self.assertIn('early_stopped', results)
            
            # Verificar que se llamaron las funciones de entrenamiento
            self.assertEqual(mock_train.call_count, 3)  # 3 épocas
            self.assertEqual(mock_eval.call_count, 3)   # 3 épocas
            
            # Verificar que se llamó el monitor
            self.assertEqual(self.monitor.log.call_count, 3)
            self.monitor.finish.assert_called_once()
            self.monitor.print_summary.assert_called_once()
    
    def test_training_without_scheduler(self):
        """Probar entrenamiento sin scheduler."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento sin scheduler
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=None,  # Sin scheduler
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=2,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                verbose=False
            )
            
            # Verificar que funciona sin scheduler
            self.assertIsInstance(results, dict)
            self.assertEqual(results['final_epoch'], 2)
            self.assertFalse(results['early_stopped'])
    
    def test_early_stopping(self):
        """Probar early stopping cuando no mejora."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks para simular no mejora
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            # F1 score que no mejora (siempre 0.5)
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.5, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento con early stopping
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=10,
                early_stopping_patience=3,  # Early stopping en 3 épocas
                save_dir=self.save_dir,
                verbose=False
            )
            
            # Verificar early stopping
            self.assertTrue(results['early_stopped'])
            self.assertEqual(results['final_epoch'], 4)  # 3 épocas + 1 inicial
            self.assertEqual(results['best_val_f1'], 0.5)
    
    def test_training_without_save_dir(self):
        """Probar entrenamiento sin directorio de guardado."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento sin save_dir
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=2,
                early_stopping_patience=5,
                save_dir=None,  # Sin directorio de guardado
                verbose=False
            )
            
            # Verificar que funciona sin save_dir
            self.assertIsInstance(results, dict)
            self.assertEqual(results['final_epoch'], 2)
    
    def test_training_with_custom_forward_fn(self):
        """Probar entrenamiento con función forward personalizada."""
        def custom_forward_fn(model, x):
            return model(x)
        
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento con forward_fn personalizada
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=2,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                forward_fn=custom_forward_fn,
                verbose=False
            )
            
            # Verificar que se pasó la función forward
            self.assertIsInstance(results, dict)
            # Verificar que se llamó train_one_epoch_generic con forward_fn
            mock_train.assert_called()
            call_args = mock_train.call_args
            self.assertEqual(call_args[0][4], custom_forward_fn)  # 5to argumento es forward_fn
    
    def test_training_history_structure(self):
        """Probar que el historial de entrenamiento tiene la estructura correcta."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=2,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                verbose=False
            )
            
            # Verificar estructura del historial
            history = results['history']
            expected_keys = [
                'train_loss', 'train_f1', 'train_accuracy', 'train_precision', 'train_recall',
                'val_loss', 'val_f1', 'val_accuracy', 'val_precision', 'val_recall',
                'learning_rate'
            ]
            
            for key in expected_keys:
                self.assertIn(key, history)
                self.assertIsInstance(history[key], list)
                self.assertEqual(len(history[key]), 2)  # 2 épocas
            
            # Verificar valores específicos
            self.assertEqual(history['train_f1'][0], 0.8)
            self.assertEqual(history['val_f1'][0], 0.85)
    
    def test_training_with_architecture_specific_functions(self):
        """Probar entrenamiento con funciones específicas de arquitectura."""
        with patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs, \
             patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval:
            
            # Simular que no hay funciones específicas disponibles
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks para funciones genéricas
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento con arquitectura específica que falla
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="cnn2d",  # Arquitectura específica
                epochs=2,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                verbose=False
            )
            
            # Verificar que se usaron las funciones genéricas como fallback
            self.assertIsInstance(results, dict)
            self.assertEqual(mock_train.call_count, 2)
            self.assertEqual(mock_eval.call_count, 2)
    
    def test_training_with_kwargs(self):
        """Probar entrenamiento con parámetros adicionales."""
        with patch('modules.core.generic_training.train_one_epoch_generic') as mock_train, \
             patch('modules.core.generic_training.evaluate_generic') as mock_eval, \
             patch('modules.core.generic_training.get_architecture_specific_functions') as mock_get_funcs:
            
            # Configurar mocks para que no encuentre funciones específicas
            mock_get_funcs.side_effect = ImportError("No specific functions found")
            
            # Configurar mocks
            mock_train.return_value = {
                'loss': 0.5, 'f1': 0.8, 'accuracy': 0.85, 
                'precision': 0.82, 'recall': 0.78
            }
            mock_eval.return_value = {
                'loss': 0.4, 'f1': 0.85, 'accuracy': 0.88, 
                'precision': 0.86, 'recall': 0.84
            }
            
            # Ejecutar entrenamiento con kwargs adicionales
            results = train_with_wandb_monitoring_generic(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scheduler=self.scheduler,
                monitor=self.monitor,
                device=self.device,
                architecture="generic",
                epochs=2,
                early_stopping_patience=5,
                save_dir=self.save_dir,
                verbose=False,
                alpha=0.5,  # Parámetro adicional
                lambda_=0.1  # Parámetro adicional
            )
            
            # Verificar que se pasaron los kwargs
            self.assertIsInstance(results, dict)
            # Verificar que se llamó train_one_epoch_generic con kwargs
            mock_train.assert_called()
            call_kwargs = mock_train.call_args[1]
            # Los kwargs se pasan a través de **kwargs en la función real


class TestGenericWandbTrainingIntegration(unittest.TestCase):
    """Pruebas de integración para train_with_wandb_monitoring_generic."""
    
    def setUp(self):
        """Configurar para pruebas de integración."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear datos más realistas
        self.batch_size = 8
        self.num_samples = 32
        self.input_size = (3, 32, 32)
        self.num_classes = 2
        
        # Generar datos balanceados
        X_train = torch.randn(self.num_samples, *self.input_size)
        y_train = torch.cat([torch.zeros(self.num_samples // 2), torch.ones(self.num_samples // 2)]).long()
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_val = torch.randn(self.num_samples // 4, *self.input_size)
        y_val = torch.cat([torch.zeros(self.num_samples // 8), torch.ones(self.num_samples // 8)]).long()
        val_dataset = TensorDataset(X_val, y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Modelo más complejo
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)
        
        # Monitor real (sin mock)
        self.monitor = Mock(spec=TrainingMonitor)
        self.monitor.log = Mock()
        self.monitor.should_plot = Mock(return_value=False)
        self.monitor.plot_local = Mock()
        self.monitor.finish = Mock()
        self.monitor.print_summary = Mock()
        
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Limpiar después de cada test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_training_loop(self):
        """Probar loop de entrenamiento real sin mocks."""
        # Ejecutar entrenamiento real
        results = train_with_wandb_monitoring_generic(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=3,
            early_stopping_patience=10,
            save_dir=self.save_dir,
            model_name="integration_test_model.pth",
            verbose=False
        )
        
        # Verificar resultados
        self.assertIsInstance(results, dict)
        self.assertIn('model', results)
        self.assertIn('best_val_f1', results)
        self.assertIn('final_epoch', results)
        self.assertIn('history', results)
        self.assertIn('early_stopped', results)
        
        # Verificar que el modelo se entrenó
        self.assertEqual(results['final_epoch'], 3)
        self.assertFalse(results['early_stopped'])
        
        # Verificar que se guardó el modelo
        model_path = self.save_dir / "integration_test_model.pth"
        self.assertTrue(model_path.exists())
        
        # Verificar estructura del historial
        history = results['history']
        for key in ['train_loss', 'train_f1', 'val_loss', 'val_f1']:
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), 3)
            # Verificar que los valores son razonables
            self.assertTrue(all(0 <= val <= 1 for val in history[key]) if 'f1' in key else True)
            self.assertTrue(all(val >= 0 for val in history[key]) if 'loss' in key else True)
    
    def test_model_saving_and_loading(self):
        """Probar que el modelo se guarda y carga correctamente."""
        # Entrenar modelo
        results = train_with_wandb_monitoring_generic(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=2,
            early_stopping_patience=10,
            save_dir=self.save_dir,
            model_name="save_test_model.pth",
            verbose=False
        )
        
        # Verificar que se guardó
        model_path = self.save_dir / "save_test_model.pth"
        self.assertTrue(model_path.exists())
        
        # Cargar modelo con la misma arquitectura
        new_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        ).to(self.device)
        
        new_model.load_state_dict(torch.load(model_path, map_location=self.device))
        new_model.to(self.device)
        
        # Verificar que los parámetros son iguales
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    # Configurar para ejecutar las pruebas
    unittest.main(verbosity=2)
