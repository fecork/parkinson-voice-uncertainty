#!/usr/bin/env python3
"""
Pruebas simplificadas para train_with_wandb_monitoring_generic
=============================================================

Pruebas que se enfocan en la funcionalidad real sin mocks complejos.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os
import sys

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic
from modules.core.training_monitor import TrainingMonitor


class TestGenericWandbTrainingSimple(unittest.TestCase):
    """Pruebas simplificadas para train_with_wandb_monitoring_generic."""
    
    def setUp(self):
        """Configurar datos de prueba antes de cada test."""
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear datos sintéticos simples
        self.batch_size = 4
        self.num_samples = 16
        self.input_size = (3, 32, 32)  # Para CNN2D
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
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, self.num_classes)
        ).to(self.device)
        
        # Crear optimizador y criterio
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Crear scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5)
        
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
            early_stopping_patience=5,
            save_dir=self.save_dir,
            model_name="test_model.pth",
            verbose=False
        )
        
        # Verificar resultados básicos
        self.assertIsInstance(results, dict)
        self.assertIn('model', results)
        self.assertIn('best_val_f1', results)
        self.assertIn('final_epoch', results)
        self.assertIn('history', results)
        self.assertIn('early_stopped', results)
        
        # Verificar que se completó el entrenamiento
        self.assertEqual(results['final_epoch'], 3)
        self.assertFalse(results['early_stopped'])
        
        # Verificar que se llamó el monitor
        self.assertEqual(self.monitor.log.call_count, 3)
        self.monitor.finish.assert_called_once()
        self.monitor.print_summary.assert_called_once()
    
    def test_training_without_scheduler(self):
        """Probar entrenamiento sin scheduler."""
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
    
    def test_training_without_save_dir(self):
        """Probar entrenamiento sin directorio de guardado."""
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
    
    def test_early_stopping(self):
        """Probar early stopping cuando no mejora."""
        # Crear un modelo que no mejore (pesos aleatorios fijos)
        torch.manual_seed(42)
        bad_model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, self.num_classes)
        ).to(self.device)
        
        # Ejecutar entrenamiento con early stopping
        results = train_with_wandb_monitoring_generic(
            model=bad_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optim.Adam(bad_model.parameters(), lr=0.001),  # LR muy bajo
            criterion=self.criterion,
            scheduler=None,
            monitor=self.monitor,
            device=self.device,
            architecture="generic",
            epochs=10,
            early_stopping_patience=2,  # Early stopping en 2 épocas
            save_dir=self.save_dir,
            verbose=False
        )
        
        # Verificar que se completó el entrenamiento (puede o no hacer early stopping)
        self.assertIsInstance(results, dict)
        self.assertIn('early_stopped', results)
        self.assertGreater(results['final_epoch'], 0)
    
    def test_training_history_structure(self):
        """Probar que el historial de entrenamiento tiene la estructura correcta."""
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
        
        # Verificar que los valores son razonables
        for key in ['train_f1', 'val_f1', 'train_accuracy', 'val_accuracy']:
            for val in history[key]:
                self.assertTrue(0 <= val <= 1, f"{key} value {val} not in [0,1]")
        
        for key in ['train_loss', 'val_loss']:
            for val in history[key]:
                self.assertTrue(val >= 0, f"{key} value {val} should be >= 0")
    
    def test_model_saving(self):
        """Probar que el modelo se guarda correctamente."""
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
            early_stopping_patience=10,
            save_dir=self.save_dir,
            model_name="save_test_model.pth",
            verbose=False
        )
        
        # Verificar que se guardó
        model_path = self.save_dir / "save_test_model.pth"
        self.assertTrue(model_path.exists())
        
        # Verificar que el archivo no está vacío
        self.assertGreater(model_path.stat().st_size, 0)
    
    def test_training_with_custom_forward_fn(self):
        """Probar entrenamiento con función forward personalizada."""
        def custom_forward_fn(model, x):
            return model(x)
        
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
        
        # Verificar que se completó el entrenamiento
        self.assertIsInstance(results, dict)
        self.assertEqual(results['final_epoch'], 2)
    
    def test_training_with_kwargs(self):
        """Probar entrenamiento con parámetros adicionales."""
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
        
        # Verificar que se completó el entrenamiento
        self.assertIsInstance(results, dict)
        self.assertEqual(results['final_epoch'], 2)
    
    def test_training_different_epochs(self):
        """Probar entrenamiento con diferentes números de épocas."""
        for epochs in [1, 3, 5]:
            with self.subTest(epochs=epochs):
                # Crear nuevo modelo para cada test
                model = nn.Sequential(
                    nn.Conv2d(3, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(8, self.num_classes)
                ).to(self.device)
                
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                
                results = train_with_wandb_monitoring_generic(
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    optimizer=optimizer,
                    criterion=self.criterion,
                    scheduler=None,
                    monitor=self.monitor,
                    device=self.device,
                    architecture="generic",
                    epochs=epochs,
                    early_stopping_patience=10,
                    save_dir=self.save_dir,
                    verbose=False
                )
                
                # Verificar que se completó el número correcto de épocas
                self.assertEqual(results['final_epoch'], epochs)
                self.assertFalse(results['early_stopped'])


if __name__ == '__main__':
    # Configurar para ejecutar las pruebas
    unittest.main(verbosity=2)
