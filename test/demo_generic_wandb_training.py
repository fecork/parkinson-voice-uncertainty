#!/usr/bin/env python3
"""
Demostración de train_with_wandb_monitoring_generic
==================================================

Script de demostración que muestra cómo usar la función de entrenamiento genérica
con monitoreo de Weights & Biases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import os
import sys

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic
from modules.core.training_monitor import TrainingMonitor


def create_demo_data(batch_size=8, num_samples=64):
    """Crear datos de demostración."""
    input_size = (3, 32, 32)  # Para CNN2D
    num_classes = 2
    
    # Generar datos de entrenamiento
    X_train = torch.randn(num_samples, *input_size)
    y_train = torch.randint(0, num_classes, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Generar datos de validación
    X_val = torch.randn(num_samples // 4, *input_size)
    y_val = torch.randint(0, num_classes, (num_samples // 4,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_demo_model(input_size=(3, 32, 32), num_classes=2):
    """Crear modelo de demostración."""
    model = nn.Sequential(
        nn.Conv2d(input_size[0], 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(32, num_classes)
    )
    return model


def demo_basic_training():
    """Demostración de entrenamiento básico."""
    print("=" * 70)
    print("DEMOSTRACIÓN: Entrenamiento Básico con Arquitectura Genérica")
    print("=" * 70)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Crear datos y modelo
    train_loader, val_loader = create_demo_data()
    model = create_demo_model().to(device)
    
    # Crear optimizador y criterio
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Crear monitor (mock para demostración)
    monitor = TrainingMonitor(
        project_name="demo-parkinson-voice",
        experiment_name="generic_training_demo",
        config={
            "architecture": "generic",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 0.001
        }
    )
    
    # Directorio temporal para guardar modelo
    temp_dir = tempfile.mkdtemp()
    save_dir = Path(temp_dir)
    
    try:
        # Ejecutar entrenamiento
        print("Iniciando entrenamiento...")
        results = train_with_wandb_monitoring_generic(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            monitor=monitor,
            device=device,
            architecture="generic",
            epochs=5,
            early_stopping_patience=10,
            save_dir=save_dir,
            model_name="demo_model.pth",
            verbose=True
        )
        
        # Mostrar resultados
        print("\n" + "=" * 50)
        print("RESULTADOS DEL ENTRENAMIENTO")
        print("=" * 50)
        print(f"Mejor val_f1: {results['best_val_f1']:.4f}")
        print(f"Épocas entrenadas: {results['final_epoch']}")
        print(f"Early stopping: {'Sí' if results['early_stopped'] else 'No'}")
        print(f"Modelo guardado: {save_dir / 'demo_model.pth'}")
        
        # Mostrar historial
        history = results['history']
        print(f"\nHistorial de entrenamiento:")
        print(f"  Train F1: {[f'{f1:.3f}' for f1 in history['train_f1']]}")
        print(f"  Val F1:   {[f'{f1:.3f}' for f1 in history['val_f1']]}")
        print(f"  Train Loss: {[f'{loss:.3f}' for loss in history['train_loss']]}")
        print(f"  Val Loss:   {[f'{loss:.3f}' for loss in history['val_loss']]}")
        
        return results
        
    finally:
        # Limpiar archivos temporales
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_training_without_scheduler():
    """Demostración de entrenamiento sin scheduler."""
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN: Entrenamiento Sin Scheduler")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_demo_data()
    model = create_demo_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    monitor = TrainingMonitor(
        project_name="demo-parkinson-voice",
        experiment_name="no_scheduler_demo",
        config={"architecture": "generic", "epochs": 3}
    )
    
    temp_dir = tempfile.mkdtemp()
    save_dir = Path(temp_dir)
    
    try:
        results = train_with_wandb_monitoring_generic(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=None,  # Sin scheduler
            monitor=monitor,
            device=device,
            architecture="generic",
            epochs=3,
            early_stopping_patience=10,
            save_dir=save_dir,
            verbose=True
        )
        
        print(f"Entrenamiento completado: {results['final_epoch']} épocas")
        print(f"Mejor val_f1: {results['best_val_f1']:.4f}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_training_with_custom_forward():
    """Demostración de entrenamiento con función forward personalizada."""
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN: Entrenamiento Con Función Forward Personalizada")
    print("=" * 70)
    
    def custom_forward_fn(model, x):
        """Función forward personalizada que agrega logging."""
        print(f"  Forward pass con input shape: {x.shape}")
        return model(x)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_demo_data()
    model = create_demo_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    monitor = TrainingMonitor(
        project_name="demo-parkinson-voice",
        experiment_name="custom_forward_demo",
        config={"architecture": "generic", "epochs": 2}
    )
    
    temp_dir = tempfile.mkdtemp()
    save_dir = Path(temp_dir)
    
    try:
        results = train_with_wandb_monitoring_generic(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=None,
            monitor=monitor,
            device=device,
            architecture="generic",
            epochs=2,
            early_stopping_patience=10,
            save_dir=save_dir,
            forward_fn=custom_forward_fn,
            verbose=True
        )
        
        print(f"Entrenamiento completado: {results['final_epoch']} épocas")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_training_with_kwargs():
    """Demostración de entrenamiento con parámetros adicionales."""
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN: Entrenamiento Con Parámetros Adicionales")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_demo_data()
    model = create_demo_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    monitor = TrainingMonitor(
        project_name="demo-parkinson-voice",
        experiment_name="kwargs_demo",
        config={"architecture": "generic", "epochs": 2, "alpha": 0.5, "lambda_": 0.1}
    )
    
    temp_dir = tempfile.mkdtemp()
    save_dir = Path(temp_dir)
    
    try:
        results = train_with_wandb_monitoring_generic(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=None,
            monitor=monitor,
            device=device,
            architecture="generic",
            epochs=2,
            early_stopping_patience=10,
            save_dir=save_dir,
            verbose=True,
            alpha=0.5,      # Parámetro adicional
            lambda_=0.1,    # Parámetro adicional
            custom_param="test"  # Otro parámetro personalizado
        )
        
        print(f"Entrenamiento completado: {results['final_epoch']} épocas")
        print("Parámetros adicionales procesados correctamente")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Ejecutar todas las demostraciones."""
    print("DEMOSTRACIÓN DE train_with_wandb_monitoring_generic")
    print("=" * 70)
    print("Esta función permite entrenar cualquier modelo PyTorch con")
    print("monitoreo de Weights & Biases de manera genérica.")
    print("=" * 70)
    
    try:
        # Demostración 1: Entrenamiento básico
        demo_basic_training()
        
        # Demostración 2: Sin scheduler
        demo_training_without_scheduler()
        
        # Demostración 3: Con función forward personalizada
        demo_training_with_custom_forward()
        
        # Demostración 4: Con parámetros adicionales
        demo_training_with_kwargs()
        
        print("\n" + "=" * 70)
        print("TODAS LAS DEMOSTRACIONES COMPLETADAS EXITOSAMENTE")
        print("=" * 70)
        print("La función train_with_wandb_monitoring_generic funciona correctamente")
        print("y puede ser usada para entrenar cualquier arquitectura de modelo.")
        
    except Exception as e:
        print(f"\nError durante la demostración: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
