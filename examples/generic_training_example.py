#!/usr/bin/env python3
"""
Ejemplo de uso del entrenamiento genérico con WanDB
=================================================

Este ejemplo muestra cómo usar train_with_wandb_monitoring con diferentes arquitecturas.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.core.wandb_training import train_with_wandb_monitoring, setup_wandb_training
from modules.core.generic_wandb_training import train_with_wandb_monitoring_generic


def example_cnn2d_training():
    """Ejemplo de entrenamiento CNN2D (compatibilidad hacia atrás)"""
    print("=" * 70)
    print("EJEMPLO: CNN2D Training (Compatibilidad hacia atrás)")
    print("=" * 70)
    
    # El código existente sigue funcionando igual
    # train_with_wandb_monitoring detecta automáticamente que es CNN2D
    pass


def example_cnn1d_training():
    """Ejemplo de entrenamiento CNN1D"""
    print("=" * 70)
    print("EJEMPLO: CNN1D Training")
    print("=" * 70)
    
    # Crear modelo CNN1D (ejemplo)
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.fc = nn.Linear(64, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.mean(dim=2)  # Global average pooling
            return self.fc(x)
    
    model = CNN1D()
    
    # Configurar monitoreo
    monitor = setup_wandb_training(
        config={"architecture": "CNN1D", "dataset": "Parkinson Voice"},
        wandb_config={"project": "parkinson-voice", "entity": "your-entity"},
        model=model,
        input_shape=(1, 100)  # Ejemplo
    )
    
    # Entrenar (detecta automáticamente que es CNN1D)
    # training_results = train_with_wandb_monitoring(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     scheduler=scheduler,
    #     monitor=monitor,
    #     device=device,
    #     epochs=100
    # )


def example_lstm_training():
    """Ejemplo de entrenamiento LSTM"""
    print("=" * 70)
    print("EJEMPLO: LSTM Training")
    print("=" * 70)
    
    # Crear modelo LSTM (ejemplo)
    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(100, 64, batch_first=True)
            self.fc = nn.Linear(64, 2)
        
        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.fc(hidden[-1])
    
    model = LSTM()
    
    # Entrenar con arquitectura específica
    # training_results = train_with_wandb_monitoring_generic(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     scheduler=scheduler,
    #     monitor=monitor,
    #     device=device,
    #     architecture="lstm",  # Especificar arquitectura
    #     epochs=100
    # )


def example_generic_training():
    """Ejemplo de entrenamiento genérico (cualquier arquitectura)"""
    print("=" * 70)
    print("EJEMPLO: Generic Training (Cualquier arquitectura)")
    print("=" * 70)
    
    # Crear modelo personalizado
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = CustomModel()
    
    # Entrenar con arquitectura genérica
    # training_results = train_with_wandb_monitoring_generic(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     scheduler=scheduler,
    #     monitor=monitor,
    #     device=device,
    #     architecture="generic",  # Usar funciones genéricas
    #     epochs=100
    # )


def example_domain_adaptation_training():
    """Ejemplo de entrenamiento con Domain Adaptation"""
    print("=" * 70)
    print("EJEMPLO: Domain Adaptation Training")
    print("=" * 70)
    
    # Para modelos con Domain Adaptation (CNN2D_DA, LSTM_DA)
    # training_results = train_with_wandb_monitoring_generic(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     scheduler=scheduler,
    #     monitor=monitor,
    #     device=device,
    #     architecture="cnn2d_da",  # o "lstm_da"
    #     epochs=100,
    #     alpha=1.0,  # Parámetro específico de DA
    #     lambda_=0.5  # Parámetro específico de DA
    # )


if __name__ == "__main__":
    print("Ejemplos de entrenamiento genérico con WanDB")
    print("=" * 70)
    
    example_cnn2d_training()
    example_cnn1d_training()
    example_lstm_training()
    example_generic_training()
    example_domain_adaptation_training()
    
    print("\n✅ Todos los ejemplos mostrados")
    print("=" * 70)
