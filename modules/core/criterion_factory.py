#!/usr/bin/env python3
"""
Criterion Factory - Creación de Criterios de Pérdida
===================================================

Funciones para crear criterios de pérdida siguiendo el paper de Ibarra 2023:
- Loss de clase: SIN pesos (fidelidad al paper)
- Loss de dominio: CON pesos balanceados (para balancear adversario)

Uso:
    from modules.core.criterion_factory import create_criterions_paper_style
    
    criterions = create_criterions_paper_style(
        y_train=y_train,
        y_domain_train=y_domain_train,
        class_weights_enabled=False,
        domain_weights_enabled=True,
        device=device
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.utils.class_weight import compute_class_weight


def compute_domain_weights_from_indices(
    train_indices: list,
    y_domain_healthy: torch.Tensor,
    y_domain_parkinson: torch.Tensor,
    method: str = 'balanced'
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Calcula pesos de dominio usando solo datos de TRAIN.
    
    Args:
        train_indices: Índices del conjunto de entrenamiento
        y_domain_healthy: Labels de dominio para muestras healthy
        y_domain_parkinson: Labels de dominio para muestras parkinson
        method: Método de cálculo ('balanced' usa sklearn)
    
    Returns:
        Tuple de (domain_weights_tensor, domain_classes)
    """
    # Recolectar labels de dominio solo de train
    y_domain_train = []
    
    # Recolectar labels de dominio de healthy (train)
    for idx in train_indices:
        if idx < len(y_domain_healthy):
            y_domain_train.append(y_domain_healthy[idx].item())
    
    # Recolectar labels de dominio de parkinson (train)
    for idx in train_indices:
        if idx >= len(y_domain_healthy):
            parkinson_idx = idx - len(y_domain_healthy)
            if parkinson_idx < len(y_domain_parkinson):
                y_domain_train.append(y_domain_parkinson[parkinson_idx].item())
    
    y_domain_train = np.array(y_domain_train)
    
    # Calcular pesos balanceados para los dominios
    domain_classes = np.unique(y_domain_train)
    
    if method == 'balanced':
        domain_weights = compute_class_weight(
            class_weight='balanced',
            classes=domain_classes,
            y=y_domain_train
        )
    else:
        # Método inverse_frequency
        class_counts = np.bincount(y_domain_train)
        domain_weights = len(y_domain_train) / (len(domain_classes) * class_counts)
    
    domain_weights_tensor = torch.FloatTensor(domain_weights)
    
    return domain_weights_tensor, domain_classes, y_domain_train


def create_criterions_paper_style(
    y_train: torch.Tensor,
    train_indices: Optional[list] = None,
    y_domain_healthy: Optional[torch.Tensor] = None,
    y_domain_parkinson: Optional[torch.Tensor] = None,
    class_weights_enabled: bool = False,
    domain_weights_enabled: bool = True,
    method: str = 'balanced',
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> Dict[str, nn.Module]:
    """
    Crea criterios de pérdida siguiendo el paper de Ibarra 2023.
    
    Args:
        y_train: Labels de clase para entrenamiento
        train_indices: Índices del conjunto de entrenamiento (para dominio)
        y_domain_healthy: Labels de dominio para muestras healthy
        y_domain_parkinson: Labels de dominio para muestras parkinson
        class_weights_enabled: Si aplicar pesos a la pérdida de clase
        domain_weights_enabled: Si aplicar pesos a la pérdida de dominio
        method: Método de cálculo ('balanced' o 'inverse_frequency')
        device: Dispositivo PyTorch
        verbose: Si imprimir información
    
    Returns:
        Diccionario con criterios:
        - 'criterion_class': Para cabeza de clasificación
        - 'criterion_domain': Para cabeza adversaria
        - 'criterion_final': Para compatibilidad (usa criterion_class)
        - 'domain_weights': Pesos de dominio (si aplicable)
        - 'class_weights': Pesos de clase (si aplicable)
    """
    if verbose:
        print("="*70)
        print("CREANDO CRITERIOS DE PÉRDIDA (Paper Ibarra 2023)")
        print("="*70)
    
    # ============================================================
    # CRITERIO PARA CLASE
    # ============================================================
    class_weights_tensor = None
    
    if class_weights_enabled:
        # Calcular class weights para clase
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights_tensor = class_weights.to(device)
        criterion_class = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        if verbose:
            print(f"✅ Criterion clase CON pesos: {class_weights.tolist()}")
    else:
        criterion_class = nn.CrossEntropyLoss()
        if verbose:
            print(f"✅ Criterion clase SIN pesos (como en paper)")
    
    # ============================================================
    # CRITERIO PARA DOMINIO
    # ============================================================
    domain_weights_tensor = None
    domain_classes = None
    
    if domain_weights_enabled and train_indices is not None:
        if y_domain_healthy is not None and y_domain_parkinson is not None:
            # Calcular pesos de dominio con datos de TRAIN
            domain_weights_tensor, domain_classes, y_domain_train = compute_domain_weights_from_indices(
                train_indices=train_indices,
                y_domain_healthy=y_domain_healthy,
                y_domain_parkinson=y_domain_parkinson,
                method=method
            )
            
            domain_weights_tensor = domain_weights_tensor.to(device)
            criterion_domain = nn.CrossEntropyLoss(weight=domain_weights_tensor)
            
            if verbose:
                print(f"✅ Criterion dominio CON pesos balanceados")
                print(f"\n📊 Pesos de dominio (calculados con TRAIN):")
                for cls, weight in zip(domain_classes, domain_weights_tensor.cpu().numpy()):
                    count = np.sum(y_domain_train == cls)
                    print(f"   - Dominio {cls}: weight={weight:.3f}, samples={count}")
        else:
            criterion_domain = nn.CrossEntropyLoss()
            if verbose:
                print(f"⚠️  Criterion dominio SIN pesos (faltan datos de dominio)")
    else:
        criterion_domain = nn.CrossEntropyLoss()
        if verbose:
            print(f"⚠️  Criterion dominio SIN pesos (deshabilitado)")
    
    # ============================================================
    # CRITERIO FINAL (compatibilidad)
    # ============================================================
    criterion_final = criterion_class
    
    if verbose:
        print(f"\n📊 Resumen:")
        print(f"   - criterion_class: {'CON pesos' if class_weights_enabled else 'SIN pesos'}")
        print(f"   - criterion_domain: {'CON pesos' if (domain_weights_enabled and domain_weights_tensor is not None) else 'SIN pesos'}")
        print(f"   - criterion_final: usa criterion_class")
        print("="*70)
    
    return {
        'criterion_class': criterion_class,
        'criterion_domain': criterion_domain,
        'criterion_final': criterion_final,
        'domain_weights': domain_weights_tensor,
        'class_weights': class_weights_tensor,
        'domain_classes': domain_classes
    }


def create_criterions_simple(
    y_train: torch.Tensor,
    enable_class_weights: bool = False,
    device: torch.device = torch.device('cpu')
) -> Dict[str, nn.Module]:
    """
    Versión simplificada para crear solo criterio de clase.
    
    Args:
        y_train: Labels de clase para entrenamiento
        enable_class_weights: Si aplicar pesos
        device: Dispositivo PyTorch
    
    Returns:
        Diccionario con 'criterion' y 'class_weights'
    """
    class_weights_tensor = None
    
    if enable_class_weights:
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return {
        'criterion': criterion,
        'class_weights': class_weights_tensor
    }






