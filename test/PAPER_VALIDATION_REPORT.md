# Reporte de Validación del Paper Ibarra 2023

**Fecha**: 2025-10-24 17:04:27
**Notebook**: research/cnn_training.ipynb
**Experimento**: Ibarra2023_VowelA_SVDD

## Resultados por Categoría

### Preprocesamiento

| Check | Esperado | Estado | Mensaje |
|-------|----------|--------|----------|
| Sampling rate | 44.1 kHz | ❌ FAIL | No se encuentra sampling rate de 44.1 kHz |
| Mel filters | 65 | ⚠️ WARNING | No se encuentra n_mels=65 explícitamente |
| Output shape | (65, 41) | ✅ PASS | Dimensiones de salida presentes |
| Vocal filtering | Solo vocal /a/ | ❌ FAIL | No se encuentra filtrado de vocal /a/ |

### Arquitectura

| Check | Esperado | Estado | Mensaje |
|-------|----------|--------|----------|
| Model class | CNN2D | ✅ PASS | Modelo CNN2D encontrado |
| Conv blocks | 2 bloques Conv2D (32, 64 filtros) | ⚠️ WARNING | No se detectan bloques convolucionales explícitamente |
| Dropout | 0.3 (conv), 0.5 (fc) | ⚠️ WARNING | Dropout presente pero valores no detectados |
| Input shape | (1, 65, 41) | ✅ PASS | Dimensiones de entrada correctas |

### Entrenamiento

| Check | Esperado | Estado | Mensaje |
|-------|----------|--------|----------|
| Optimizer | SGD | ✅ PASS | Optimizer SGD encontrado |
| Learning rate | 0.1 | ❌ FAIL | Learning rate no es 0.1 |
| Momentum | 0.9 | ❌ FAIL | Momentum no encontrado o incorrecto |
| Scheduler | StepLR | ✅ PASS | Scheduler StepLR encontrado |
| Scheduler step_size | 10 | ❌ FAIL | Step size no encontrado o incorrecto |
| Scheduler gamma | 0.1 | ❌ FAIL | Gamma no encontrado o incorrecto |
| Loss function | Weighted CrossEntropyLoss | ✅ PASS | CrossEntropyLoss con pesos encontrado |
| Metric | F1-macro | ✅ PASS | Métrica F1-macro encontrada |

### Validación

| Check | Esperado | Estado | Mensaje |
|-------|----------|--------|----------|
| Cross-validation | 10-fold CV | ⚠️ WARNING | K-Fold encontrado pero n_splits != 10 |
| Speaker stratification | Split por hablante | ✅ PASS | Estratificación por hablante detectada |
| Metric averaging | Promedio de métricas de folds | ✅ PASS | Promedio de métricas detectado |

### Hiperparámetros

| Check | Esperado | Estado | Mensaje |
|-------|----------|--------|----------|
| Batch size search space | [16, 32, 64] | ⚠️ WARNING | No todos los batch sizes encontrados |
| Dropout search space | [0.2, 0.5] | ✅ PASS | Valores de dropout presentes |
| FC units search space | [16, 32, 64] | ✅ PASS | FC units presentes |

## Recomendaciones de Corrección

1. **Sampling rate**: No se encuentra sampling rate de 44.1 kHz
2. **Vocal filtering**: No se encuentra filtrado de vocal /a/
3. **Learning rate**: Learning rate no es 0.1
4. **Momentum**: Momentum no encontrado o incorrecto
5. **Scheduler step_size**: Step size no encontrado o incorrecto
6. **Scheduler gamma**: Gamma no encontrado o incorrecto
