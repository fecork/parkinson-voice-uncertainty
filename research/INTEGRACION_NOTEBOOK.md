# 🔧 INTEGRACIÓN DEL SISTEMA DE HIPERPARÁMETROS EN EL NOTEBOOK

## 📋 INSTRUCCIONES PASO A PASO

### 1. **INSERTAR CELDA DEL SELECTOR** (Nueva celda)

**Ubicación**: Después de la celda de configuración (Cell 3), antes de la celda de Optuna

**Contenido**: Copia el código de `research/notebook_cell_selector.py`

**Propósito**: Permite elegir entre parámetros de Ibarra o Optuna

### 2. **REEMPLAZAR CELDA DE OPTUNA** (Celda existente)

**Ubicación**: La celda que dice "OPTUNA OPTIMIZATION"

**Contenido**: Reemplaza con el código de `research/notebook_cell_optuna_replacement.py`

**Propósito**: Ejecuta Optuna solo si `USE_IBARRA_HYPERPARAMETERS = False`

### 3. **REEMPLAZAR CELDA DE CREACIÓN DEL MODELO** (Celda existente)

**Ubicación**: La celda que crea el modelo CNN2D

**Contenido**: Reemplaza con el código de `research/notebook_cell_model_creation.py`

**Propósito**: Usa los parámetros seleccionados (Ibarra o Optuna)

### 4. **REEMPLAZAR CELDA DE CONFIGURACIÓN DE ENTRENAMIENTO** (Celda existente)

**Ubicación**: La celda que configura el entrenamiento final

**Contenido**: Reemplaza con el código de `research/notebook_cell_training_config.py`

**Propósito**: Usa los parámetros de entrenamiento seleccionados

## 🎯 RESULTADO FINAL

Después de la integración, tu notebook tendrá:

### **MODO IBARRA** (`USE_IBARRA_HYPERPARAMETERS = True`)
- ✅ Usa parámetros exactos del paper de Ibarra 2023
- ✅ Salta la optimización de Optuna
- ✅ Entrena directamente con parámetros del paper
- ✅ Más rápido (no hay optimización)

### **MODO OPTUNA** (`USE_IBARRA_HYPERPARAMETERS = False`)
- ✅ Ejecuta optimización automática de hiperparámetros
- ✅ Busca mejores parámetros que Ibarra
- ✅ Usa los mejores parámetros encontrados
- ✅ Más lento (requiere optimización)

## 🔄 CÓMO CAMBIAR ENTRE MODOS

### Para usar Ibarra:
```python
USE_IBARRA_HYPERPARAMETERS = True
```

### Para usar Optuna:
```python
USE_IBARRA_HYPERPARAMETERS = False
```

## 📊 VARIABLES DISPONIBLES

Después de ejecutar la celda del selector, tendrás:

- `BEST_PARAMS`: Diccionario con todos los hiperparámetros
- `USE_IBARRA`: Boolean indicando si usar Ibarra o Optuna
- `HYPERPARAMETER_SOURCE`: String con la fuente de los parámetros

## 🚀 EJEMPLO DE USO

```python
# 1. Configurar modo
USE_IBARRA_HYPERPARAMETERS = True  # o False para Optuna

# 2. Ejecutar celda del selector
# (Se ejecuta automáticamente)

# 3. Usar variables en el resto del notebook
print(f"Usando parámetros de: {HYPERPARAMETER_SOURCE}")
print(f"Batch size: {BEST_PARAMS['batch_size']}")
print(f"Learning rate: {BEST_PARAMS['learning_rate']}")
```

## ⚠️ NOTAS IMPORTANTES

1. **Ejecutar en orden**: Las celdas deben ejecutarse en el orden correcto
2. **Variables globales**: El selector crea variables que usan las otras celdas
3. **Configuración persistente**: Los cambios se guardan en `config/hyperparameter_config.json`
4. **Fallback automático**: Si no hay archivos de Optuna, usa Ibarra automáticamente

## 🎉 BENEFICIOS

- **Flexibilidad**: Cambia entre Ibarra y Optuna con una sola variable
- **Consistencia**: Mismo código para ambos modos
- **Trazabilidad**: Siempre sabes qué parámetros estás usando
- **Reproducibilidad**: Fácil de replicar experimentos
- **Mantenimiento**: Un solo lugar para cambiar parámetros
