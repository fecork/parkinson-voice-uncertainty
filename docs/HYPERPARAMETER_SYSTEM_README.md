# Sistema de Configuración de Hiperparámetros: Ibarra vs Optuna

## 🎯 **Descripción**

Este sistema permite elegir fácilmente entre usar los hiperparámetros exactos del paper de Ibarra 2023 o los mejores hiperparámetros encontrados por Optuna, sin necesidad de modificar código manualmente.

## 📚 **Hiperparámetros de Ibarra (Paper)**

```python
# Valores exactos del paper de Ibarra 2023
kernel_size_1 = 6
kernel_size_2 = 9
depth_CL = 64          # filters_2
neurons_MLP = 32       # dense_units
drop_out = 0.2         # p_drop_conv
batch_size = 64
learning_rate = 0.1
momentum = 0.9
```

## 🔍 **Hiperparámetros de Optuna (Optimizados)**

```python
# Mejores valores encontrados por Optuna
kernel_size_1 = 4      # Optimizado
kernel_size_2 = 9      # Igual a Ibarra
filters_1 = 128        # Optimizado
filters_2 = 32         # Optimizado
dense_units = 64       # Optimizado
p_drop_conv = 0.2      # Igual a Ibarra
batch_size = 32        # Optimizado
learning_rate = 0.0005379125214937586  # Optimizado
```

## 🚀 **Uso Rápido**

### **Opción 1: En el Notebook**

1. **Reemplaza la celda de configuración** con la celda del selector:

```python
# ============================================================
# SELECTOR DE HIPERPARÁMETROS: IBARRA vs OPTUNA
# ============================================================

# 🔧 CONFIGURACIÓN PRINCIPAL - CAMBIA ESTE VALOR
USE_IBARRA_HYPERPARAMETERS = True  # True = Ibarra, False = Optuna

# ... resto del código del selector ...
```

2. **Cambia el valor** de `USE_IBARRA_HYPERPARAMETERS`:
   - `True` = Usar parámetros exactos del paper de Ibarra
   - `False` = Usar mejores parámetros de Optuna

3. **Ejecuta la celda** y el resto del notebook usará automáticamente los parámetros seleccionados.

### **Opción 2: En Código Python**

```python
from modules.core.hyperparameter_config import get_hyperparameters

# Usar parámetros de Ibarra
ibarra_params = get_hyperparameters(use_ibarra=True)

# Usar parámetros de Optuna
optuna_params = get_hyperparameters(use_ibarra=False)

# Crear modelo
model = CNN2D(
    kernel_size_1=ibarra_params["kernel_size_1"],
    kernel_size_2=ibarra_params["kernel_size_2"],
    filters_1=ibarra_params["filters_1"],
    filters_2=ibarra_params["filters_2"],
    dense_units=ibarra_params["dense_units"],
    p_drop_conv=ibarra_params["p_drop_conv"],
    p_drop_fc=ibarra_params["p_drop_fc"],
)
```

## 📁 **Archivos del Sistema**

```
modules/core/
├── hyperparameter_config.py          # Sistema principal
config/
├── hyperparameter_config.json        # Configuración guardada
research/
├── hyperparameter_selector_cell.py   # Celda para notebook
├── model_creation_cell.py            # Celda de creación de modelo
├── training_config_cell.py           # Celda de configuración de entrenamiento
├── notebook_integration_cells.py     # Todas las celdas para copiar
examples/
├── hyperparameter_usage_example.py   # Ejemplos de uso
test/
├── test_hyperparameter_system.py     # Pruebas unitarias
```

## 🔧 **Configuración Avanzada**

### **Gestión de Configuración**

```python
from modules.core.hyperparameter_config import HyperparameterManager

manager = HyperparameterManager()

# Guardar configuración para usar Ibarra
manager.save_config(use_ibarra=True)

# Guardar configuración para usar Optuna
manager.save_config(use_ibarra=False)

# Cargar configuración guardada
config = manager.load_config()
```

### **Comparación de Parámetros**

```python
from modules.core.hyperparameter_config import compare_hyperparameters

# Mostrar tabla comparativa
compare_hyperparameters()
```

## 📊 **Tabla Comparativa**

| **Parámetro** | **Ibarra 2023** | **Optuna** | **Diferencia** |
|---------------|-----------------|------------|----------------|
| kernel_size_1 | 6 | 4 | -2 (más pequeño) |
| kernel_size_2 | 9 | 9 | 0 (igual) |
| filters_1 | 32 | 128 | +96 (más filtros) |
| filters_2 | 64 | 32 | -32 (menos filtros) |
| dense_units | 32 | 64 | +32 (más neuronas) |
| p_drop_conv | 0.2 | 0.2 | 0 (igual) |
| p_drop_fc | 0.5 | 0.5 | 0 (igual) |
| batch_size | 64 | 32 | -32 (menos muestras) |
| learning_rate | 0.1 | 0.0005 | -0.0995 (mucho menor) |

## 🧪 **Pruebas**

```bash
# Ejecutar pruebas unitarias
python test/test_hyperparameter_system.py

# Ejecutar ejemplo de uso
python examples/hyperparameter_usage_example.py
```

## 💡 **Ventajas del Sistema**

### **✅ Flexibilidad**
- Cambio fácil entre Ibarra y Optuna
- Sin modificación manual de código
- Configuración persistente

### **✅ Transparencia**
- Comparación clara entre ambos enfoques
- Documentación de diferencias
- Trazabilidad de parámetros

### **✅ Robustez**
- Fallback automático a Ibarra si Optuna falla
- Validación de parámetros
- Manejo de errores

### **✅ Facilidad de Uso**
- Una sola variable para cambiar (`USE_IBARRA_HYPERPARAMETERS`)
- Variables globales preparadas (`BEST_PARAMS`)
- Integración transparente con notebooks

## 🎯 **Casos de Uso**

### **1. Reproducibilidad del Paper**
```python
USE_IBARRA_HYPERPARAMETERS = True  # Usar valores exactos del paper
```

### **2. Optimización de Rendimiento**
```python
USE_IBARRA_HYPERPARAMETERS = False  # Usar mejores valores de Optuna
```

### **3. Comparación de Métodos**
```python
# Entrenar con Ibarra
USE_IBARRA_HYPERPARAMETERS = True
# ... entrenar modelo ...

# Entrenar con Optuna
USE_IBARRA_HYPERPARAMETERS = False
# ... entrenar modelo ...

# Comparar resultados
```

## 🔄 **Migración desde Código Existente**

### **Antes:**
```python
# Valores hardcodeados
best_model = CNN2D(
    kernel_size_1=4,
    kernel_size_2=9,
    filters_1=128,
    filters_2=32,
    # ... más parámetros
)
```

### **Después:**
```python
# Valores dinámicos
BEST_PARAMS = get_hyperparameters(use_ibarra=True)  # o False

best_model = CNN2D(
    kernel_size_1=BEST_PARAMS["kernel_size_1"],
    kernel_size_2=BEST_PARAMS["kernel_size_2"],
    filters_1=BEST_PARAMS["filters_1"],
    filters_2=BEST_PARAMS["filters_2"],
    # ... más parámetros
)
```

## 🚨 **Troubleshooting**

### **Error: "BEST_PARAMS no está definido"**
- Ejecuta primero la celda del selector de hiperparámetros

### **Error: "No se encontraron resultados de Optuna"**
- El sistema usará automáticamente los parámetros de Ibarra como fallback

### **Error: "Módulo no encontrado"**
- Asegúrate de que el path del proyecto esté correctamente configurado

## 📝 **Notas Importantes**

1. **Los parámetros de Ibarra son exactos** del paper de 2023
2. **Los parámetros de Optuna son optimizados** para tu dataset específico
3. **El sistema es retrocompatible** con código existente
4. **La configuración se guarda automáticamente** en `config/hyperparameter_config.json`

## 🎉 **Conclusión**

Este sistema te permite:
- ✅ **Reproducir exactamente** el paper de Ibarra
- ✅ **Usar parámetros optimizados** de Optuna
- ✅ **Cambiar fácilmente** entre ambos enfoques
- ✅ **Comparar resultados** de manera transparente
- ✅ **Mantener código limpio** y organizado

**¡Solo cambia `USE_IBARRA_HYPERPARAMETERS = True/False` y listo!** 🚀
