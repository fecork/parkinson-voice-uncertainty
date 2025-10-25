# Migración de Talos a Optuna

Este documento describe la migración del proyecto de Talos a Optuna para la optimización de hiperparámetros.

## ¿Por qué migrar a Optuna?

### Problemas con Talos:
- ❌ **No mantenido**: Sin actualizaciones desde 2020
- ❌ **Errores de instalación**: Problemas con scipy y dependencias
- ❌ **Solo Keras**: Diseñado para Keras, no para PyTorch
- ❌ **Sin soporte**: Comunidad inactiva

### Ventajas de Optuna:
- ✅ **Activamente mantenido**: Actualizaciones frecuentes
- ✅ **Fácil instalación**: Sin problemas de dependencias
- ✅ **PyTorch nativo**: Diseñado para PyTorch
- ✅ **Más eficiente**: Pruning automático de trials malos
- ✅ **Mejores visualizaciones**: Gráficas interactivas
- ✅ **Mejor documentación**: Amplia documentación y ejemplos

## Cambios Realizados

### 1. Dependencias

**Antes (`requirements.txt`):**
```
talos>=0.6.5
```

**Ahora:**
```
optuna>=3.0.0
```

**Instalación:**
```bash
pip install optuna
```

### 2. Configuración

**Antes:**
```python
TALOS_CONFIG = {
    "enabled": True,
    "experiment_name": "cnn2d_talos_optimization",
    "search_method": "random",
    "fraction_limit": 0.3,
    "n_epochs": 20,
    "metric": "f1"
}
```

**Ahora:**
```python
OPTUNA_CONFIG = {
    "enabled": True,
    "experiment_name": "cnn2d_optuna_optimization",
    "n_trials": 30,  # Número de configuraciones a probar
    "n_epochs_per_trial": 20,
    "metric": "f1",
    "direction": "maximize"
}
```

### 3. Imports

**Antes:**
```python
from modules.core.cnn2d_talos_wrapper import optimize_cnn2d
from modules.core.talos_optimization import TalosOptimizer
```

**Ahora:**
```python
from modules.core.cnn2d_optuna_wrapper import optimize_cnn2d, create_cnn2d_optimizer
from modules.core.optuna_optimization import OptunaOptimizer
```

### 4. Uso

**Antes (Talos):**
```python
from modules.core.cnn2d_talos_wrapper import optimize_cnn2d

results = optimize_cnn2d(
    X_train, y_train, X_val, y_val,
    input_shape=(1, 65, 41),
    fraction_limit=0.3,
    n_epochs=20
)
```

**Ahora (Optuna):**
```python
from modules.core.cnn2d_optuna_wrapper import optimize_cnn2d

results = optimize_cnn2d(
    X_train, y_train, X_val, y_val,
    input_shape=(1, 65, 41),
    n_trials=30,
    n_epochs_per_trial=20,
    save_dir='results/optuna'
)
```

## Nuevas Funcionalidades

### 1. Pruning Automático
Optuna automáticamente detiene trials que no están progresando bien:

```python
# Automático en Optuna
# Los trials malos se podan early, ahorrando tiempo
```

### 2. Visualizaciones Mejoradas

```python
optimizer = create_cnn2d_optimizer(input_shape=(1, 65, 41))
optimizer.optimize(X_train, y_train, X_val, y_val)

# Visualizaciones interactivas
fig1 = optimizer.plot_optimization_history()
fig2 = optimizer.plot_param_importances()
fig3 = optimizer.plot_parallel_coordinate()

# Guardar visualizaciones
fig1.write_html('optimization_history.html')
```

### 3. Análisis de Resultados

```python
# Obtener análisis completo
analysis = optimizer.analyze_results()

print(f"Mejores parámetros: {analysis['best_params']}")
print(f"Mejor valor: {analysis['best_value']}")
print(f"Importancia de parámetros: {analysis['param_importances']}")
```

## Archivos Nuevos

- `modules/core/optuna_optimization.py` - Core de Optuna
- `modules/core/cnn2d_optuna_wrapper.py` - Wrapper para CNN2D
- `docs/MIGRATION_TALOS_TO_OPTUNA.md` - Este documento

## Archivos Deprecados

Los siguientes archivos de Talos permanecen en el proyecto pero están deprecados:
- `modules/core/talos_optimization.py`
- `modules/core/cnn2d_talos_wrapper.py`
- `modules/core/talos_evaluator.py`
- `modules/core/talos_analysis.py`
- `modules/core/talos_visualization.py`

**Nota:** Estos archivos se mantendrán temporalmente por compatibilidad pero no recibirán actualizaciones.

## Ejemplo Completo

```python
# Configurar entorno
from modules.core.notebook_setup import setup_notebook
ENV, PATHS = setup_notebook()

# Cargar datos
# ... (tu código de carga de datos)

# Optimizar con Optuna
from modules.core.cnn2d_optuna_wrapper import optimize_cnn2d

results = optimize_cnn2d(
    X_train, y_train, X_val, y_val,
    input_shape=(1, 65, 41),
    n_trials=30,
    n_epochs_per_trial=20,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir=PATHS['results'] / 'optuna'
)

# Ver resultados
print(f"Mejor F1: {results['best_value']:.4f}")
print("\nMejores hiperparámetros:")
for param, value in results['best_params'].items():
    print(f"  {param}: {value}")

# Entrenar modelo final con mejores parámetros
best_params = results['best_params']
# ... (entrenar con best_params)
```

## Solución de Problemas

### Error: "No module named 'optuna'"
```bash
pip install optuna>=3.0.0
```

### Configuración de Pruning
Si quieres desactivar el pruning:

```python
from optuna.pruners import NopPruner

optimizer = OptunaOptimizer(
    model_wrapper=wrapper,
    pruner=NopPruner()  # No pruning
)
```

### Almacenamiento Persistente
Para guardar estudios entre ejecuciones:

```python
optimizer = OptunaOptimizer(
    model_wrapper=wrapper,
    storage='sqlite:///optuna_study.db'  # Base de datos SQLite
)
```

## Referencias

- [Documentación oficial de Optuna](https://optuna.readthedocs.io/)
- [Tutorial de Optuna con PyTorch](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [Ejemplos de Optuna](https://github.com/optuna/optuna-examples)

## Soporte

Para preguntas o problemas con la migración, consulta:
1. Este documento
2. El README principal del proyecto
3. La documentación de Optuna

