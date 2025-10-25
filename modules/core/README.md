# Core Module - Hyperparameter Optimization

Este módulo centralizado proporciona funcionalidades reutilizables para optimización de hiperparámetros con Talos y evaluación de modelos.

## Estructura

```
modules/core/
├── __init__.py                    # Exports principales
├── environment.py                # Detección de entorno (Local/Colab)
├── talos_optimization.py         # Sistema centralizado de Talos
├── model_evaluation.py           # Evaluación y comparación de modelos
└── README.md                      # Esta documentación
```

## Características Principales

### 1. Detección de Entorno y Configuración de Rutas

- **Detección automática**: Identifica si el código corre en Local o Colab
- **Rutas dinámicas**: Configura automáticamente rutas según el entorno
- **Google Drive**: Funciones para montar y usar Google Drive en Colab
- **Portabilidad**: Mismo código funciona en local y Colab sin cambios

### 2. Sistema Centralizado de Talos

- **TalosOptimizer**: Clase principal para optimización con Talos
- **TalosModelWrapper**: Interfaz abstracta para wrappers de modelos
- **Funciones de utilidad**: Análisis, evaluación y visualización

### 2. Evaluación de Modelos

- **ModelEvaluator**: Evaluación estándar de modelos
- **Comparación**: Comparar múltiples modelos
- **Métricas**: Accuracy, F1, Precision, Recall, AUC
- **Visualización**: Gráficas de comparación y matrices de confusión

### 3. Extensibilidad

- Fácil extensión para nuevas arquitecturas
- Wrappers específicos por modelo
- Funciones de conveniencia

## Uso Básico

### Configuración de Entorno

```python
from modules.core.environment import setup_environment

# Detectar entorno automáticamente y configurar rutas
ENV, PATHS = setup_environment(verbose=True)

# Usar las rutas en tu código
cache_healthy = PATHS['cache_original'] / "healthy_ibarra.pkl"
cache_parkinson = PATHS['cache_augmented'] / "augmented_dataset.pkl"
results_dir = PATHS['results'] / "mi_experimento"

# El mismo código funciona en Local y en Colab
print(f"Entorno: {ENV}")  # 'local' o 'colab'
print(f"Ruta de cache: {cache_healthy}")
```

**Características importantes:**
- **Búsqueda automática de raíz**: En local, encuentra automáticamente la raíz del proyecto aunque ejecutes desde subdirectorios (como `research/` o `notebooks/`)
- **Sin configuración manual**: No necesitas especificar rutas, todo se detecta automáticamente
- **Funciona desde cualquier lugar**: Ejecuta notebooks desde cualquier carpeta del proyecto

**En Local:**
```
CONFIGURACIÓN DE ENTORNO
======================================================================
Entorno detectado: LOCAL
Ruta base: /path/to/parkinson-voice-uncertainty
Cache original: /path/to/parkinson-voice-uncertainty/cache/original
Cache augmented: /path/to/parkinson-voice-uncertainty/cache/augmented

MODO LOCAL: Usando rutas relativas
======================================================================
```

**En Colab:**
```
CONFIGURACIÓN DE ENTORNO
======================================================================
Entorno detectado: COLAB
Ruta base: /content/drive/Othercomputers/ZenBook/parkinson-voice-uncertainty
Cache original: /content/drive/Othercomputers/.../cache/original
Cache augmented: /content/drive/Othercomputers/.../cache/augmented

MODO COLAB: Usando rutas de Google Drive
======================================================================
```

### Montar Google Drive en Colab

```python
from modules.core.environment import mount_google_drive

# Montar Google Drive automáticamente si estás en Colab
if mount_google_drive(verbose=True):
    ENV, PATHS = setup_environment()
    # Ahora puedes acceder a tus archivos
```

### Para CNN2D

```python
from modules.models.cnn2d.talos_wrapper import create_cnn2d_optimizer

# Crear optimizador
optimizer = create_cnn2d_optimizer(
    experiment_name="mi_experimento",
    fraction_limit=0.1  # 10% de combinaciones
)

# Ejecutar optimización
results_df = optimizer.optimize(X_train, y_train, X_val, y_val)

# Obtener mejores parámetros
best_params = optimizer.get_best_params()

# Análisis de resultados
analysis = optimizer.analyze_results()
```

### Para Arquitecturas Personalizadas

```python
from modules.core.talos_optimization import TalosOptimizer, TalosModelWrapper

class MiModeloWrapper(TalosModelWrapper):
    def create_model(self, params):
        return MiModelo(**params)
    
    def train_model(self, model, train_loader, val_loader, params, n_epochs):
        # Implementar entrenamiento
        return f1_score, metrics
    
    def get_search_params(self):
        return {
            'param1': [valor1, valor2],
            'param2': [valor3, valor4]
        }

# Usar wrapper personalizado
wrapper = MiModeloWrapper()
optimizer = TalosOptimizer(wrapper)
results = optimizer.optimize(X_train, y_train, X_val, y_val)
```

## Evaluación de Modelos

```python
from modules.core.model_evaluation import ModelEvaluator, compare_models

# Evaluar un modelo
evaluator = ModelEvaluator(modelo)
metrics = evaluator.evaluate(X_test, y_test)

# Comparar múltiples modelos
modelos = {
    'Modelo1': modelo1,
    'Modelo2': modelo2
}
results_df = compare_models(modelos, X_test, y_test)
```

## Funciones de Utilidad

### Análisis de Hiperparámetros

```python
from modules.core.talos_optimization import analyze_hyperparameter_importance

# Analizar importancia de hiperparámetros
analysis = analyze_hyperparameter_importance(results_df)
print("Top 5 parámetros más importantes:")
for param, importance in analysis['top_important']:
    print(f"  {param}: {importance:.4f}")
```

### Guardar Resultados

```python
from modules.core.model_evaluation import save_model_results

# Guardar modelo y métricas
save_model_results(
    model, 
    metrics, 
    "ruta/modelo",
    additional_info={'experiment': 'mi_experimento'}
)
```

## Arquitecturas Soportadas

### CNN2D
- Wrapper completo implementado
- Parámetros optimizables: filtros, kernels, dropout, learning rate
- Función de conveniencia: `optimize_cnn2d()`

### Arquitecturas Personalizadas
- Implementar `TalosModelWrapper`
- Definir `create_model()`, `train_model()`, `get_search_params()`
- Usar con `TalosOptimizer`

## Parámetros de Búsqueda por Defecto

```python
{
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.0001, 0.00001],
    'dropout_rate': [0.2, 0.3, 0.5],
    'weight_decay': [0.0, 1e-4, 1e-5],
    'optimizer': ['adam', 'sgd']
}
```

## Ejemplos Completos

Ver `research/talos_optimization_example.py` para ejemplos completos de uso.

## Ventajas del Sistema Centralizado

1. **Reutilización**: Un solo sistema para todas las arquitecturas
2. **Consistencia**: Misma interfaz para todos los modelos
3. **Extensibilidad**: Fácil agregar nuevas arquitecturas
4. **Mantenibilidad**: Código centralizado y organizado
5. **Testing**: Pruebas unitarias completas

## Migración desde Sistema Anterior

### Antes (Sistema Específico)
```python
from modules.models.cnn2d.talos_optimization import create_talos_model
f1, metrics = create_talos_model(X_train, y_train, X_val, y_val, params)
```

### Ahora (Sistema Centralizado)
```python
from modules.models.cnn2d.talos_wrapper import optimize_cnn2d
results = optimize_cnn2d(X_train, y_train, X_val, y_val)
```

## Próximos Pasos

1. Agregar wrappers para otras arquitecturas (CNN1D, LSTM, etc.)
2. Implementar optimización multi-objetivo
3. Agregar visualizaciones avanzadas
4. Integrar con sistemas de experimentación (MLflow, Weights & Biases)
