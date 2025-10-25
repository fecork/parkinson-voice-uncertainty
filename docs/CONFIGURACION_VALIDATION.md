# Validación de Configuración de Entrenamiento

## Comparación: Configuración Actual vs. Recomendaciones

### ✅ 1. Métrica a vigilar: val_f1_macro o val_balanced_accuracy

**❌ PROBLEMA ACTUAL:**
- La función `train_model()` usa **solo `val_loss`** para early stopping y guardar el mejor modelo
- Líneas 199-202 en `modules/models/cnn2d/training.py`:
  ```python
  early_stopping = EarlyStopping(
      patience=early_stopping_patience,
      mode="min",  # Minimizar val_loss
  )
  ```
- Líneas 252-254: Guarda el modelo basándose en `val_loss`:
  ```python
  if val_metrics["loss"] < best_val_loss:
      best_val_loss = val_metrics["loss"]
      best_model_state = model.state_dict().copy()
  ```

**✅ RECOMENDACIÓN:**
- Cambiar a `val_f1_macro` o `val_balanced_accuracy` como métrica principal
- En datasets desbalanceados (65.8% PD vs 34.2% HC), `val_loss` puede ser engañoso

**ACCIÓN NECESARIA:** Modificar `train_model()` para usar `val_f1` como métrica de monitoreo.

---

### ⚠️ 2. EarlyStopping: patience=8-10, restore_best_weights=True

**ESTADO ACTUAL:**
- **Patience**: 15 épocas (en `TRAINING_CONFIG`)
  ```python
  TRAINING_CONFIG = {
      "n_epochs": 100,
      "early_stopping_patience": 15,  # ⚠️ Debería ser 8-10
      ...
  }
  ```

**✅ IMPLEMENTACIÓN:**
- Ya tiene `restore_best_weights=True` implícitamente (líneas 254-255 guardan best_model_state)
- Línea 302 restaura el mejor modelo:
  ```python
  model.load_state_dict(best_model_state)
  ```

**ACCIÓN NECESARIA:** Reducir patience de 15 a 8-10.

---

### ✅ 3. Épocas máximas: 60-100

**ESTADO ACTUAL:**
- ✅ **100 épocas** configuradas correctamente
  ```python
  TRAINING_CONFIG = {
      "n_epochs": 100,  # ✅ OK
      ...
  }
  ```

**ACCIÓN:** Ninguna, está correcto.

---

### ❌ 4. SGD: momentum=0.9, nesterov=True, weight_decay=1e-4

**PROBLEMA ACTUAL:**
- **Configuración incompleta**:
  ```python
  OPTIMIZER_CONFIG = {
      "type": "SGD",
      "learning_rate": 0.1,
      "momentum": 0.9,        # ✅ OK
      "weight_decay": 0.0     # ❌ Debería ser 1e-4
  }
  ```
- ❌ **Falta `nesterov=True`**
- ❌ **`weight_decay=0.0`** debería ser `1e-4`

**IMPLEMENTACIÓN ACTUAL (celda 26 del notebook):**
```python
optimizer_final = optim.SGD(
    best_model.parameters(),
    0.1,  # learning_rate
    0.9,  # momentum
    0.0,  # weight_decay  ❌ Debería ser 1e-4
)
# ❌ Falta nesterov=True
```

**ACCIÓN NECESARIA:** 
1. Agregar `nesterov=True`
2. Cambiar `weight_decay` de `0.0` a `1e-4`

---

### ✅ 5. Class weights: activos

**ESTADO ACTUAL:**
- ✅ **Habilitado** en la configuración:
  ```python
  CLASS_WEIGHTS_CONFIG = {
      "enabled": True,
      "method": "inverse_frequency"
  }
  ```

- ✅ **Implementado** en celda 26:
  ```python
  class_counts = torch.bincount(y_train)
  class_weights = 1.0 / class_counts.float()
  class_weights = class_weights / class_weights.sum()
  criterion_final = nn.CrossEntropyLoss(weight=class_weights.to(device))
  ```

**ACCIÓN:** Ninguna, está correcto.

---

### ❌ 6. Checkpoint por fold: mejor epoch por fold + promedio de 10 folds

**PROBLEMA ACTUAL:**
- ❌ El notebook **NO implementa 10-fold CV completo**
- Solo usa **1 fold** del K-fold para entrenamiento:
  ```python
  # Para este notebook, usaremos el primer fold como ejemplo
  # En el paper real se promedian los resultados de los 10 folds
  train_indices = fold_splits[0]["train"]  # ❌ Solo usa fold 0
  val_indices = fold_splits[0]["val"]
  ```

**RECOMENDACIÓN:**
- Implementar loop completo de 10 folds
- Guardar mejor modelo de cada fold
- Reportar métricas promedio ± desviación estándar de los 10 folds

**CÓDIGO NECESARIO:**
```python
# Iterar sobre los 10 folds
for fold_idx in range(10):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/10")
    print(f"{'='*70}")
    
    # Obtener índices del fold
    train_indices = fold_splits[fold_idx]["train"]
    val_indices = fold_splits[fold_idx]["val"]
    
    # Entrenar modelo para este fold
    # Guardar mejor modelo del fold
    # Evaluar en validation del fold
    
# Calcular métricas promedio de todos los folds
mean_f1 = np.mean([fold['f1'] for fold in fold_results])
std_f1 = np.std([fold['f1'] for fold in fold_results])
print(f"F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
```

**NOTA:** Ya existe `train_model_da_kfold()` en `training.py` que implementa esto correctamente.

---

## 📋 Resumen de Acciones Necesarias

### Críticas (❌):
1. **Cambiar métrica de early stopping** de `val_loss` a `val_f1_macro`
2. **Agregar `nesterov=True`** al optimizador SGD
3. **Cambiar `weight_decay`** de `0.0` a `1e-4`
4. **Implementar 10-fold CV completo** (actualmente solo usa 1 fold)

### Menores (⚠️):
5. **Reducir patience** de 15 a 8-10 épocas

### Correctas (✅):
- ✅ Épocas máximas: 100
- ✅ SGD momentum: 0.9
- ✅ Class weights: habilitado
- ✅ Restore best weights: implementado

---

## 🔧 Archivos a Modificar

1. **`research/cnn_training.ipynb`**:
   - Celda 3: Actualizar `OPTIMIZER_CONFIG` y `TRAINING_CONFIG`
   - Celda 26: Agregar `nesterov=True` y `weight_decay=1e-4` al optimizador
   - Agregar loop de 10-fold CV completo

2. **`modules/models/cnn2d/training.py`**:
   - Modificar `train_model()` para usar `val_f1` en lugar de `val_loss` para early stopping
   - Agregar parámetro `monitor_metric` (default: "f1")

3. **`modules/core/cnn2d_optuna_wrapper.py`**:
   - Verificar que Optuna también use la métrica correcta

---

## 📊 Impacto Esperado

- **Métrica correcta**: Mejor selección de modelo en datasets desbalanceados
- **Nesterov momentum**: Convergencia más rápida y estable
- **Weight decay**: Mejor regularización, reduce overfitting
- **10-fold CV**: Estimación más robusta del rendimiento real del modelo
- **Patience reducida**: Ahorra tiempo de entrenamiento sin sacrificar rendimiento

