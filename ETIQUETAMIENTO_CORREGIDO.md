# 🏷️ Corrección de Etiquetamiento

## ❌ Problema Identificado

El pipeline original en `modules/dataset.py` etiqueta **INCORRECTAMENTE** basándose en la condición del nombre del archivo:

```python
# ❌ INCORRECTO
def map_condition_to_task(condition: str) -> int:
    mapping = {
        "h": 1,    # ❌ NO es Parkinson, es pitch HIGH
        "l": 0,    # ❌ NO es Control, es pitch LOW  
        "n": 0,    # ❌ Es pitch NORMAL
        "lhl": 1,  # ❌ Es pitch Low-High-Low
    }
```

### ¿Qué significa cada condición?

Las condiciones en los nombres de archivo (`h`, `l`, `n`, `lhl`) se refieren a **TONOS DE PITCH** (entonación), NO a la condición de salud:

- `h` = High pitch (tono alto)
- `l` = Low pitch (tono bajo)
- `n` = Normal pitch (tono normal)
- `lhl` = Low-High-Low pitch (patrón de entonación)

### Ejemplo de nombres de archivo:

```
data/vowels_healthy/
  ├── 97-a_h-egg.egg      → Speaker 97, vocal 'a', HIGH pitch → HC (0)
  ├── 1143-a_l-egg.egg    → Speaker 1143, vocal 'a', LOW pitch → HC (0)
  ├── 1705-a_n-egg.egg    → Speaker 1705, vocal 'a', NORMAL pitch → HC (0)

data/vowels_pk/
  ├── 1580-a_h-egg.egg    → Speaker 1580, vocal 'a', HIGH pitch → PD (1)
  ├── 1580-a_l-egg.egg    → Speaker 1580, vocal 'a', LOW pitch → PD (1)
  ├── 1580-a_n-egg.egg    → Speaker 1580, vocal 'a', NORMAL pitch → PD (1)
```

## ✅ Solución Implementada

La etiqueta correcta viene del **DIRECTORIO**, no del nombre del archivo:

```python
# ✅ CORRECTO
if directorio == "data/vowels_healthy":
    label = 0  # HC (Healthy Control)
elif directorio == "data/vowels_pk":
    label = 1  # PD (Parkinson Disease)
```

### Implementación en `train_cnn.py`

```python
# Procesar HC
hc_result = build_full_pipeline(hc_files)
hc_dataset = hc_result["torch_ds"]

# CORREGIR ETIQUETAS: Todos HC deben ser 0
for i in range(len(hc_dataset.y_task)):
    hc_dataset.y_task[i] = 0

# Procesar PD
pd_result = build_full_pipeline(pd_files)
pd_dataset = pd_result["torch_ds"]

# CORREGIR ETIQUETAS: Todos PD deben ser 1
for i in range(len(pd_dataset.y_task)):
    pd_dataset.y_task[i] = 1
```

## 🧪 Verificación

Ejecuta el script de verificación:

```bash
python verify_labels.py
```

Esto mostrará:
1. ❌ Etiquetas incorrectas (antes de corregir)
2. ✅ Etiquetas correctas (después de corregir)
3. 📊 Distribución del dataset combinado
4. 📋 Ejemplos de etiquetamiento

### Output esperado:

```
✅ VERIFICACIÓN EXITOSA: Todas las etiquetas son correctas

📊 DATASET COMBINADO
====================================================================
Total segmentos: XXX
Etiqueta 0 (HC): YYY segmentos
Etiqueta 1 (PD): ZZZ segmentos
```

## 📊 Distribución Esperada

Basándose en tus datos:

- **HC (Healthy)**: 13 speakers, ~150-200 segmentos → Label 0
- **PD (Parkinson)**: 1 speaker (1580), ~15-20 segmentos → Label 1

⚠️ **Desbalance importante**: Tienes ~10x más datos HC que PD. Por eso es crítico usar:
- `--use_class_weights` en entrenamiento
- Split speaker-independent
- Métricas balanceadas (F1-Score)

## 🎯 Equivalencia con tu código

Tu código de referencia:

```python
# Tu código ✓
y_healthy = torch.zeros(len(X_healthy), dtype=torch.long)   # 0
y_parkinson = torch.ones(len(X_parkinson), dtype=torch.long) # 1

X = torch.cat([X_healthy, X_parkinson], dim=0)
y = torch.cat([y_healthy, y_parkinson], dim=0)
```

Nuestro código equivalente (ahora corregido):

```python
# Nuestro código corregido ✓
hc_dataset.y_task[:] = 0  # Todos HC → 0
pd_dataset.y_task[:] = 1  # Todos PD → 1

combined = ConcatDataset([hc_dataset, pd_dataset])
# Ahora combined tiene las etiquetas correctas
```

## 🔍 Cómo verificar manualmente

En Python:

```python
from modules.dataset import build_full_pipeline
from pathlib import Path

# Cargar HC
hc_files = list(Path("data/vowels_healthy").glob("*.egg"))
hc_result = build_full_pipeline(hc_files)
hc_dataset = hc_result["torch_ds"]

# Ver etiquetas ANTES de corregir
print("HC labels antes:", set(hc_dataset.y_task.numpy()))  # ❌ {0, 1} INCORRECTO

# Corregir
for i in range(len(hc_dataset.y_task)):
    hc_dataset.y_task[i] = 0

# Ver etiquetas DESPUÉS de corregir
print("HC labels después:", set(hc_dataset.y_task.numpy()))  # ✅ {0} CORRECTO
```

## 📝 Resumen

| Concepto | Valor | Significado |
|----------|-------|-------------|
| **Label 0** | HC | Healthy Control (sanos) |
| **Label 1** | PD | Parkinson Disease (enfermos) |
| **Condición "h"** | Pitch | High pitch (tono alto) |
| **Condición "l"** | Pitch | Low pitch (tono bajo) |
| **Condición "n"** | Pitch | Normal pitch (tono normal) |
| **Condición "lhl"** | Pitch | Low-High-Low pattern |

✅ **Ahora el etiquetamiento es correcto** y coincide con tu lógica de `torch.zeros` para HC y `torch.ones` para PD.

