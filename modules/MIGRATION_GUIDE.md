# Guía de Migración v4.0

## 📦 Cambios en la Estructura de Imports

### Versión 4.0 - Código Compartido Centralizado

La estructura de módulos ha cambiado para eliminar duplicación de código.

## 🔄 Actualizar Imports en Notebooks

### ❌ Antes (v2.0 - v3.0)

```python
# INCORRECTO - NO funciona
from modules import preprocessing, augmentation, dataset, visualization
from modules.preprocessing import SAMPLE_RATE
from modules.augmentation import create_augmented_dataset
from modules.dataset import build_full_pipeline
from modules.visualization import plot_spectrogram_comparison
```

### ✅ Después (v4.0)

```python
# CORRECTO - Estructura modular
from modules.core import preprocessing
from modules.core.preprocessing import SAMPLE_RATE
from modules.data.augmentation import create_augmented_dataset
from modules.core.dataset import build_full_pipeline
from modules.core.visualization import plot_spectrogram_comparison
```

## 📋 Tabla de Migración Rápida

| Módulo Antiguo | Nuevo Import (v4.0) |
|----------------|---------------------|
| `modules.preprocessing` | `modules.core.preprocessing` |
| `modules.augmentation` | `modules.data.augmentation` |
| `modules.dataset` | `modules.core.dataset` |
| `modules.visualization` | `modules.core.visualization` |
| `modules.cache_utils` | `modules.data.cache_utils` |
| `modules.utils` | `modules.core.utils` |
| `modules.sequence_dataset` | `modules.core.sequence_dataset` ⭐ NUEVO |
| `modules.cnn_model` | `modules.models.cnn2d.model` |
| `modules.cnn_training` | `modules.models.cnn2d.training` |

## 🆕 Nuevos Módulos v4.0

### `modules/models/common/` ⭐ NUEVO
Componentes compartidos entre modelos:

```python
from modules.models.common.layers import (
    FeatureExtractor,        # CNN 2D compartido
    GradientReversalLayer,   # GRL para Domain Adaptation
    ClassifierHead,          # Cabeza FC reutilizable
)
```

**Usado por:**
- ✅ CNN2D / CNN2D_DA
- ✅ CNN1D_DA
- ✅ Time-CNN-BiLSTM-DA

### `modules/core/sequence_dataset.py` ⭐ NUEVO
Funciones para secuencias LSTM:

```python
from modules.core.sequence_dataset import (
    group_spectrograms_to_sequences,
    save_sequence_cache,
    load_sequence_cache,
    SequenceLSTMDataset,
)
```

### `modules/models/lstm_da/` ⭐ NUEVO
Modelo Time-CNN-BiLSTM con Domain Adaptation:

```python
from modules.models.lstm_da import TimeCNNBiLSTM_DA
from modules.models.lstm_da.training import train_model_da_kfold
```

## 🛠️ Cómo Actualizar tus Notebooks

### Paso 1: Actualizar Cell de Imports

Reemplaza:
```python
from modules import preprocessing
```

Por:
```python
from modules.core import preprocessing
```

### Paso 2: Verificar Imports Específicos

Reemplaza:
```python
from modules.preprocessing import SAMPLE_RATE
from modules.augmentation import create_augmented_dataset
```

Por:
```python
from modules.core.preprocessing import SAMPLE_RATE
from modules.data.augmentation import create_augmented_dataset
```

### Paso 3: Ejecutar Notebook

```bash
# Reiniciar kernel
Kernel → Restart & Clear Output

# Ejecutar todo
Cell → Run All
```

## ✅ Verificación

Si ves este mensaje, los imports están correctos:
```
🎵 DATA PREPROCESSING & AUGMENTATION - v4.0
✅ Librerías cargadas correctamente
```

Si ves `ImportError`, revisa que:
1. Estés usando la estructura v4.0 (`modules.core.*`, `modules.data.*`)
2. Los módulos existan en la nueva ubicación
3. Los `__init__.py` exporten correctamente

## 🎯 Notebooks Actualizados

- ✅ `data_preprocessing.ipynb` - Actualizado a v4.0
- ℹ️ `cnn_training.ipynb` - Verificar si necesita actualización
- ℹ️ `cnn_da_training.ipynb` - Verificar si necesita actualización
- ℹ️ `cnn1d_da_training.ipynb` - Verificar si necesita actualización
- ℹ️ `cnn_uncertainty_training.ipynb` - Verificar si necesita actualización

## 📞 Ayuda

Si encuentras errores de import, revisa:
1. Esta guía de migración
2. El archivo `modules/__init__.py` (debe ser v4.0.0)
3. La estructura de carpetas en `modules/`

---

**Última actualización:** v4.0.0 - Octubre 2025



