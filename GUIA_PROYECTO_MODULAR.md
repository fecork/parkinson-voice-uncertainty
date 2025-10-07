# 🎯 Guía del Proyecto Modular - Parkinson Voice Detection

## 📁 Estructura del Proyecto (ACTUALIZADO)

```
parkinson-voice-uncertainty/
│
├── 📓 parkinson_voice_analysis.ipynb    # ⭐ NOTEBOOK PRINCIPAL
│   └── Importa y usa módulos de /modules
│
├── 📦 modules/                          # ⭐ MÓDULOS REUTILIZABLES
│   ├── __init__.py                      # Inicialización del paquete
│   ├── preprocessing.py                 # ✅ Funciones de preprocesamiento
│   ├── augmentation.py                  # ✅ Técnicas de augmentation
│   └── utils.py                         # ✅ Utilidades comunes
│
├── 📄 Templates & Docs
│   ├── 02_data_augmentation_template.py # Template para notebook 02
│   ├── README_MODULAR.md                # Documentación completa
│   └── GUIA_PROYECTO_MODULAR.md         # Esta guía
│
└── 🎵 vowels/                           # Datos de audio
    └── *.egg, *.nsp
```

---

## 🔄 Flujo de Trabajo

### 1️⃣ Notebook Principal (`parkinson_voice_analysis.ipynb`)

Este es tu notebook central que:
- ✅ **Importa** funciones de los módulos
- ✅ **Ejecuta** el pipeline de preprocesamiento
- ✅ **Genera** el dataset PyTorch
- ✅ **NO duplica** código

#### Estructura de Celdas:

```python
# Celda 1: Setup Colab/Local
# Celda 2: Imports (IMPORTA MÓDULOS)
# Celda 3: Título markdown
# Celda 4: Cargar archivos (USA utils.get_data_path())
# Celda 5: Título markdown
# Celda 6: Mensaje (SIN FUNCIONES DUPLICADAS)
# Celda 7: Pipeline dataset (USA build_full_pipeline())
# Celda 8: Verificación rápida
# Celda 9: Configuración (USA constantes del módulo)
# Celda 10: Estadísticas (USA utils.print_dataset_stats())
# Celda 11: Visualización
```

---

## 🎯 ¿Cómo Usar el Notebook Principal?

### Paso 1: Ejecutar Setup

```python
# Celda 1 - Ya está configurada
# Detecta automáticamente si estás en Colab o Local
```

### Paso 2: Importar Módulos

```python
# Celda 2 - Ejecutar esta celda
import sys
from pathlib import Path

# Agregar módulos al path
sys.path.insert(0, str(Path.cwd()))

# Importar módulos propios
from modules import preprocessing, utils
from modules.preprocessing import (
    SAMPLE_RATE, N_MELS, TARGET_FRAMES,
    load_audio_file,
    preprocess_audio_paper,
    get_preprocessing_config,
    print_preprocessing_config
)
```

**Salida esperada:**
```
✅ Librerías y módulos cargados correctamente
🔧 Dispositivo: cpu
📦 PyTorch: 2.8.0+cpu
📦 Librosa: 0.11.0
============================================================
⚙️ Preprocessing Configuration:
  • SAMPLE_RATE: 44100
  • WINDOW_MS: 400
  • N_MELS: 65
  ...
```

### Paso 3: Cargar Archivos

```python
# Celda 4 - Ejecutar
DATA_PATH = utils.get_data_path()  # Auto-detecta Colab/Local
audio_files = utils.list_audio_files(DATA_PATH, extension="*.egg")
```

**Salida esperada:**
```
📂 Ruta de datos: ./vowels
📁 Encontrados 13 archivos *.egg
🔍 Primeros 5 archivos de audio:
  1. 1580-a_h-egg.egg
  2. 1580-a_l-egg.egg
  ...
```

### Paso 4: Procesar Dataset

```python
# Celda 7 - Este usa las funciones del módulo
# Ya está configurado, solo ejecutar
```

**Salida esperada:**
```
🔄 Procesando 13 archivos...
✅ 121 muestras generadas
📊 PyTorch tensors listos:
  - X: (121, 1, 65, 41)
  - y_task: (121,)
  ...
```

---

## 📚 Módulos Disponibles

### 1. `modules/preprocessing.py`

**Constantes:**
```python
SAMPLE_RATE = 44100
WINDOW_MS = 400
OVERLAP = 0.5
N_MELS = 65
HOP_MS = 10
FFT_WINDOW_A = 40
FFT_WINDOW_OTHER = 25
TARGET_FRAMES = 41
```

**Funciones:**
```python
# Cargar audio
audio, sr = load_audio_file('archivo.egg')

# Segmentar
segments = segment_audio(audio, sr=44100)

# Crear espectrograma
spec = create_mel_spectrogram(segment, vowel_type='a')

# Normalizar
norm_spec = normalize_spectrogram(spec)

# Pipeline completo
spectrograms, segments = preprocess_audio_paper('archivo.egg', vowel_type='a')

# Obtener configuración
config = get_preprocessing_config()
print_preprocessing_config()
```

### 2. `modules/augmentation.py`

**Audio Domain:**
```python
from modules.augmentation import (
    time_stretch,
    pitch_shift,
    add_white_noise,
    augment_audio
)

# Usar
audio_aug = time_stretch(audio, rate=1.1)
audio_aug = pitch_shift(audio, sr=44100, n_steps=2)
audio_aug = add_white_noise(audio, noise_factor=0.005)
```

**Spectrogram Domain:**
```python
from modules.augmentation import (
    spec_augment,
    mixup_spectrograms,
    random_erasing,
    augment_spectrogram
)

# Usar
spec_aug = spec_augment(spectrogram)
mixed, lam = mixup_spectrograms(spec1, spec2)
spec_aug = augment_spectrogram(spectrogram)
```

### 3. `modules/utils.py`

**Detección de Entorno:**
```python
from modules import utils

# Detectar entorno
if utils.is_colab():
    utils.setup_colab_environment()

# Obtener rutas
data_path = utils.get_data_path()  # Auto-detecta
modules_path = utils.get_modules_path()

# Agregar al path
utils.add_modules_to_path()
```

**Operaciones de Archivos:**
```python
# Listar archivos
audio_files = utils.list_audio_files('./vowels', extension='*.egg')

# Crear directorio
utils.ensure_directory('./results/models')
```

**Visualización:**
```python
# Mostrar estadísticas
utils.print_dataset_stats(X, y_task, y_domain, metadata)

# Mostrar configuración
utils.print_config(config, title="Mi Config")

# Headers
utils.print_section_header("ENTRENAMIENTO")
```

**Tracking de Experimentos:**
```python
# Guardar configuración
config = {'lr': 0.001, 'batch_size': 32}
utils.save_experiment_config(config, 'exp1_config.json')

# Cargar configuración
config = utils.load_experiment_config('exp1_config.json')
```

---

## 🔧 Modificar el Código

### ¿Dónde hacer cambios?

| Si quieres cambiar... | Edita... |
|----------------------|----------|
| **Parámetros del preprocesamiento** | `modules/preprocessing.py` (constantes al inicio) |
| **Funciones de preprocesamiento** | `modules/preprocessing.py` |
| **Técnicas de augmentation** | `modules/augmentation.py` |
| **Utilidades comunes** | `modules/utils.py` |
| **Pipeline del notebook principal** | `parkinson_voice_analysis.ipynb` (Celda 7) |

### Ejemplo: Cambiar Sample Rate

**ANTES (❌ Malo):**
- Cambiar en el notebook principal
- Duplicado en varios lugares
- Difícil de mantener

**AHORA (✅ Bueno):**
```python
# Editar modules/preprocessing.py
SAMPLE_RATE = 48000  # Cambiar aquí

# Guardar archivo
# Reiniciar kernel del notebook
# ¡Listo! Todos los notebooks usan el nuevo valor
```

---

## 🆕 Crear Nuevos Notebooks

### Template para Notebook Nuevo:

```python
# ============================================================
# NUEVO NOTEBOOK: [Nombre]
# ============================================================

# 1. Setup paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# 2. Importar módulos
from modules import preprocessing, augmentation, utils

# 3. Setup entorno
if utils.is_colab():
    utils.setup_colab_environment()
    utils.add_modules_to_path()

# 4. Tu código aquí
data_path = utils.get_data_path()
audio_files = utils.list_audio_files(data_path)

# ... resto del código
```

### Notebooks Recomendados para Crear:

1. **`02_data_augmentation.ipynb`** ← Ya tienes el template
2. **`03_model_training.ipynb`** - Entrenar modelos CNN
3. **`04_evaluation.ipynb`** - Evaluar y validar
4. **`05_inference.ipynb`** - Predicciones en nuevos datos

---

## 🐛 Troubleshooting

### Problema 1: "ModuleNotFoundError: No module named 'modules'"

**Causa:** El path no está configurado correctamente.

**Solución:**
```python
import sys
from pathlib import Path

# Agregar directorio del proyecto
sys.path.insert(0, str(Path.cwd()))

# Verificar
print("Python path:", sys.path[:3])

# Ahora importar
from modules import preprocessing
```

### Problema 2: "Los cambios en el módulo no se reflejan"

**Causa:** Python cachea los módulos importados.

**Solución:**
```python
# Opción 1: Reiniciar kernel del notebook (recomendado)
# Runtime → Restart runtime

# Opción 2: Recargar módulo
import importlib
importlib.reload(preprocessing)
importlib.reload(augmentation)
importlib.reload(utils)
```

### Problema 3: "Audio files not found"

**Causa:** Ruta incorrecta.

**Solución:**
```python
# Verificar ruta
data_path = utils.get_data_path()
print(f"Buscando en: {data_path}")

# Verificar contenido
from pathlib import Path
if Path(data_path).exists():
    files = list(Path(data_path).glob('*'))
    print(f"Archivos encontrados: {len(files)}")
    print(files[:5])
else:
    print("❌ Directorio no existe")
```

---

## ✅ Ventajas del Diseño Modular

### 1. **Sin Duplicación de Código**
- ❌ Antes: Funciones duplicadas en cada notebook
- ✅ Ahora: Funciones en un solo lugar (`modules/`)

### 2. **Fácil Mantenimiento**
- ❌ Antes: Cambiar en 5 notebooks diferentes
- ✅ Ahora: Cambiar en 1 archivo `.py`

### 3. **Reutilización**
- ❌ Antes: Copy/paste entre notebooks
- ✅ Ahora: `from modules import preprocessing`

### 4. **Testing**
- ❌ Antes: Imposible hacer tests unitarios
- ✅ Ahora: Puedes crear `tests/test_preprocessing.py`

### 5. **Producción**
- ❌ Antes: Refactorizar todo del notebook
- ✅ Ahora: Los módulos ya son código de producción

### 6. **Colaboración**
- ❌ Antes: Merge conflicts constantes
- ✅ Ahora: Cada persona trabaja en su notebook

---

## 📝 Checklist de Verificación

Antes de ejecutar un notebook, verifica:

- [ ] Ejecutaste la celda de imports (Celda 2)
- [ ] No hay errores de `ModuleNotFoundError`
- [ ] La ruta de datos está correcta (`utils.get_data_path()`)
- [ ] Los archivos `.egg` están en la carpeta `vowels/`
- [ ] Tienes las dependencias instaladas (librosa, torch, etc.)

---

## 🎓 Próximos Pasos

1. **Completar Data Augmentation:**
   - Usar `02_data_augmentation_template.py`
   - Crear notebook `02_data_augmentation.ipynb`

2. **Crear Módulo de Modelos:**
   - Crear `modules/models.py` con arquitecturas CNN
   - Crear `modules/training.py` con loops de entrenamiento

3. **Crear Notebook de Training:**
   - `03_model_training.ipynb`
   - Importar de `modules.models` y `modules.training`

4. **Testing:**
   - Crear `tests/test_preprocessing.py`
   - Usar pytest para tests automatizados

---

## 📚 Recursos

- **Módulos Python Oficiales:** `modules/preprocessing.py`, `augmentation.py`, `utils.py`
- **Documentación Completa:** `README_MODULAR.md`
- **Templates:** `02_data_augmentation_template.py`
- **Paper Original:** Ver `ENTREGA1.pdf`

---

## 💡 Tips & Best Practices

1. **Siempre importar módulos al inicio del notebook**
2. **No duplicar código - usar funciones de módulos**
3. **Documentar cambios importantes en módulos**
4. **Reiniciar kernel después de modificar módulos**
5. **Usar `utils` para operaciones comunes**
6. **Mantener notebooks limpios y concisos**

---

**¿Preguntas? Revisa `README_MODULAR.md` o los archivos en `modules/`** 🚀

