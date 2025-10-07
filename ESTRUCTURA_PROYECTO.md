# 🏗️ Estructura Final del Proyecto - Diagrama de Arquitectura

## 📐 Diagrama de Dependencias

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│       🎯 parkinson_voice_analysis.ipynb (NOTEBOOK PRINCIPAL)   │
│                                                                 │
│   Celda 2: Importa TODOS los módulos                          │
│   Celda 4: Usa utils.get_data_path()                          │
│   Celda 6: ← modules/preprocessing.py                         │
│   Celda 7: ← modules/dataset.py                               │
│   Celda 11: ← modules/visualization.py                        │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ IMPORTA
                     ▼
         ┌────────────────────────┐
         │                        │
         │   📦 modules/          │
         │                        │
         └───────────┬────────────┘
                     │
         ┌───────────┼───────────────────────┬─────────────┬──────────────┐
         │           │                       │             │              │
         ▼           ▼                       ▼             ▼              ▼
    ┌────────┐  ┌──────────┐          ┌─────────┐   ┌─────────┐   ┌──────────────┐
    │__init__│  │preproc...│          │dataset  │   │  utils  │   │visualization │
    │  .py   │  │  .py     │◄─────────│  .py    │   │   .py   │   │     .py      │
    └────────┘  └──────────┘  importa └─────────┘   └─────────┘   └──────────────┘
                     │                      │
                     │                      │
                 220 líneas             334 líneas
                     │                      │
              ┌──────┴───────┐       ┌─────┴──────┐
              │              │       │            │
         • load_audio    • segment  │  • build_  │
         • create_mel    • normalize│    full_   │
         • CONSTANTES              │    pipeline │
                                   │  • to_      │
                                   │    pytorch_ │
                                   │    tensors  │
                                   └────────────┘
```

---

## 🔄 Flujo de Datos

```
┌──────────────┐
│ vowels/*.egg │  Audio files
└──────┬───────┘
       │
       │ 1. Load
       ▼
┌─────────────────────────┐
│ preprocessing.py        │
│ • load_audio_file()     │  Carga y resamplea a 44.1kHz
│ • segment_audio()       │  Segmenta en ventanas 400ms
│ • create_mel_spectrogram│  Crea Mel spec 65×41
│ • normalize_spectrogram │  Normaliza z-score
└──────────┬──────────────┘
           │
           │ 2. Process
           ▼
┌─────────────────────────┐
│ dataset.py              │
│ • process_dataset()     │  Procesa todos los archivos
│ • to_pytorch_tensors()  │  Convierte a tensores
│ • VowelSegmentsDataset  │  Crea PyTorch Dataset
│ • build_full_pipeline() │  Orquesta todo
└──────────┬──────────────┘
           │
           │ 3. Output
           ▼
┌─────────────────────────┐
│ PyTorch Tensors         │
│ • X: (N, 1, 65, 41)    │  Espectrogramas
│ • y_task: (N,)         │  Etiquetas de tarea
│ • y_domain: (N,)       │  Etiquetas de dominio
│ • metadata: List       │  Metadatos
│ • torch_dataset        │  Dataset para DataLoader
└──────────┬──────────────┘
           │
           │ 4. Visualize
           ▼
┌─────────────────────────┐
│ visualization.py        │
│ • visualize_...()       │  Muestra audio + specs
│ • plot_distribution()   │  Muestra distribuciones
└─────────────────────────┘
```

---

## 🎯 Módulos y Sus Responsabilidades

### 📦 `preprocessing.py`
**Responsabilidad:** Preprocesamiento de señales de audio

**Input:** Archivo de audio (.egg)
**Output:** Espectrogramas Mel (65×41) + segmentos

**Funciones principales:**
- `load_audio_file()` → Carga audio
- `segment_audio()` → Segmenta en ventanas
- `create_mel_spectrogram()` → Crea espectrograma
- `normalize_spectrogram()` → Normaliza
- `preprocess_audio_paper()` → Pipeline completo

---

### 🎨 `augmentation.py`
**Responsabilidad:** Data augmentation para training

**Input:** Audio o espectrograma
**Output:** Versión aumentada

**Técnicas:**
- **Audio:** time_stretch, pitch_shift, add_noise
- **Spectrogram:** spec_augment, mixup, random_erasing

---

### 🏗️ `dataset.py`
**Responsabilidad:** Crear datasets PyTorch

**Input:** Lista de archivos de audio
**Output:** Tensores PyTorch + PyTorch Dataset

**Componentes:**
- `SampleMeta` → Metadatos estructurados
- `process_dataset()` → Procesar múltiples archivos
- `to_pytorch_tensors()` → Convertir a tensores
- `VowelSegmentsDataset` → PyTorch Dataset
- `build_full_pipeline()` → Orquestador maestro

---

### 🛠️ `utils.py`
**Responsabilidad:** Utilidades transversales

**Funciones:**
- Detección de entorno (Colab/Local)
- Paths automáticos
- Listado de archivos
- Estadísticas de dataset
- Tracking de experimentos

---

### 📊 `visualization.py`
**Responsabilidad:** Visualización de datos y resultados

**Funciones:**
- Visualizar audio + espectrogramas
- Comparar augmentations
- Plots de entrenamiento
- Matrices de confusión
- Distribuciones

---

## 🔍 Tabla de Funciones - Dónde Está Cada Cosa

| Necesitas... | Usa función... | Del módulo... |
|-------------|----------------|---------------|
| Cargar audio | `load_audio_file()` | `preprocessing` |
| Crear espectrograma | `create_mel_spectrogram()` | `preprocessing` |
| Procesar archivos | `build_full_pipeline()` | `dataset` |
| Augmentar datos | `spec_augment()`, `time_stretch()` | `augmentation` |
| Visualizar | `visualize_audio_and_spectrograms()` | `visualization` |
| Path de datos | `get_data_path()` | `utils` |
| Listar archivos | `list_audio_files()` | `utils` |
| Estadísticas | `print_dataset_stats()` | `utils` |
| Setup Colab | `setup_colab_environment()` | `utils` |

---

## 💻 Comandos Útiles

### Ver estructura del proyecto:
```bash
tree -L 2 -I '__pycache__|*.egg|*.nsp'
```

### Contar líneas de código:
```bash
wc -l modules/*.py
```

### Buscar una función:
```bash
grep -r "def nombre_funcion" modules/
```

### Ver imports de un módulo:
```bash
head -20 modules/preprocessing.py
```

---

## 🎓 Guía Rápida de Referencia

### Importar TODO:
```python
from modules import preprocessing, augmentation, dataset, utils, visualization
```

### Constantes del paper:
```python
from modules.preprocessing import SAMPLE_RATE, N_MELS, TARGET_FRAMES
```

### Pipeline completo:
```python
from modules.dataset import build_full_pipeline
results = build_full_pipeline(audio_files)
```

### Visualizar:
```python
from modules.visualization import visualize_audio_and_spectrograms
fig, audios = visualize_audio_and_spectrograms(dataset)
```

---

## 📈 Progreso del Proyecto

```
Fase 1: ✅ Preprocesamiento  (COMPLETADO)
         └─ modules/preprocessing.py

Fase 2: ✅ Dataset Pipeline  (COMPLETADO)
         └─ modules/dataset.py

Fase 3: ✅ Visualización     (COMPLETADO)
         └─ modules/visualization.py

Fase 4: ✅ Augmentation      (COMPLETADO)
         └─ modules/augmentation.py

Fase 5: 🔜 Model Training    (PRÓXIMO)
         └─ modules/models.py
         └─ modules/training.py
         └─ 03_model_training.ipynb

Fase 6: 🔜 Evaluation        (FUTURO)
         └─ modules/evaluation.py
         └─ 04_evaluation.ipynb
```

---

**🎯 Tu código ahora es:** Modular, Profesional, Mantenible, Reutilizable y Escalable! 🚀

