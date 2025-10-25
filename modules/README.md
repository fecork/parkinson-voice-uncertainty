# 📦 Módulos del Proyecto

Esta carpeta contiene todo el código reutilizable del proyecto, organizado en módulos especializados.

## 📋 Módulos Disponibles

### 🎵 `preprocessing.py`
**Preprocesamiento de señales de audio**

**Funciones principales**:
- `load_audio()`: Carga archivos de audio
- `preprocess_audio()`: Resampling y segmentación
- `compute_mel_spectrogram()`: Genera espectrogramas Mel
- `normalize_spectrogram()`: Normalización z-score

**Parámetros configurables**:
```python
SAMPLE_RATE = 16000      # Hz
WINDOW_MS = 100          # ms
OVERLAP = 0.5            # 50%
N_MELS = 65              # Bandas Mel
HOP_MS = 10              # ms
TARGET_FRAMES = 41       # Frames por espectrograma
```

---

### 🎨 `augmentation.py`
**Data augmentation para audio y espectrogramas**

**Técnicas implementadas**:
- `pitch_shift()`: Cambio de tono
- `time_stretch()`: Estiramiento temporal
- `add_noise()`: Inyección de ruido
- `spec_augment()`: SpecAugment (máscaras de frecuencia/tiempo)

**Funciones principales**:
- `preprocess_audio_with_augmentation()`: Aplica augmentation a un archivo
- `create_augmented_dataset()`: Genera dataset completo augmentado

**Configuración**:
```python
AUGMENTATION_TYPES = [
    "original",
    "pitch_shift",
    "time_stretch",
    "noise"
]
NUM_SPEC_AUGMENT_VERSIONS = 2
```

---

### 💾 `dataset.py`
**Gestión de datasets y conversión a tensores**

**Funciones principales**:
- `parse_filename()`: Extrae información del nombre de archivo
- `build_full_pipeline()`: Pipeline completo de preprocesamiento
- `to_pytorch_tensors()`: Convierte dataset a tensores PyTorch

**Clases**:
- Dataset management y conversión de formatos

---

### 💾 `cache_utils.py`
**Gestión de cache de datos preprocesados**

**Funciones principales**:
- `compute_cache_key()`: Calcula hash único para configuración
- `save_to_cache()`: Guarda dataset en cache
- `load_from_cache()`: Carga dataset desde cache

**Ventajas**:
- Ahorra ~6 minutos por ejecución
- Cache invalidado automáticamente si cambian parámetros
- Formato pickle comprimido

---

### 🧠 `cnn_model.py`
**Arquitecturas de redes neuronales**

**Modelos implementados**:

#### CNN2D (Baseline)
```python
class CNN2D(nn.Module):
    # Single-head CNN
    # Input: (B, 1, 65, 41)
    # Output: (B, 2) - [Healthy, Parkinson]
    # Parámetros: 674,562
```

#### CNN2D_DA (Domain Adaptation)
```python
class CNN2D_DA(nn.Module):
    # Dual-head CNN con GRL
    # Input: (B, 1, 65, 41)
    # Output PD: (B, 2) - [Healthy, Parkinson]
    # Output Domain: (B, n_domains)
```

**Componentes**:
- `GradientReversalLayer`: Implementación de GRL

---

### 🏋️ `cnn_training.py`
**Funciones de entrenamiento**

**Funciones principales**:

#### Para CNN2D (sin DA)
- `train_one_epoch()`: Entrena una época
- `evaluate()`: Evalúa el modelo
- `train_model()`: Entrenamiento completo con early stopping
- `detailed_evaluation()`: Evaluación detallada con métricas

#### Para CNN2D_DA (con DA)
- `train_one_epoch_da()`: Entrena una época multi-task
- `evaluate_da()`: Evalúa ambas tareas (PD + Domain)
- `train_model_da()`: Entrenamiento completo con GRL
- `train_model_da_kfold()`: Entrenamiento con K-fold CV

**Características**:
- Early stopping
- Métricas detalladas (Accuracy, F1, Precision, Recall)
- Guardado automático del mejor modelo
- Logging detallado

---

### 🛠️ `cnn_utils.py`
**Utilidades para CNN**

**Funciones principales**:
- `plot_confusion_matrix()`: Grafica matriz de confusión
- `compute_class_weights_auto()`: Calcula pesos de clase
- `print_model_architecture()`: Imprime arquitectura
- `create_dataloaders()`: Crea DataLoaders PyTorch

---

### 📊 `cnn_visualization.py`
**Visualizaciones de entrenamiento**

**Funciones principales**:
- `plot_training_history()`: Gráfica progreso CNN2D
- `plot_da_training_progress()`: Gráfica progreso CNN2D_DA (multi-task)
- Funciones auxiliares de plotting

---

### 🔮 `cnn_inference.py`
**Inferencia con MC Dropout**

**Funciones principales**:
- `mc_dropout_predict()`: Predicción con MC Dropout
- `quantify_uncertainty()`: Cuantifica incertidumbre
- Análisis de incertidumbre epistémica

---

### 📈 `visualization.py`
**Visualizaciones generales**

**Funciones principales**:
- `visualize_audio_and_spectrograms()`: Visualiza audio y espectrogramas
- `plot_spectrogram_comparison()`: Compara espectrogramas
- Utilidades de plotting generales

---

### 🔧 `utils.py`
**Utilidades generales del proyecto**

Funciones auxiliares y utilidades comunes.

---

### 🚀 `notebook_setup.py`
**Configuración automática de entornos para notebooks**

**Funciones principales**:

#### `setup_environment(verbose=True)`
Configuración rápida para notebooks locales:
```python
from modules.core.notebook_setup import setup_environment

# Configurar entorno local automáticamente
setup_environment()
```

#### `setup_colab_git(computer_name, project_dir, branch)`
Configuración completa para Google Colab con Git:
```python
from modules.core.notebook_setup import setup_colab_git

# Configuración por defecto
project_path = setup_colab_git()

# Configuración personalizada
project_path = setup_colab_git(
    computer_name="ZenBook",
    project_dir="parkinson-voice-uncertainty",
    branch="main"
)
```

**Características**:
- Monta Google Drive automáticamente
- Configura repositorio Git y cambia de rama
- Actualiza código desde repositorio remoto
- Instala dependencias del `requirements.txt`
- Activa autoreload para notebooks
- Manejo de errores robusto

**Parámetros**:
- `computer_name`: Nombre del PC en Google Drive
- `project_dir`: Carpeta del repositorio
- `branch`: Rama de Git a usar (main, dev, etc.)

---

## 💡 Cómo Usar los Módulos

### En Notebooks

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Importar módulos
from modules.augmentation import create_augmented_dataset
from modules.cnn_model import CNN2D, CNN2D_DA
from modules.cnn_training import train_model, train_model_da
from modules.cnn_visualization import plot_training_history
```

### En Scripts (pipelines/)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos
from modules.cnn_model import CNN2D
from modules.cnn_training import train_model
```

---

## 🔄 Separación de Responsabilidades

| Módulo | Responsabilidad |
|--------|----------------|
| `preprocessing.py` | Procesar audio → espectrogramas |
| `augmentation.py` | Augmentar datos |
| `dataset.py` | Gestionar datasets |
| `cache_utils.py` | Cache de datos |
| `cnn_model.py` | Arquitecturas de redes |
| `cnn_training.py` | Entrenar modelos |
| `cnn_utils.py` | Utilidades CNN |
| `cnn_visualization.py` | Visualizar entrenamientos |
| `cnn_inference.py` | Inferencia con MC Dropout |
| `visualization.py` | Visualizaciones generales |
| `utils.py` | Utilidades generales |

---

## 🎯 Ventajas de la Organización

### ✅ Reutilización
Todo el código está centralizado y puede ser importado desde:
- Notebooks
- Pipelines
- Tests

### ✅ Mantenimiento
- Un solo lugar para modificar funcionalidad
- No duplicación de código
- Fácil de testear

### ✅ Claridad
- Cada módulo tiene una responsabilidad clara
- Imports explícitos
- Código organizado

---

## 🧪 Ejemplo de Uso Completo

```python
# 1. Importar módulos necesarios
from modules.augmentation import create_augmented_dataset
from modules.dataset import to_pytorch_tensors
from modules.cnn_model import CNN2D
from modules.cnn_training import train_model
from modules.cnn_visualization import plot_training_history

# 2. Cargar datos augmentados
dataset = create_augmented_dataset(
    audio_files,
    augmentation_types=["original", "pitch_shift"],
    use_cache=True
)

# 3. Convertir a tensores
X, y_task, y_domain, meta = to_pytorch_tensors(dataset)

# 4. Crear modelo
model = CNN2D(n_classes=2)

# 5. Entrenar
results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    n_epochs=100
)

# 6. Visualizar
plot_training_history(results["history"])
```

---

## 📝 Buenas Prácticas

### ✅ Hacer
1. Importar siempre desde `modules/`
2. No modificar código en notebooks (modificar aquí)
3. Testear cambios antes de usar en producción

### ❌ Evitar
1. Duplicar código en notebooks
2. Modificar funciones en notebooks temporalmente
3. Hard-coding de valores (usar constantes en módulos)

---

**Última actualización**: 2025-10-17

