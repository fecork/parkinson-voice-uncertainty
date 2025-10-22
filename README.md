# Detección de Parkinson mediante Análisis de Voz

Sistema de clasificación binaria (Healthy vs Parkinson) usando redes neuronales convolucionales (CNN2D, CNN1D, LSTM) sobre espectrogramas Mel de señales de voz.

**Implementación exacta según Ibarra et al. (2023)**: Preprocesamiento sin augmentation, espectrogramas individuales reutilizables para CNN2D y Time-CNN-LSTM.

---

## Índice

1. [Resumen del Proyecto](#resumen-del-proyecto)
2. [Preprocesamiento (Paper Ibarra 2023)](#preprocesamiento-paper-ibarra-2023)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Flujo de Trabajo](#flujo-de-trabajo)
5. [Notebooks Disponibles](#notebooks-disponibles)
6. [Pipelines Automatizados](#pipelines-automatizados)
7. [Instalación y Configuración](#instalación-y-configuración)
8. [Resultados](#resultados)

---

## Resumen del Proyecto

### Objetivo
Clasificar automáticamente señales de voz para detectar Parkinson usando técnicas de Deep Learning.

### Metodología
Implementación fiel al paper de Ibarra et al. (2023):
- **Preprocesamiento exacto**: Sin augmentation, según especificaciones del paper
- **Modelos**:
  - **CNN2D**: Modelo baseline sin Domain Adaptation
  - **CNN2D_DA**: Modelo con Domain Adaptation y Gradient Reversal Layer (GRL)
  - **CNN1D_DA**: CNN 1D con atención temporal y Domain Adaptation
  - **Time-CNN-BiLSTM-DA**: CNN time-distributed + BiLSTM con Domain Adaptation

### Referencia
Paper: **Ibarra et al. (2023)** - "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"

---

## Preprocesamiento (Paper Ibarra 2023)

Pipeline exacto sin augmentation:

### 1. Resample a 44.1 kHz
Todos los audios se resamplea a frecuencia estándar de 44100 Hz (cuando aplique).

### 2. Normalización por amplitud máxima absoluta
```python
audio = audio / np.max(np.abs(audio))
```

### 3. Segmentación: 400ms ventanas, 50% overlap
- **Duración ventana**: 400 ms
- **Overlap**: 50%
- **Hop**: 200 ms

### 4. Mel Spectrogram: 65 bandas, ventana FFT 40ms, hop 10ms
- **Bandas Mel**: 65
- **Ventana FFT**: 40 ms (para vocales sostenidas)
- **Hop length**: 10 ms
- **Frecuencia máxima**: Nyquist (22.05 kHz)

### 5. Conversión a dB
```python
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
```

### 6. Normalización z-score por espectrograma individual
```python
normalized = (mel_db - mean(mel_db)) / std(mel_db)
```

### 7. Dimensión final: 65×41 píxeles
- **Altura**: 65 bandas Mel (frecuencia)
- **Ancho**: 41 frames temporales

### 8. Sin augmentation (Paper Exacto)
El paper NO menciona data augmentation. Solo el preprocesamiento descrito arriba.

**IMPORTANTE**: Este preprocesamiento (sin augmentation) se usa en:
- `cnn_da_training.ipynb` (CNN2D con Domain Adaptation)
- `cnn1d_da_training.ipynb` (CNN1D)
- `time_cnn_lstm_training.ipynb` (Time-CNN-BiLSTM)

### 9. Data Augmentation (Solo para Baseline)

Para el modelo baseline CNN2D (`cnn_training.ipynb`), se aplica augmentation adicional para mejorar generalización:

**Augmentation de audio**:
- Pitch shifting: ±2 semitonos
- Time stretching: 0.9x - 1.1x
- Noise injection: factor 0.005

**Augmentation de espectrograma**:
- SpecAugment: máscaras de frecuencia (param=10) y tiempo (param=5)

**Factor**: ~5x más datos

**Cache separado**: `cache/healthy_augmented.pkl` y `cache/parkinson_augmented.pkl`

### Uso de espectrogramas

#### Para CNN2D:
- **Input**: Un espectrograma (1, 65, 41) por vez
- **Evaluación**: Probabilidad conjunta agrupando predicciones de todos los espectrogramas del mismo paciente

#### Para Time-CNN-LSTM:
- **Input**: Secuencia de n espectrogramas consecutivos (n, 1, 65, 41) del mismo audio
- **Padding**: Zero-padding cuando hay menos de n frames
- **Masking**: LSTM ignora frames con padding
- **Hiperparámetro n**: Probar con {3, 5, 7, 9} según paper

---

## 📁 Estructura del Proyecto

```
parkinson-voice-uncertainty/
│
├── 📓 NOTEBOOKS (Ejecutar en este orden)
│   ├── 1. data_preprocessing.ipynb    ← Preprocesar datos (UNA VEZ)
│   ├── 2. cnn_training.ipynb          ← CNN2D baseline
│   └── 3. cnn_da_training.ipynb       ← CNN2D_DA con GRL
│
├── 🚀 pipelines/                      ← Scripts automatizados
│   ├── README.md                      ← Documentación
│   ├── train_cnn.py                   ← Pipeline CNN2D + MC Dropout
│   ├── train_cnn_da_kfold.py         ← Pipeline CNN2D_DA + K-fold
│   ├── train_cnn_uncertainty.py      ← Pipeline con incertidumbre
│   └── train_lstm_da_kfold.py        ← Pipeline LSTM-DA + K-fold (NUEVO)
│
├── 📦 modules/                        ← Código compartido (REORGANIZADO v4.0)
│   ├── __init__.py
│   ├── core/                          ← Módulos base
│   │   ├── dataset.py                 ← Gestión de datasets
│   │   ├── sequence_dataset.py        ← Secuencias para LSTM (NUEVO)
│   │   ├── preprocessing.py           ← Preprocesamiento
│   │   ├── utils.py                   ← Utilidades generales
│   │   └── visualization.py           ← Visualizaciones generales
│   ├── data/                          ← Manejo de datos
│   │   ├── augmentation.py            ← Data augmentation
│   │   └── cache_utils.py             ← Gestión de cache
│   └── models/                        ← Modelos de ML
│       ├── common/                    ← Componentes compartidos (NUEVO)
│       │   ├── __init__.py            ← Exports
│       │   └── layers.py              ← FeatureExtractor, GRL, ClassifierHead
│       ├── cnn2d/                     ← CNN 2D
│       │   ├── model.py               ← CNN2D y CNN2D_DA
│       │   ├── training.py            ← Entrenamiento
│       │   ├── inference.py           ← Inferencia MC Dropout
│       │   ├── visualization.py       ← Visualizaciones
│       │   └── utils.py               ← Utilidades CNN2D
│       ├── cnn1d/                     ← CNN 1D
│       │   ├── model.py               ← CNN1D_DA
│       │   ├── training.py            ← Entrenamiento
│       │   └── visualization.py       ← Visualizaciones
│       ├── lstm_da/                   ← Time-CNN-BiLSTM (NUEVO)
│       │   ├── model.py               ← TimeCNNBiLSTM_DA
│       │   ├── training.py            ← Entrenamiento + K-fold
│       │   └── visualization.py       ← Visualizaciones
│       └── uncertainty/               ← Modelos con incertidumbre
│           ├── model.py               ← UncertaintyCNN
│           ├── loss.py                ← Heteroscedastic loss
│           ├── training.py            ← Entrenamiento
│           └── visualization.py       ← Visualizaciones
│
├── 💾 cache/                          ← Datos preprocesados
│   ├── healthy/
│   └── parkinson/
│
├── 📊 data/                           ← Datos raw
│   ├── vowels_healthy/
│   └── vowels_pk/
│
├── 🎯 results/                        ← Resultados de entrenamientos
│   ├── cnn_no_da/                     ← Desde cnn_training.ipynb
│   ├── cnn_da/                        ← Desde cnn_da_training.ipynb
│   └── cnn_da_kfold/                  ← Desde pipelines/
│
└── 📝 Documentación adicional
    └── data_preparation/              ← Guías de preparación de datos
```

---

## Flujo de Trabajo

### Diagrama de Flujo

```
┌──────────────────────────────────────────────┐
│ Paso 1: data_preprocessing.ipynb            │
│ Ejecutar UNA VEZ                             │
│ ⏱️  2-3 minutos (sin augmentation)           │
│ ↓                                            │
│ Genera cache/*_ibarra.pkl (65×41 píxeles)   │
└──────────────────────────────────────────────┘
              ↓
    ┌─────────┴─────────────┐
    ↓                       ↓
┌────────────────┐   ┌─────────────────────┐
│ Paso 2A:       │   │ Paso 2B:            │
│ CNN2D          │   │ CNN2D_DA            │
│ (con augment)  │   │ (sin augment)       │
│                │   │                     │
│ cnn_training   │   │ cnn_da_training     │
│                │   │                     │
│ Genera cache   │   │ Usa cache ibarra    │
│ *_augmented    │   │ (paper exacto)      │
│ (~5x datos)    │   │                     │
│                │   │                     │
│ ⏱️ 15-20 min    │   │ ⏱️ 10-15 min         │
└────────────────┘   └─────────────────────┘
    ↓                       ↓
results/              results/
cnn_no_da/            cnn_da/
(con augment)         (sin augment)
    ↓                       ↓
    └───────────────┬───────┘
                    ↓
            3. Comparar
               Resultados
```

### Estrategia de Augmentation

**SIN Augmentation (Paper Exacto)**:
- `data_preprocessing.ipynb` → `cache/*_ibarra.pkl`
- Usado por: `cnn_da_training.ipynb`, `cnn1d_da_training.ipynb`, `time_cnn_lstm_training.ipynb`
- Objetivo: Seguir paper de Ibarra et al. (2023) exactamente

**CON Augmentation (Mejora Generalización)**:
- `cnn_training.ipynb` genera automáticamente → `cache/*_augmented.pkl`
- Usado solo por: `cnn_training.ipynb`
- Objetivo: Mejorar robustez del modelo baseline con más datos

| Notebook | Augmentation | Cache | Propósito |
|----------|--------------|-------|-----------|
| `cnn_training.ipynb` | ✅ SÍ | `*_augmented.pkl` | Baseline robusto |
| `cnn_da_training.ipynb` | ❌ NO | `*_ibarra.pkl` | Paper exacto |
| Otros notebooks | ❌ NO | `*_ibarra.pkl` | Paper exacto |

### Quick Start

```bash
# Primera vez (validar preprocesamiento):
python test/test_ibarra_preprocessing.py  # Validar que cumple paper

# Generar datos preprocesados:
jupyter notebook data_preprocessing.ipynb  # 1. Generar cache (~2-3 min)

# Entrenar modelos:
jupyter notebook cnn_training.ipynb        # 2A. Baseline (~10-15 min)
jupyter notebook cnn_da_training.ipynb     # 2B. Domain Adapt (~15-20 min)

# Pipelines automatizados:
python pipelines/train_cnn.py --lr 0.001
python pipelines/train_cnn_da_kfold.py --n_folds 10
python pipelines/train_lstm_da_kfold.py --n_frames 7 --lstm_units 64
```

### Validación del Preprocesamiento

Ejecutar pruebas unitarias para verificar cumplimiento del paper:

```bash
python test/test_ibarra_preprocessing.py
```

Las pruebas validan:
- ✅ Constantes (SAMPLE_RATE=44100, N_MELS=65, etc.)
- ✅ Normalización por max-abs
- ✅ Segmentación 400ms con 50% overlap
- ✅ Dimensiones finales 65×41
- ✅ Normalización z-score
- ✅ Sin augmentation (reproducibilidad)

---

## 📓 Notebooks Disponibles

### 1️⃣ `data_preprocessing.ipynb`

**Propósito**: Generar cache de espectrogramas preprocesados según Ibarra et al. (2023)

**Ejecutar**: UNA VEZ (o cuando cambies parámetros de preprocesamiento)

**Contenido**:
- Visualización de audio raw
- Preprocesamiento exacto según paper (sin augmentation):
  - Resample 44.1 kHz + normalización max-abs
  - Segmentación 400ms con 50% overlap
  - Mel spectrograms: 65 bandas, FFT 40ms, hop 10ms
  - Conversión a dB + z-score individual
- Generación y guardado de cache
- Espectrogramas individuales (65×41) reutilizables para CNN2D y Time-CNN-LSTM

**Output**:
```
cache/
├── healthy_ibarra.pkl     (~50-80 espectrogramas)
└── parkinson_ibarra.pkl   (~50-80 espectrogramas)
```

**Tiempo**: ~2-3 minutos (sin augmentation, más rápido)

---

### 2️⃣ `cnn_training.ipynb`

**Propósito**: Entrenar modelo CNN2D baseline CON augmentation

**Prerequisito**: Tener archivos de audio en `data/`

**Contenido**:
- Carga/genera espectrogramas CON augmentation:
  - Pitch shifting
  - Time stretching
  - Noise injection
  - SpecAugment
- Factor: ~5x más datos (mejora generalización)
- Split train/val/test (70/15/15)
- Modelo CNN2D con backbone Ibarra (sin DA)
- Input: un espectrograma (1, 65, 41) por vez
- Entrenamiento con Adam + early stopping
- Evaluación y visualización

**Nota**: Este es el ÚNICO notebook que usa augmentation. El objetivo es mejorar la generalización del modelo baseline.

**Output**:
```
results/cnn_no_da/
├── best_model.pth
├── test_metrics.json
├── training_progress.png
└── confusion_matrix_test.png
```

**Tiempo**: ~10-15 minutos

---

### 3️⃣ `cnn_da_training.ipynb`

**Propósito**: Entrenar modelo CNN2D_DA con Domain Adaptation (según paper exacto)

**Prerequisito**: Cache generado (ejecutar `data_preprocessing.ipynb` primero)

**Contenido**:
- Carga cache de espectrogramas SIN augmentation (paper exacto)
- Split train/val/test (70/15/15)
- Modelo CNN2D_DA (dual-head con GRL)
- Input: un espectrograma (1, 65, 41) por vez
- Entrenamiento multi-task con SGD
- Evaluación PD + Domain

**Nota**: Este notebook sigue el paper de Ibarra et al. (2023) exactamente (sin augmentation).

**Output**:
```
results/cnn_da/
├── best_model_da.pth
├── test_metrics_da.json
├── training_progress_da.png
└── confusion_matrix_test_da.png
```

**Tiempo**: ~15-20 minutos

---

## 🚀 Pipelines Automatizados

### `pipelines/train_cnn.py`

Pipeline completo para entrenar CNN2D con MC Dropout:

```bash
python pipelines/train_cnn.py --epochs 100 --lr 0.001
```

**Características**:
- Entrenamiento automatizado CNN2D
- Implementa MC Dropout para incertidumbre
- Configuración vía argumentos de línea de comandos

---

### `pipelines/train_cnn_da_kfold.py`

Pipeline completo para entrenar CNN2D_DA con validación cruzada:

```bash
python pipelines/train_cnn_da_kfold.py --n_folds 10
```

**Características**:
- Entrenamiento automatizado CNN2D_DA
- K-fold cross-validation (10-fold por defecto)
- Implementación según Ibarra (2023)

---

### `pipelines/train_lstm_da_kfold.py` (NUEVO)

Pipeline completo para entrenar Time-CNN-BiLSTM-DA con validación cruzada:

```bash
python pipelines/train_lstm_da_kfold.py --n_frames 7 --lstm_units 64 --n_folds 10
```

**Características**:
- Entrenamiento automatizado Time-CNN-BiLSTM-DA
- Procesa secuencias de n espectrogramas (n=7, 9)
- BiLSTM con masking para secuencias de longitud variable
- K-fold cross-validation speaker-independent
- Lambda warm-up para GRL (0→1 en 5 épocas)
- SGD con momentum 0.9, LR scheduler StepLR
- Implementación según Ibarra (2023)

**Argumentos principales**:
- `--n_frames`: Número de frames por secuencia (default: 7, paper sugiere: 3, 5, 7, 9)
- `--lstm_units`: Unidades LSTM por dirección (default: 64, paper sugiere: 16, 32, 64)
- `--lambda_warmup`: Épocas de warm-up para lambda GRL (default: 5)

---

## 💻 Instalación y Configuración

### Requisitos del Sistema

- Python 3.8+
- PyTorch 1.8+
- CUDA (opcional, para GPU)

### Instalación

1. **Clonar el repositorio**:
```bash
git clone <repo-url>
cd parkinson-voice-uncertainty
```

2. **Crear entorno virtual**:
```bash
python -m venv parkinson_env
source parkinson_env/bin/activate  # Linux/Mac
# o
parkinson_env\Scripts\activate  # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

### Estructura de Datos

Colocar archivos de audio en:
```
data/
├── vowels_healthy/  ← Archivos .egg de sujetos sanos
└── vowels_pk/       ← Archivos .egg de pacientes Parkinson
```

---

## 📊 Resultados

### Comparación de Modelos

| Modelo | Accuracy | F1-Score | Parámetros | Tipo |
|--------|----------|----------|------------|------|
| CNN2D (baseline) | ~98.8% | ~98.8% | 674,562 | Sin DA |
| CNN2D_DA (con GRL) | TBD | TBD | ~800,000+ | Con DA |
| CNN1D_DA | TBD | TBD | ~350,000+ | Con DA |
| Time-CNN-BiLSTM-DA | TBD | TBD | ~950,000+ | Con DA + Temporal |

### ⚠️ Diferencias Arquitectónicas Importantes

**Los modelos NO son idénticos**. Diferencias clave:

#### 1️⃣ MaxPooling
- **CNN2D**: MaxPool **2×2** (configuración estándar)
- **CNN2D_DA**: MaxPool **3×3** (según paper Ibarra 2023)

#### 2️⃣ Dimensiones de Features
```python
# Después de feature extraction:
CNN2D:    (B, 64, 16, 10) → Flatten → (B, 10,240)
CNN2D_DA: (B, 64, 17, 11) → Flatten → (B, 11,968)
```

#### 3️⃣ Estructura
- **CNN2D**: Single-head (solo clasificación PD)
- **CNN2D_DA**: Dual-head (clasificación PD + Domain con GRL)

#### 4️⃣ Entrenamiento
- **CNN2D**: Adam (LR=0.001), Loss simple
- **CNN2D_DA**: SGD (LR=0.1), Loss multi-task

### Tabla Comparativa Detallada

| Característica | CNN2D | CNN2D_DA |
|----------------|-------|----------|
| **MaxPool** | 2×2 | 3×3 |
| **Feature Dim** | 10,240 | 11,968 |
| **Heads** | 1 (PD) | 2 (PD + Domain) |
| **GRL** | ❌ No | ✅ Sí |
| **Parámetros** | 674,562 | ~800,000+ |
| **Loss** | CrossEntropy | Multi-task |
| **Optimizer** | Adam | SGD |
| **LR** | 0.001 | 0.1 |
| **Uso** | Baseline | Domain Adapt |

### Características de los Modelos

**⚡ Ambos modelos comparten el MISMO Feature Extractor para comparación justa:**
- 2 bloques Conv2D → BatchNorm → ReLU → MaxPool(3×3) → Dropout

#### CNN2D (Baseline)
- **Arquitectura**: Single-head CNN (solo cabeza PD)
- **Feature Extractor**: Idéntico a CNN2D_DA (arquitectura Ibarra 2023)
- **Entrenamiento**: Adam optimizer
- **Output**: Clasificación Healthy/Parkinson
- **Ventaja**: Simplicidad, sin Domain Adaptation

#### CNN2D_DA (Domain Adaptation)
- **Arquitectura**: Dual-head CNN con GRL
- **Feature Extractor**: Compartido con CNN2D (arquitectura Ibarra 2023)
- **Entrenamiento**: SGD optimizer (según paper)
- **Output**: Clasificación PD + Domain
- **Ventaja**: Robustez ante diferentes dominios
- **Paper**: Implementación fiel a Ibarra et al. (2023)

**🔄 Ventaja del diseño modular:**
- Domain Adaptation es un módulo que se puede agregar/quitar
- Comparación justa: mismo backbone, diferente cabeza
- Sin duplicación de código (FeatureExtractor compartido)

---

## 🎓 Conceptos Clave

| Término | Significado |
|---------|-------------|
| **Pipeline** | Flujo completo automatizado end-to-end |
| **Module** | Código reutilizable (librería) |
| **Notebook** | Experimento interactivo Jupyter |
| **Cache** | Datos preprocesados guardados en disco |
| **DA** | Domain Adaptation (adaptación de dominio) |
| **GRL** | Gradient Reversal Layer |
| **MC Dropout** | Monte Carlo Dropout (cuantificación de incertidumbre) |
| **K-fold** | Validación cruzada en K particiones |
| **BiLSTM** | Bidirectional Long Short-Term Memory |
| **Time-distributed** | Aplicar mismas capas a cada frame de secuencia |
| **Masking** | Ignorar frames de padding en cálculos |
| **Lambda warm-up** | Incremento gradual de lambda GRL durante entrenamiento |

---

## 📝 Guía de Uso Paso a Paso

### Primera Ejecución (Setup Completo)

**Día 1: Preparación y Entrenamiento (Total: ~35-45 min)**

1. **Generar Cache** (~7-10 min)
```bash
jupyter notebook data_preprocessing.ipynb
# Ejecutar todas las celdas (Cell → Run All)
```
✅ Resultado: Cache en `cache/healthy/` y `cache/parkinson/`

2. **Entrenar Baseline** (~10-15 min)
```bash
jupyter notebook cnn_training.ipynb
# Ejecutar todas las celdas
```
✅ Resultado: Modelo en `results/cnn_no_da/`

3. **Entrenar con DA** (~15-20 min)
```bash
jupyter notebook cnn_da_training.ipynb
# Ejecutar todas las celdas
```
✅ Resultado: Modelo en `results/cnn_da/`

---

### Experimentación (Con Cache Existente)

**Solo Modificar y Entrenar (~10-15 min por experimento)**

```bash
# Abrir notebook
jupyter notebook cnn_training.ipynb

# Modificar hiperparámetros en la celda correspondiente:
N_EPOCHS = 150
LEARNING_RATE = 5e-4

# Ejecutar todas las celdas
```

---

## 🔧 Configuración de Parámetros

### Preprocesamiento (en `modules/core/preprocessing.py`)

```python
SAMPLE_RATE = 44100      # Hz (actualizado desde 16k)
WINDOW_MS = 400          # ms
OVERLAP = 0.5            # 50%
N_MELS = 65              # Bandas Mel
TARGET_FRAMES = 41       # Frames por espectrograma
```

### Secuencias LSTM (en `modules/core/sequence_dataset.py`)

```python
N_FRAMES = 7             # Frames por secuencia
MIN_FRAMES = 3           # Mínimo de frames para crear secuencia
NORMALIZE = True         # Normalizar POR SECUENCIA (no por frame)
```

### Data Augmentation

```python
AUGMENTATION_TYPES = [
    "original",
    "pitch_shift",
    "time_stretch", 
    "noise"
]
NUM_SPEC_AUGMENT_VERSIONS = 2
```

### Entrenamiento CNN2D

```python
N_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
```

### Entrenamiento CNN2D_DA

```python
N_EPOCHS = 100
LEARNING_RATE = 0.1      # SGD según Ibarra
ALPHA = 1.0              # Peso de loss_domain
LAMBDA_CONSTANT = 1.0    # Lambda para GRL
```

### Entrenamiento Time-CNN-BiLSTM-DA

```python
N_EPOCHS = 100
N_FRAMES = 7             # Secuencia de frames (paper: 3, 5, 7, 9)
LSTM_UNITS = 64          # Unidades por dirección (paper: 16, 32, 64)
LEARNING_RATE = 0.1      # SGD con momentum 0.9
ALPHA = 1.0              # Peso de loss_domain
LAMBDA_WARMUP = 5        # Épocas para warm-up de GRL (0→1)
BATCH_SIZE = 32
```

---

## 🔍 Detalles Técnicos

### Arquitectura CNN2D (sin DA)

```
Input: (B, 1, 65, 41)
↓
[Feature Extractor - Ibarra 2023]
Block1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
Block2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
↓
[PD Head]
Flatten → FC(64) → ReLU → Dropout → FC(2)
↓
Output: Softmax (Healthy/Parkinson)
```

**Nota**: Usa el mismo FeatureExtractor que CNN2D_DA para comparación justa.

---

### Arquitectura CNN2D_DA (con DA)

```
Input: (B, 1, 65, 41)
↓
[Feature Extractor - Ibarra 2023] (COMPARTIDO)
Block1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
Block2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
↓
├─────────────────────────┬─────────────────────────┐
│ [PD Head]               │ [Domain Head]           │
│ Flatten → FC(64) → FC(2)│ GRL → FC(64) → FC(n_dom)│
│ ↓                       │ ↓                       │
│ Healthy/Parkinson       │ Domain ID               │
└─────────────────────────┴─────────────────────────┘
```

**Diseño Modular**: El DA es un módulo que se puede agregar/quitar sin duplicar código.

---

### Arquitectura Time-CNN-BiLSTM-DA (NUEVO)

```
Input: (B, T, 1, 65, 41)  donde T = n_frames (7, 9)
↓
[Time-Distributed Feature Extractor] (REUTILIZA FeatureExtractor de CNN2D)
Para cada frame t en T:
  Block1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
  Block2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
  Projection: Flatten → FC(128) → ReLU → Dropout
↓
Output: (B, T, 128) embeddings

[BiLSTM Temporal con Masking]
BiLSTM(128 → 64 bidirectional) con pack_padded_sequence
↓
Output: (B, T, 128) LSTM hidden states

[Global Pooling Temporal]
Mean pooling considerando solo frames válidos (no padding)
↓
Output: (B, 128) embedding global

├─────────────────────────┬─────────────────────────┐
│ [PD Head]               │ [Domain Head]           │
│ FC(64) → ReLU → FC(2)   │ GRL → FC(64) → FC(4)    │
│ ↓                       │ ↓                       │
│ Healthy/Parkinson       │ Domain ID (4 corpus)    │
└─────────────────────────┴─────────────────────────┘
```

**Ventajas del modelo**:
- ✅ **No requiere post-proceso por paciente**: BiLSTM procesa toda la secuencia
- ✅ **Masking automático**: Ignora frames de padding
- ✅ **Reutiliza código**: FeatureExtractor compartido con CNN2D
- ✅ **Modelado temporal**: BiLSTM captura dependencias entre frames
- ✅ **Lambda warm-up**: GRL aumenta gradualmente (0→1 en 5 épocas)

**Diferencias con CNN2D/CNN1D**:
- CNN2D/CNN1D: Procesan espectrogramas individuales → post-proceso por paciente
- LSTM-DA: Procesa secuencias completas → predicción directa por secuencia

---

## 💡 Ventajas de la Organización Actual

### ✅ Versión 4.0 - Código Compartido Centralizado (NUEVO)

**Mejoras principales:**
- 🎯 **modules/models/common/**: Componentes compartidos entre modelos
  - `FeatureExtractor`: CNN 2D usado por CNN2D y LSTM-DA
  - `GradientReversalLayer (GRL)`: Usado por CNN2D_DA, CNN1D_DA, LSTM-DA
  - `ClassifierHead`: Cabeza de clasificación reutilizable
- ✅ **Sin duplicación**: Un solo lugar para código compartido
- 🔄 **Fácil mantenimiento**: Cambios en un lugar afectan todos los modelos
- 📝 **Imports claros**: `from modules.models.common.layers import FeatureExtractor`

### ✅ Versión 3.0 - Nueva Estructura Modular

**Cambios principales:**
- 📦 **Módulos reorganizados por funcionalidad**:
  - `core/`: Módulos base compartidos (dataset, preprocessing, utils)
  - `data/`: Manejo de datos (augmentation, cache)
  - `models/`: Modelos organizados por tipo (cnn2d, cnn1d, lstm_da, uncertainty)
  - `models/common/`: Componentes compartidos entre modelos
- 🔄 **CNN renombrado a CNN2D** para claridad
- 🎯 **Agrupación lógica**: Cada carpeta agrupa funcionalidad relacionada
- 📝 **Imports simplificados**: `from modules.models.cnn2d import ...`

### ✅ Antes (v2.0 - Problemas)
- ❌ Archivos sueltos en `/modules`
- ❌ `cnn_*.py` sin claridad si es 2D o 1D
- ❌ Módulos de uncertainty, cnn1d y cnn2d mezclados
- ❌ Difícil encontrar qué archivo modificar

### ✅ Ahora (v3.0 - Soluciones)
- ✅ Estructura jerárquica por funcionalidad
- ✅ CNN2D claramente separado de CNN1D
- ✅ Cada modelo tiene su carpeta con todo su código
- ✅ Fácil navegar y mantener
- ✅ Imports más descriptivos y organizados

---

## 🆘 Troubleshooting

### Error: "Cache not found"
**Solución**: Ejecutar `data_preprocessing.ipynb` primero

### Error: "ImportError"
**Solución**: Verificar que estás en la raíz del proyecto

### Error: "Out of memory"
**Solución**: Reducir `BATCH_SIZE` en notebook de entrenamiento

### Cache desactualizado
**Solución**:
```python
# En data_preprocessing.ipynb, modificar:
FORCE_REGENERATE = True  # ← Regenera cache
```

---

## 📈 Comparación: Notebooks vs Pipelines

| Característica | Notebooks | Pipelines |
|---------------|-----------|-----------|
| **Interfaz** | Jupyter (interactivo) | CLI (automatizado) |
| **Uso** | Exploración, debugging | Producción, batch |
| **Supervisión** | Paso a paso | Desatendido |
| **Visualización** | Inline | Archivos PNG |
| **Configuración** | En celdas | Argumentos CLI |

---

## 🎯 Casos de Uso

### 📊 Exploración y Desarrollo
**→ Usar Notebooks**
- Ver resultados paso a paso
- Modificar hiperparámetros fácilmente
- Visualizaciones inline
- Ideal para entender el proceso

### 🚀 Producción y Batch
**→ Usar Pipelines**
- Ejecutar múltiples experimentos
- Automatizar entrenamiento
- Configuración vía CLI
- Ideal para validación cruzada

---

## 📚 Información Técnica Adicional

### Cache de Datos

**Ubicación**: `cache/healthy/` y `cache/parkinson/`

**Contenido**:
- Espectrogramas Mel augmentados
- ~1553 muestras Healthy
- ~1219 muestras Parkinson
- Total: ~2772 espectrogramas

**Ventaja**: 
- Carga instantánea (~5 segundos)
- Ahorro de ~6 minutos por experimento
- Mismos datos para todos los modelos

### Resultados Guardados

Cada entrenamiento guarda:
- ✓ Modelo entrenado (`.pth`)
- ✓ Métricas de test (`.json`)
- ✓ Gráficas de progreso (`.png`)
- ✓ Matriz de confusión (`.png`)

---

## 🔄 Variables Importantes

### Después de `data_preprocessing.ipynb`:
```python
X_healthy     # Tensor (1553, 1, 65, 41)
X_parkinson   # Tensor (1219, 1, 65, 41)
cache/        # Archivos .pkl para reutilizar
```

### Después de `cnn_training.ipynb`:
```python
model         # CNN2D entrenado
history       # Historial de entrenamiento
test_metrics  # Accuracy, F1, Precision, Recall
```

### Después de `cnn_da_training.ipynb`:
```python
model_da         # CNN2D_DA entrenado
history_da       # Historial multi-task
test_metrics_da  # Métricas PD + Domain
```

---

## ✅ Checklist de Inicio

### Primera Vez
- [ ] Instalar dependencias (`pip install -r requirements.txt`)
- [ ] Colocar datos en `data/vowels_healthy/` y `data/vowels_pk/`
- [ ] Ejecutar `data_preprocessing.ipynb`
- [ ] Verificar que `cache/` existe
- [ ] Ejecutar `cnn_training.ipynb`
- [ ] Ejecutar `cnn_da_training.ipynb`
- [ ] Comparar resultados en `results/`

### Experimentación
- [ ] Cache ya existe
- [ ] Modificar hiperparámetros según necesidad
- [ ] Ejecutar notebook de entrenamiento
- [ ] Comparar con runs previos

---

## 🧪 Tests de Validación

### Suite de Tests para Secuencias LSTM

**Archivo**: `test/test_lstm_sequences.py`  
**Tests**: 14/14 pasando

**Validaciones implementadas:**
- ✅ Orden temporal (segment_id consecutivos)
- ✅ Correlación entre frames adyacentes (>0.6)
- ✅ Normalización por secuencia (no por frame)
- ✅ Padding correcto (ceros + masking)
- ✅ No mezcla de frames de diferentes audios
- ✅ SpecAugment consistente (cuando aplica)
- ✅ Compatibilidad con modelos LSTM

**Ejecutar tests:**
```bash
python test/test_lstm_sequences.py
# [PASS] TODOS LOS TESTS PASARON (14/14)
```

**Documentación detallada**: Ver `LSTM_SEQUENCE_IMPROVEMENTS.md`

---

## 📖 Referencias

**Paper Principal**:
- Ibarra et al. (2023): "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"

**Papers Relacionados**:
- Park et al. (2019): "SpecAugment: A Simple Data Augmentation Method for ASR"
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning..."

**Técnicas Implementadas**:
- Domain Adaptation con Gradient Reversal Layer (GRL)
- Monte Carlo Dropout para cuantificación de incertidumbre
- Data Augmentation (SpecAugment global para LSTM)
- K-fold Cross-Validation
- Normalización por secuencia para modelos temporales

---

## 🎯 Próximos Pasos (Futuro)

1. **MC Dropout**: Implementar inferencia con MC Dropout en notebooks
2. **Análisis de Incertidumbre**: Cuantificar incertidumbre en predicciones
3. **Comparación Completa**: Notebook dedicado a comparar CNN2D vs CNN2D_DA
4. **Limpieza**: Eliminar notebooks legacy si es necesario

---

## 💬 Soporte

Para preguntas o problemas:
1. Revisar la documentación en `pipelines/README.md`
2. Verificar troubleshooting en esta guía
3. Revisar logs de ejecución

---

## 📄 Licencia

[Especificar licencia del proyecto]

---

**Última actualización**: 2025-10-21

**Autor**: PHD Research Team

**Versión**: 3.0 (Reorganización modular + Pipeline LSTM optimizado)

**Tests**: 14/14 pasando en `test/test_lstm_sequences.py`
