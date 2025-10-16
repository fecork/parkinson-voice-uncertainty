# Detección de Parkinson desde Voz con Domain Adaptation

Sistema completo de análisis de voz para detección de Parkinson usando CNN 2D con Domain Adaptation, incertidumbre epistémica (MC Dropout) y explicabilidad (Grad-CAM).

---

## 📋 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Arquitecturas Disponibles](#-arquitecturas-disponibles)
3. [Instalación y Setup](#-instalación-y-setup)
4. [Pipeline Completo](#-pipeline-completo)
5. [Uso Rápido](#-uso-rápido)
6. [Módulos del Sistema](#-módulos-del-sistema)
7. [Documentación Detallada](#-documentación-detallada)

---

## 🎯 Descripción General

### Paper de Referencia
Ibarra et al. (2023) *"Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"*

### Características Principales

- ✅ **Preprocesamiento exacto del paper**: Mel-spectrograms 65×41, z-score
- ✅ **Data Augmentation**: Pitch shift, time stretch, noise, SpecAugment
- ✅ **2 Arquitecturas CNN**:
  - Baseline: CNN2D simple
  - **Domain Adaptation**: Dual-head con GRL (Gradient Reversal Layer)
- ✅ **Incertidumbre Epistémica**: MC Dropout (30 muestras)
- ✅ **Explicabilidad**: Grad-CAM para mapas de atención
- ✅ **Agregación Multinivel**: Segmento → Archivo → Paciente
- ✅ **Split Speaker-Independent**: Evita data leakage
- ✅ **Código Modular**: PEP 8, type hints, documentación completa

---

## 🏗️ Arquitecturas Disponibles

### 1. CNN2D Baseline

```
Input (B, 1, 65, 41)
    ↓
Conv2D(32) → BN → ReLU → MaxPool(2×2) → Dropout(0.3)
    ↓
Conv2D(64) → BN → ReLU → MaxPool(2×2) → Dropout(0.3)
    ↓
Flatten → FC(64) → ReLU → Dropout(0.5) → FC(2)
    ↓
Output: HC/PD (2 clases)
```

**Uso**: Clasificación binaria simple  
**Parámetros**: ~674K

### 2. CNN2D con Domain Adaptation (Recomendado)

```
Input (B, 1, 65, 41)
    ↓
┌────────────────────────────┐
│  Feature Extractor (Shared) │
│  Conv2D(32) → MaxPool(3×3)  │
│  Conv2D(64) → MaxPool(3×3)  │
└────────────────────────────┘
    ↓
Features (B, 64, 17, 11)
    ↓
    ├─────────────┬──────────────┐
    ↓             ↓              ↓
PD Head      GRL (λ)      Domain Head
Linear(2)       ↓          Linear(N)
    ↓       Inverted           ↓
HC/PD    Features         Domains
```

**Ventajas**:
- Features invariantes al dominio (corpus/micrófono)
- Mejor generalización cross-corpus
- Menos overfitting

**Parámetros**: ~1.55M

---

## 🚀 Instalación y Setup

### Requisitos
```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install librosa soundfile scikit-learn
pip install matplotlib seaborn pandas numpy tqdm
```

### Estructura del Proyecto
```
parkinson-voice-uncertainty/
├── data/
│   ├── vowels_healthy/    # Datos HC (13 archivos)
│   └── vowels_pk/         # Datos PD (13 archivos)
├── modules/
│   ├── preprocessing.py   # Preprocesamiento según paper
│   ├── augmentation.py    # Data augmentation
│   ├── dataset.py         # PyTorch datasets
│   ├── cnn_model.py       # CNN Baseline + DA
│   ├── cnn_training.py    # Pipeline de entrenamiento
│   ├── cnn_utils.py       # Utilidades
│   └── cnn_visualization.py  # Visualizaciones
├── parkinson_voice_analysis.ipynb  # Notebook principal
├── train_cnn.py           # Script de entrenamiento
└── README.md             # Este archivo
```

---

## 📊 Pipeline Completo

### 1. Preprocesamiento
```python
# Configuración según paper Ibarra
SAMPLE_RATE = 44100 Hz
WINDOW_MS = 400 ms (50% overlap)
N_MELS = 65 bandas
HOP_MS = 10 ms
FFT_WINDOW = 40 ms
OUTPUT_SHAPE = 65 × 41
NORMALIZATION = z-score
```

### 2. Data Augmentation

**Nivel Audio** (antes de segmentar):
- Pitch shift: ±1, ±2 semitonos
- Time stretch: 0.9x, 1.1x
- Noise: SNR ≈30 dB

**Nivel Espectrograma** (después de Mel):
- SpecAugment: frequency + time masking

**Factor total**: ~10x multiplicación de datos

### 3. Entrenamiento

**Split**: 70% train / 15% val / 15% test (speaker-independent)

**Hiperparámetros recomendados**:
```python
batch_size = 32
learning_rate = 0.01
optimizer = SGD(momentum=0.9, weight_decay=1e-4)
epochs = 100
early_stopping = 15 épocas
```

**Domain Adaptation**:
```python
alpha = 1.0  # Peso de loss_domain
lambda_scheduler = progresivo (0 → 1)
```

### 4. Evaluación

**Métricas**:
- Accuracy, Precision, Recall, F1
- Confusion Matrix
- MC Dropout: Incertidumbre epistémica
- Grad-CAM: Explicabilidad visual

**Agregación**:
- Segmento (65×41 espectrograma)
- Archivo (promedio probabilidades)
- Paciente (promedio por subject_id)

---

## 🎮 Uso Rápido

### Opción 1: Notebook (Recomendado)

```python
# Abrir parkinson_voice_analysis.ipynb
# Ejecutar celdas secuencialmente:

# 1. Setup y carga de datos
# 2. Preprocesamiento
# 3. Data Augmentation
# 4. Entrenamiento CNN-DA
# 5. Evaluación y visualizaciones
```

### Opción 2: Script de Línea de Comandos

```bash
# Entrenamiento básico con DA
python train_cnn.py \
  --hc_dir data/vowels_healthy \
  --pd_dir data/vowels_pk \
  --architecture da \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.01 \
  --mc_samples 30

# Ver todas las opciones
python train_cnn.py --help
```

### Opción 3: API Programática

```python
from modules.cnn_model import CNN2D_DA
from modules.cnn_training import train_model_da, compute_lambda_schedule
import torch

# 1. Crear modelo
model_da = CNN2D_DA(n_domains=26).to(device)

# 2. Configurar optimizador
optimizer = torch.optim.SGD(model_da.parameters(), lr=0.01)
criterion_pd = torch.nn.CrossEntropyLoss()
criterion_domain = torch.nn.CrossEntropyLoss()

# 3. Entrenar
results = train_model_da(
    model_da, train_loader, val_loader,
    optimizer, criterion_pd, criterion_domain,
    device, n_epochs=100, alpha=1.0,
    lambda_scheduler=lambda e: compute_lambda_schedule(e, 100)
)

# 4. Visualizar
from modules.cnn_visualization import plot_da_training_progress
plot_da_training_progress(results['history'])
```

---

## 🧩 Módulos del Sistema

### Core

| Módulo | Descripción | Funciones Principales |
|--------|-------------|----------------------|
| `preprocessing.py` | Preprocesamiento según paper | `preprocess_audio_paper()` |
| `augmentation.py` | Data augmentation | `preprocess_audio_with_augmentation()`, `create_augmented_dataset()` |
| `dataset.py` | PyTorch datasets | `VowelSegmentsDataset`, `build_full_pipeline()` |

### CNN

| Módulo | Descripción | Clases/Funciones |
|--------|-------------|------------------|
| `cnn_model.py` | Arquitecturas | `CNN2D`, `CNN2D_DA`, `GradientReversalLayer`, `mc_dropout_predict()`, `GradCAM` |
| `cnn_training.py` | Entrenamiento | `train_model()`, `train_model_da()`, `evaluate_da()`, `compute_lambda_schedule()` |
| `cnn_utils.py` | Utilidades | `plot_lambda_schedule()`, `print_model_architecture()`, `calculate_class_weights()` |
| `cnn_visualization.py` | Visualizaciones | `plot_da_training_progress()`, `visualize_gradcam()`, `create_da_summary_report()` |

### Utilidades

| Script | Propósito |
|--------|-----------|
| `sample_healthy_data.py` | Muestreo balanceado de datos HC |
| `verify_sampling.py` | Verificación de muestreo |
| `train_cnn.py` | Script de entrenamiento CLI |

---

## 📚 Documentación Detallada

Para información detallada sobre cada componente:

### 🔬 Preprocesamiento y Augmentation
- Ver celdas 1-13 en `parkinson_voice_analysis.ipynb`
- `modules/preprocessing.py` - Configuración exacta del paper
- `modules/augmentation.py` - Todas las técnicas de augmentation

### 🧠 CNN Baseline
- `modules/cnn_model.py` - Clase `CNN2D`
- MC Dropout, Grad-CAM, agregación multinivel
- Ver sección "CNN2D Baseline" arriba

### 🌐 Domain Adaptation (Recomendado)
- `modules/cnn_model.py` - Clases `CNN2D_DA`, `GradientReversalLayer`
- `modules/cnn_training.py` - `train_model_da()`, `compute_lambda_schedule()`
- Ver sección "CNN2D con Domain Adaptation" arriba
- Arquitectura dual-head, lambda scheduling, multi-task learning

### 📊 Muestreo de Datos
- Script: `data_preparation/sample_healthy_data.py --help`
- Ver sección "Quick Start" para uso básico

---

## 📈 Resultados Esperados

### Output del Entrenamiento

```
results/cnn_da/
├── best_model_da.pth              # Mejor checkpoint
├── training_history.json          # Métricas por época
├── config.json                    # Configuración
├── training_progress.png          # 4 gráficas: losses, accuracy, F1, lambda
├── confusion_matrix_test.png      # Matriz de confusión
└── metrics_summary.png            # Resumen visual
```

### Métricas Típicas

**Nivel Segmento**:
- Accuracy: 75-85%
- F1 Score: 0.70-0.80

**Nivel Archivo** (agregado):
- Accuracy: 80-90%
- F1 Score: 0.75-0.85

**Nivel Paciente** (agregado):
- Accuracy: 85-95%
- F1 Score: 0.80-0.90

---

## 🔧 Configuración Avanzada

### Hiperparámetros Clave

```python
# CNN Baseline
p_drop_conv = 0.3
p_drop_fc = 0.5

# Domain Adaptation
alpha = 1.0        # Peso loss_domain
gamma = 10.0       # Scheduler GRL
power = 0.75       # Exponente scheduler

# SpecAugment
freq_mask_param = 10
time_mask_param = 5
num_freq_masks = 2
num_time_masks = 2

# MC Dropout
n_samples = 30
```

### Lambda Scheduling (DA)

El factor λ de GRL aumenta progresivamente:
```
λ(epoch) = 2/(1 + exp(-γ·p))^power - 1
donde p = epoch / max_epoch
```

- Época 0: λ ≈ 0 (solo aprende tarea PD)
- Época 50: λ ≈ 0.7
- Época 100: λ ≈ 1.0 (máxima inversión)

---

## 🐛 Troubleshooting

### Error: Dimensiones incorrectas
```python
# Verificar shape de espectrogramas
print(X_combined.shape)  # Debe ser (N, 1, 65, 41)

# Convertir labels a Long
y_task = y_task.long()
y_domain = y_domain.long()
```

### VRAM insuficiente
```bash
# Reducir batch size
python train_cnn.py --batch_size 16  # o 8
```

### Loss Domain no disminuye
```python
# Verificar alpha > 0
# Aumentar gamma para activar GRL más rápido
lambda_scheduler = lambda e: compute_lambda_schedule(e, 100, gamma=15.0)
```

### Overfitting
```python
# Aumentar dropout
model_da = CNN2D_DA(p_drop_conv=0.4, p_drop_fc=0.6)

# Más augmentation
# Reducir learning rate
# Aumentar weight_decay
```

---

## 📖 Referencias

1. **Ibarra et al. (2023)**: Domain Adaptation para Parkinson
2. **Ganin & Lempitsky (2015)**: Gradient Reversal Layer
3. **Park et al. (2019)**: SpecAugment
4. **Gal & Ghahramani (2016)**: MC Dropout
5. **Selvaraju et al. (2017)**: Grad-CAM

---

## 👥 Créditos

**Autor**: PhD Research Team  
**Versión**: 2.0  
**Fecha**: Octubre 2025

Implementación basada en:
- Paper Ibarra et al. (2023)
- Arquitectura MARTA
- Buenas prácticas: PEP 8, modularidad, documentación

---

## 📝 Licencia

Este código es parte de investigación doctoral en detección de Parkinson mediante análisis de voz.






