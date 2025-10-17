# 🎵 Detección de Parkinson mediante Análisis de Voz

Sistema de clasificación binaria (Healthy vs Parkinson) usando redes neuronales convolucionales 2D sobre espectrogramas Mel de señales de voz.

---

## 📋 Índice

1. [Resumen del Proyecto](#resumen-del-proyecto)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Flujo de Trabajo](#flujo-de-trabajo)
4. [Notebooks Disponibles](#notebooks-disponibles)
5. [Pipelines Automatizados](#pipelines-automatizados)
6. [Instalación y Configuración](#instalación-y-configuración)
7. [Resultados](#resultados)

---

## 🎯 Resumen del Proyecto

### Objetivo
Clasificar automáticamente señales de voz para detectar Parkinson usando técnicas de Deep Learning.

### Metodología
- **Preprocesamiento**: Resampling, segmentación, Mel spectrograms
- **Data Augmentation**: Pitch shift, time stretch, noise, SpecAugment
- **Modelos**:
  - **CNN2D**: Modelo baseline sin Domain Adaptation
  - **CNN2D_DA**: Modelo con Domain Adaptation y Gradient Reversal Layer (GRL)

### Implementación
Basado en el paper: **Ibarra et al. (2023)** - "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"

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
│   └── train_cnn_da_kfold.py         ← Pipeline CNN2D_DA + K-fold
│
├── 📦 modules/                        ← Código compartido
│   ├── __init__.py
│   ├── augmentation.py                ← Data augmentation
│   ├── cache_utils.py                 ← Gestión de cache
│   ├── cnn_inference.py               ← Inferencia con MC Dropout
│   ├── cnn_model.py                   ← CNN2D y CNN2D_DA
│   ├── cnn_training.py                ← Funciones de entrenamiento
│   ├── cnn_utils.py                   ← Utilidades
│   ├── cnn_visualization.py           ← Visualizaciones
│   ├── dataset.py                     ← Gestión de datasets
│   ├── preprocessing.py               ← Preprocesamiento
│   ├── utils.py                       ← Utilidades generales
│   └── visualization.py               ← Visualizaciones generales
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

## 🔄 Flujo de Trabajo

### 📊 Diagrama de Flujo

```
┌─────────────────────────────────────┐
│ Paso 1: data_preprocessing.ipynb   │
│ Ejecutar UNA VEZ                    │
│ ⏱️  7-10 minutos                     │
│ ↓                                   │
│ Genera cache/ con datos augmentados │
└─────────────────────────────────────┘
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
┌─────────────┐   ┌──────────────────┐
│ Paso 2A:    │   │ Paso 2B:         │
│ CNN2D       │   │ CNN2D_DA         │
│ (baseline)  │   │ (domain adapt)   │
│             │   │                  │
│ cnn_        │   │ cnn_da_          │
│ training    │   │ training         │
│ .ipynb      │   │ .ipynb           │
│             │   │                  │
│ ⏱️ 10-15 min │   │ ⏱️ 15-20 min     │
└─────────────┘   └──────────────────┘
    ↓                   ↓
results/          results/
cnn_no_da/        cnn_da/
    ↓                   ↓
    └───────────┬───────┘
                ↓
        3. Comparar
           Resultados
```

### ⚡ Quick Start

```bash
# Primera vez (setup completo):
jupyter notebook data_preprocessing.ipynb  # 1. Generar cache (~7-10 min)
jupyter notebook cnn_training.ipynb        # 2A. Baseline (~10-15 min)
jupyter notebook cnn_da_training.ipynb     # 2B. Domain Adapt (~15-20 min)

# Experimentación (cache ya existe):
jupyter notebook cnn_training.ipynb        # Modificar hiperparámetros y ejecutar

# Producción (automatizado):
python pipelines/train_cnn.py --lr 0.001
python pipelines/train_cnn_da_kfold.py --n_folds 10
```

---

## 📓 Notebooks Disponibles

### 1️⃣ `data_preprocessing.ipynb`

**Propósito**: Generar cache de datos preprocesados y augmentados

**Ejecutar**: UNA VEZ (o cuando cambies parámetros de preprocesamiento)

**Contenido**:
- 🔊 Visualización de audio raw
- 🎵 Preprocesamiento (resampling, segmentación, Mel spectrograms)
- 🎨 Data augmentation (pitch shift, time stretch, noise, SpecAugment)
- 💾 Generación de cache

**Output**:
```
cache/
├── healthy/augmented_dataset_*.pkl (~1553 muestras)
└── parkinson/augmented_dataset_*.pkl (~1219 muestras)
```

**Tiempo**: ~7-10 minutos

---

### 2️⃣ `cnn_training.ipynb`

**Propósito**: Entrenar modelo CNN2D baseline sin Domain Adaptation

**Prerequisito**: ⚠️ Cache generado (ejecutar `data_preprocessing.ipynb` primero)

**Contenido**:
- 📁 Carga cache (~5 segundos)
- 📊 Split train/val/test (70/15/15)
- 🏗️ Modelo CNN2D con backbone Ibarra (sin DA)
- 🚀 Entrenamiento con Adam + early stopping
- 📈 Evaluación y visualización

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

**Propósito**: Entrenar modelo CNN2D_DA con Domain Adaptation

**Prerequisito**: ⚠️ Cache generado (ejecutar `data_preprocessing.ipynb` primero)

**Contenido**:
- 📁 Carga cache (~5 segundos)
- 📊 Split train/val/test (70/15/15)
- 🏗️ Modelo CNN2D_DA (dual-head con GRL)
- 🚀 Entrenamiento multi-task con SGD
- 📈 Evaluación PD + Domain

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

| Modelo | Accuracy | F1-Score | Parámetros |
|--------|----------|----------|------------|
| CNN2D (baseline) | ~98.8% | ~98.8% | 674,562 |
| CNN2D_DA (con GRL) | TBD | TBD | ~800,000+ |

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

### Preprocesamiento (en `modules/preprocessing.py`)

```python
SAMPLE_RATE = 16000      # Hz
WINDOW_MS = 100          # ms
OVERLAP = 0.5            # 50%
N_MELS = 65              # Bandas Mel
TARGET_FRAMES = 41       # Frames por espectrograma
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

## 💡 Ventajas de la Organización Actual

### ✅ Antes (Problemas)
- ❌ Todo mezclado en un notebook gigante
- ❌ Reprocesar datos en cada experimento (~6 min cada vez)
- ❌ Scripts duplicando código de notebooks
- ❌ Difícil saber qué ejecutar primero

### ✅ Ahora (Soluciones)
- ✅ Notebooks modulares (una responsabilidad por notebook)
- ✅ Cache reutilizable (ahorro de ~6 min/experimento)
- ✅ Scripts organizados en `pipelines/` (sin duplicación)
- ✅ Flujo claro y documentado

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

## 📖 Referencias

**Paper Principal**:
- Ibarra et al. (2023): "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"

**Técnicas Implementadas**:
- Domain Adaptation con Gradient Reversal Layer (GRL)
- Monte Carlo Dropout para cuantificación de incertidumbre
- Data Augmentation (SpecAugment)
- K-fold Cross-Validation

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

**Última actualización**: 2025-10-17

**Autor**: [Tu nombre/equipo]

**Versión**: 2.0 (Reorganización modular)
