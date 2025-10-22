# Guía Rápida de Uso - Preprocesamiento Dual

## Resumen Ejecutivo

Este proyecto implementa dos pipelines de preprocesamiento:

1. **SIN Augmentation** (Paper Ibarra et al. 2023) → Para CNN_DA, LSTM
2. **CON Augmentation** (Baseline robusto) → Para CNN baseline

## Validación Completa ✅

```bash
$ python test/test_ibarra_preprocessing.py
Tests ejecutados: 19
Exitosos: 19 ✅
Fallidos: 0
```

El preprocesamiento cumple 100% con el paper de Ibarra et al. (2023).

## Flujo de Trabajo

### Paso 1: Generar Datos Base (SIN Augmentation)

```bash
jupyter notebook data_preprocessing.ipynb
```

**Ejecuta**:
- Resample 44.1 kHz + normalización max-abs
- Segmentación 400ms con 50% overlap
- Mel spectrograms: 65 bandas, FFT 40ms, hop 10ms
- Conversión a dB + z-score individual
- **SIN augmentation** (paper exacto)

**Genera**:
- `cache/healthy_ibarra.pkl` (~20-30 espectrogramas)
- `cache/parkinson_ibarra.pkl` (~20-30 espectrogramas)

**Tiempo**: ~2-3 minutos

---

### Paso 2a: Entrenar CNN2D Baseline (CON Augmentation)

```bash
jupyter notebook cnn_training.ipynb
```

**Ejecuta**:
- Carga archivos de audio originales
- Aplica preprocesamiento base + augmentation:
  - Pitch shifting
  - Time stretching
  - Noise injection
  - SpecAugment
- Genera cache automáticamente

**Genera**:
- `cache/healthy/augmented_dataset_*.pkl` (~150-200 espectrogramas, ~5x)
- `cache/parkinson/augmented_dataset_*.pkl` (~120-150 espectrogramas, ~5x)
- Modelo entrenado en `results/cnn_no_da/`

**Tiempo**: ~15-20 minutos (primera vez, incluye generación de augmentation)

---

### Paso 2b: Entrenar CNN2D_DA (SIN Augmentation, Paper Exacto)

```bash
jupyter notebook cnn_da_training.ipynb
```

**Ejecuta**:
- Carga `cache/healthy_ibarra.pkl` y `cache/parkinson_ibarra.pkl`
- Entrena con datos exactos del paper (sin augmentation)
- Domain Adaptation con GRL

**Genera**:
- Modelo entrenado en `results/cnn_da/`

**Tiempo**: ~10-15 minutos

---

## Archivos de Cache

```
cache/
├── healthy_ibarra.pkl        ← Datos paper exacto (sin augmentation)
├── parkinson_ibarra.pkl      ← Datos paper exacto (sin augmentation)
├── healthy/
│   └── augmented_dataset_*.pkl  ← Datos con augmentation (~5x)
└── parkinson/
    └── augmented_dataset_*.pkl  ← Datos con augmentation (~5x)
```

## Tabla de Uso por Notebook

| Notebook | Datos que usa | Augmentation | Cache | Propósito |
|----------|---------------|--------------|-------|-----------|
| `data_preprocessing.ipynb` | Genera | ❌ NO | `*_ibarra.pkl` | Paper exacto |
| `cnn_training.ipynb` | Genera auto | ✅ SÍ | `*/augmented_*.pkl` | Baseline robusto |
| `cnn_da_training.ipynb` | Carga | ❌ NO | `*_ibarra.pkl` | Paper exacto DA |
| `cnn1d_da_training.ipynb` | Carga | ❌ NO | `*_ibarra.pkl` | Paper exacto 1D |
| `time_cnn_lstm_training.ipynb` | Carga | ❌ NO | `*_ibarra.pkl` | Paper exacto LSTM |

## Comandos Rápidos

### Validar Preprocesamiento
```bash
python test/test_ibarra_preprocessing.py
```

### Generar Datos Base (una vez)
```bash
jupyter notebook data_preprocessing.ipynb
# Ejecutar todas las celdas
```

### Entrenar CNN2D con Augmentation
```bash
jupyter notebook cnn_training.ipynb
# Ejecutar todas las celdas
```

### Entrenar CNN2D_DA sin Augmentation (Paper Exacto)
```bash
jupyter notebook cnn_da_training.ipynb
# Ejecutar todas las celdas
```

## Preguntas Frecuentes

### ¿Por qué dos pipelines diferentes?

**Paper Exacto (sin augmentation)**:
- Permite comparación directa con resultados del paper
- Domain Adaptation funciona mejor sin datos artificiales
- Reproducibilidad científica

**Con Augmentation (baseline)**:
- Mejora generalización del modelo simple
- Más datos para entrenar
- Reduce overfitting

### ¿Qué archivo ejecuto primero?

1. `data_preprocessing.ipynb` (una vez) → Genera datos base
2. Luego cualquier notebook de entrenamiento

### ¿Dónde está el augmentation?

- `cnn_training.ipynb` lo genera automáticamente usando `modules/data/augmentation.py`
- Se cachea en `cache/healthy/` y `cache/parkinson/`
- No necesitas ejecutar un notebook separado

### ¿Cómo comparo con/sin augmentation?

- **Con**: `cnn_training.ipynb` → `results/cnn_no_da/`
- **Sin**: `cnn_da_training.ipynb` → `results/cnn_da/`
- Compara métricas en `test_metrics*.json`

## Arquitectura Implementada

### Preprocesamiento Base (Ibarra 2023)
```
Audio (.egg 50kHz)
    ↓
Resample 44.1kHz
    ↓
Norm max-abs
    ↓
Segment 400ms (50% overlap)
    ↓
Mel 65 bandas (FFT 40ms, hop 10ms)
    ↓
dB conversion
    ↓
Z-score normalization
    ↓
Espectrograma 65×41
```

### Augmentation (Solo cnn_training)
```
Espectrograma base 65×41
    ↓
Aplicar (aleatoriamente):
  - Pitch shift desde audio
  - Time stretch desde audio
  - Noise desde audio
  - SpecAugment sobre espectro
    ↓
Dataset ~5x mayor
```

## Validación del Paper

✅ **19/19 tests unitarios PASANDO**

Verifica:
- Todas las constantes del paper
- Normalización correcta
- Segmentación correcta
- Dimensiones correctas
- Z-score correcto
- Sin augmentation en pipeline base
- Propiedades matemáticas

**Conclusión**: El preprocesamiento cumple 100% con Ibarra et al. (2023)

