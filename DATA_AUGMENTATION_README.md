# Data Augmentation - Estrategia de Implementación

## Estructura de Preprocesamiento

### 1. Preprocesamiento Base (Paper Ibarra 2023)
**Archivo**: `data_preprocessing.ipynb`
- **Pipeline**: Exacto según paper (SIN augmentation)
- **Output**: `cache/healthy_ibarra.pkl`, `cache/parkinson_ibarra.pkl`
- **Uso**: CNN_DA, Time-CNN-LSTM (modelos que siguen paper exacto)

### 2. Data Augmentation (Para mejorar generalización)
**Archivo**: `data_augmentation.ipynb`
- **Input**: Carga cache de `data_preprocessing.ipynb`
- **Augmentation**: 
  - Pitch shifting
  - Time stretching  
  - Noise injection
  - SpecAugment (máscaras de frecuencia/tiempo)
- **Output**: `cache/healthy_augmented.pkl`, `cache/parkinson_augmented.pkl`
- **Uso**: CNN_training (modelo baseline con augmentation)

## Notebooks y sus Datos

| Notebook | Datos que usa | Augmentation |
|----------|---------------|--------------|
| `cnn_training.ipynb` | `*_augmented.pkl` | ✅ SÍ |
| `cnn_da_training.ipynb` | `*_ibarra.pkl` | ❌ NO (paper exacto) |
| `cnn1d_da_training.ipynb` | `*_ibarra.pkl` | ❌ NO (paper exacto) |
| `time_cnn_lstm_training.ipynb` | `*_ibarra.pkl` | ❌ NO (paper exacto) |

## Flujo de Trabajo

```
1. data_preprocessing.ipynb
   ↓
   cache/*_ibarra.pkl (preprocesamiento exacto paper)
   ↓
   ├─→ data_augmentation.ipynb → cache/*_augmented.pkl
   │   ↓
   │   └─→ cnn_training.ipynb (usa augmentation)
   │
   └─→ cnn_da_training.ipynb, otros (sin augmentation, paper exacto)
```

## Beneficios

1. **Separación clara**: Preprocesamiento base vs augmentation
2. **Reproducibilidad**: Datos paper siempre disponibles sin augmentation
3. **Flexibilidad**: Fácil comparar con/sin augmentation
4. **Modularidad**: Cada técnica de augmentation es independiente

## Uso

### Generar datos base (una vez):
```bash
jupyter notebook data_preprocessing.ipynb
```

### Generar datos augmentados (una vez):
```bash
jupyter notebook data_augmentation.ipynb
```

### Entrenar modelos:
```bash
# Con augmentation
jupyter notebook cnn_training.ipynb

# Sin augmentation (paper exacto)
jupyter notebook cnn_da_training.ipynb
```

