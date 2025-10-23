# 📓 Notebooks Principales - Guía de Ejecución

Esta carpeta contiene los **4 notebooks principales** que deben ejecutarse en orden para el proyecto de detección de Parkinson mediante análisis de voz.

## 🎯 Objetivo del Proyecto

Implementar un sistema de clasificación binaria (Healthy vs Parkinson) usando redes neuronales convolucionales con **cuantificación de incertidumbre** y **explicabilidad** mediante GradCAM.

---

## 📋 Orden de Ejecución (OBLIGATORIO)

### 1️⃣ **`data_preprocessing.ipynb`** - Preprocesamiento de Datos
### 2️⃣ **`data_augmentation.ipynb`** - Augmentation de Datos  
### 3️⃣ **`cnn_uncertainty_training.ipynb`** - Entrenamiento con Incertidumbre
### 4️⃣ **`gradcam_inference.ipynb`** - Visualización GradCAM

---

## 📖 Documentación Detallada por Notebook

### 1️⃣ **`data_preprocessing.ipynb`** - Preprocesamiento de Datos

#### 🎯 **¿Qué hace?**
Implementa el **preprocesamiento exacto** según el paper de Ibarra et al. (2023) para convertir archivos de audio (.egg) en espectrogramas Mel normalizados.

#### 📚 **Base Científica**
- **Paper**: Ibarra et al. (2023) - "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"
- **Metodología**: Pipeline exacto sin augmentation para reproducibilidad

#### ⚙️ **Pipeline de Preprocesamiento**
1. **Resample**: 44.1 kHz (estándar de audio)
2. **Normalización**: Por amplitud máxima absoluta
3. **Segmentación**: Ventanas de 400ms con 50% overlap (200ms hop)
4. **Mel Spectrogram**: 65 bandas Mel, ventana FFT 40ms, hop 10ms
5. **Conversión**: Amplitud a dB (logarítmica)
6. **Normalización**: Z-score por espectrograma individual
7. **Dimensión final**: 65×41 píxeles

#### 📊 **¿Qué debería ver?**
- **Gráficas de audio**: Formas de onda originales
- **Espectrogramas**: Visualización Mel antes/después de normalización
- **Estadísticas**: Dimensiones, rangos de valores, distribución
- **Cache generado**: `cache/original/healthy_ibarra.pkl` y `cache/original/parkinson_ibarra.pkl`

#### ⏱️ **Tiempo estimado**: 2-3 minutos

#### ✅ **Indicadores de éxito**:
- Cache generado correctamente
- Espectrogramas con dimensión 65×41
- Valores normalizados (media≈0, std≈1)

---

### 2️⃣ **`data_augmentation.ipynb`** - Augmentation de Datos

#### 🎯 **¿Qué hace?**
Aplica **SpecAugment** a los datos Parkinson para mejorar el balance de clases y robustez del modelo.

#### 📚 **Base Científica**
- **Paper**: Park et al. (2019) - "SpecAugment: A Simple Data Augmentation Method for ASR"
- **Técnica**: Máscaras de frecuencia y tiempo en espectrogramas

#### ⚙️ **Pipeline de Augmentation**
1. **Carga datos**: Desde cache preprocesado
2. **SpecAugment**: Máscaras conservadoras (freq=8, time=4)
3. **Generación**: 2 versiones augmentadas por espectrograma original
4. **Guardado**: Dataset augmentado reutilizable

#### 📊 **¿Qué debería ver?**
- **Espectrogramas originales vs augmentados**: Comparación visual
- **Máscaras aplicadas**: Visualización de las máscaras de SpecAugment
- **Estadísticas de balance**: Conteo de muestras por clase
- **Cache generado**: `cache/augmented/augmented_dataset_specaugment.pkl`

#### ⏱️ **Tiempo estimado**: 1-2 minutos

#### ✅ **Indicadores de éxito**:
- Balance mejorado (más muestras Parkinson)
- Espectrogramas augmentados visualmente diferentes
- Cache augmentado generado

---

### 3️⃣ **`cnn_uncertainty_training.ipynb`** - Entrenamiento con Incertidumbre

#### 🎯 **¿Qué hace?**
Entrena una CNN con **dos tipos de incertidumbre**: epistémica (modelo) y aleatoria (datos) según Kendall & Gal (2017).

#### 📚 **Base Científica**
- **Paper**: Kendall & Gal (2017) - "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- **Técnica**: Heteroscedastic loss + MC Dropout

#### 🏗️ **Arquitectura CNN**
```
Input: (B, 1, 65, 41) espectrograma
↓
[Feature Extractor - Ibarra 2023]
Block1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
Block2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(3×3) → Dropout
↓
├─────────────────────────┬─────────────────────────┐
│ [Prediction Head]       │ [Noise Head]            │
│ FC(64) → ReLU → FC(2)  │ FC(64) → ReLU → FC(2)   │
│ ↓                       │ ↓                       │
│ logits (predicción)     │ s_logit (ruido)         │
└─────────────────────────┴─────────────────────────┘
```

#### 🧮 **Matemática de Incertidumbre**
1. **Epistémica (BALD)**: `H(p̄) - E[H(p_t)]` - Incertidumbre del modelo
2. **Aleatoria**: `E[H(p_t)]` - Incertidumbre de los datos
3. **Total**: `H(p̄) = Epistémica + Aleatoria`
4. **Ruido**: `σ = exp(0.5 * s_logit)` - Desviación estándar

#### 📊 **¿Qué debería ver?**
- **Curvas de entrenamiento**: Loss, accuracy, F1-score
- **Matriz de confusión**: Rendimiento por clase
- **Histogramas de incertidumbre**: Distribución de incertidumbre por clase
- **Reliability diagram**: Calibración del modelo
- **Scatter plot**: Incertidumbre vs accuracy
- **Modelo guardado**: `results/cnn_uncertainty/best_model_uncertainty.pth`

#### ⏱️ **Tiempo estimado**: 15-20 minutos

#### ✅ **Indicadores de éxito**:
- Accuracy > 95%
- Incertidumbre mayor en predicciones incorrectas
- Modelo bien calibrado (reliability diagram)

---

### 4️⃣ **`gradcam_inference.ipynb`** - Visualización GradCAM

#### 🎯 **¿Qué hace?**
Genera **mapas de explicabilidad** usando GradCAM para entender qué regiones del espectrograma son importantes para la decisión del modelo.

#### 📚 **Base Científica**
- **Paper**: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **Técnica**: Gradient-weighted Class Activation Mapping

#### 🧮 **Matemática de GradCAM**
1. **Forward pass**: Obtener activaciones de la última capa convolucional
2. **Backward pass**: Calcular gradientes de la clase objetivo
3. **Global Average Pooling**: `w = GAP(∂y/∂A)`
4. **Combinación ponderada**: `CAM = ReLU(Σ w * A)`
5. **Normalización**: `CAM = (CAM - min) / (max - min)`

#### 📊 **¿Qué debería ver?**
- **Espectrogramas originales**: Datos de entrada
- **Mapas GradCAM**: Regiones importantes (colores cálidos)
- **Superposiciones**: GradCAM sobre espectrograma original
- **Comparación por clase**: Diferencias entre Healthy vs Parkinson
- **Análisis de casos**: Predicciones correctas vs incorrectas

#### ⏱️ **Tiempo estimado**: 5-10 minutos

#### ✅ **Indicadores de éxito**:
- Mapas GradCAM coherentes (regiones importantes)
- Diferencias claras entre clases
- Explicaciones visuales interpretables

---

## 🔧 Configuración del Entorno

### Prerequisitos
```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar que los datos están en data/
ls data/vowels_healthy/  # Archivos .egg de sujetos sanos
ls data/vowels_pk/       # Archivos .egg de pacientes Parkinson
```

### Estructura de Datos Requerida
```
data/
├── vowels_healthy/     # Archivos .egg de sujetos sanos
│   ├── 1022-a_lhl-egg.egg
│   ├── 103-u_n-egg.egg
│   └── ...
└── vowels_pk/          # Archivos .egg de pacientes Parkinson
    ├── 1580-a_h-egg.egg
    ├── 1580-a_l-egg.egg
    └── ...
```

---

## 📊 Resultados Esperados

### Después del Notebook 1 (Preprocesamiento)
- **Cache generado**: `cache/original/`
- **Espectrogramas**: 65×41 píxeles, normalizados
- **Tiempo**: ~2-3 minutos

### Después del Notebook 2 (Augmentation)
- **Cache augmentado**: `cache/augmented/`
- **Balance mejorado**: +200% muestras Parkinson
- **Tiempo**: ~1-2 minutos

### Después del Notebook 3 (Entrenamiento)
- **Modelo entrenado**: `results/cnn_uncertainty/`
- **Accuracy**: >95%
- **Incertidumbre cuantificada**: Epistémica + Aleatoria
- **Tiempo**: ~15-20 minutos

### Después del Notebook 4 (GradCAM)
- **Mapas de explicabilidad**: `results/cnn_uncertainty/gradcam_outputs/`
- **Visualizaciones**: Espectrogramas + GradCAM
- **Tiempo**: ~5-10 minutos

---

## 🚨 Troubleshooting

### Error: "Cache not found"
**Solución**: Ejecutar `data_preprocessing.ipynb` primero

### Error: "Model not found"
**Solución**: Ejecutar `cnn_uncertainty_training.ipynb` primero

### Error: "Out of memory"
**Solución**: Reducir `BATCH_SIZE` en el notebook de entrenamiento

### Error: "ImportError"
**Solución**: Verificar que estás en la raíz del proyecto

---

## 📚 Referencias Científicas

1. **Ibarra et al. (2023)**: "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"
2. **Kendall & Gal (2017)**: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
3. **Selvaraju et al. (2017)**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
4. **Park et al. (2019)**: "SpecAugment: A Simple Data Augmentation Method for ASR"

---

## 🎯 Resumen Ejecutivo

Este proyecto implementa un sistema completo de detección de Parkinson que:

1. **Preprocesa** audio según estándares científicos (Ibarra 2023)
2. **Aumenta** datos para mejorar robustez (SpecAugment)
3. **Entrena** CNN con cuantificación de incertidumbre (Kendall & Gal 2017)
4. **Explica** decisiones mediante GradCAM (Selvaraju 2017)

**Resultado**: Sistema de clasificación con >95% accuracy, incertidumbre cuantificada y explicabilidad visual.

---

**Autor**: Wilberth Ferney Córdoba Canchala  
**Fecha**: 2025-01-21  
**Versión**: 4.0 (CNN + Uncertainty + GradCAM)  
