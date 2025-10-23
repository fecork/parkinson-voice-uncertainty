# 🔬 Research - Investigaciones Doctorado

Esta carpeta contiene notebooks de investigación personal para el doctorado en detección de Parkinson mediante análisis de voz.

## ⚠️ Nota Importante

**Los notebooks en esta carpeta NO deben ser revisados por el profesor.** Son investigaciones personales y experimentos adicionales que no forman parte del entregable principal.

## 📁 Contenido

### Notebooks de Investigación

#### 1. `cnn_training.ipynb`
- **Propósito**: CNN2D baseline con augmentation
- **Arquitectura**: Single-head CNN sin Domain Adaptation
- **Augmentation**: Pitch shifting, time stretching, noise injection, SpecAugment
- **Uso**: Modelo baseline para comparación

#### 2. `cnn_da_training.ipynb`
- **Propósito**: CNN2D con Domain Adaptation
- **Arquitectura**: Dual-head CNN con Gradient Reversal Layer (GRL)
- **Referencia**: Implementación exacta según Ibarra et al. (2023)
- **Uso**: Modelo con adaptación de dominio

#### 3. `cnn1d_da_training.ipynb`
- **Propósito**: CNN1D con Domain Adaptation y atención temporal
- **Arquitectura**: CNN1D + atención temporal + GRL
- **Uso**: Modelado temporal de características

#### 4. `lstm_da_training.ipynb`
- **Propósito**: Time-CNN-BiLSTM con Domain Adaptation
- **Arquitectura**: Time-distributed CNN + BiLSTM + GRL
- **Uso**: Modelado temporal de secuencias completas

## 🎯 Objetivos de Investigación

### Comparación de Arquitecturas
- **CNN2D**: Baseline sin Domain Adaptation
- **CNN2D_DA**: Con Domain Adaptation
- **CNN1D_DA**: Modelado temporal con atención
- **LSTM_DA**: Modelado temporal de secuencias

### Técnicas Implementadas
- **Domain Adaptation**: Gradient Reversal Layer (GRL)
- **Data Augmentation**: SpecAugment, pitch shifting, time stretching
- **Attention Mechanisms**: Temporal attention en CNN1D
- **Sequence Modeling**: BiLSTM para dependencias temporales

### Métricas de Evaluación
- **Accuracy**: Precisión de clasificación
- **F1-Score**: Balance entre precisión y recall
- **Confusion Matrix**: Análisis de errores
- **t-SNE**: Visualización de embeddings

## 🔬 Experimentos Realizados

### 1. Comparación de Modelos
- CNN2D vs CNN2D_DA
- CNN1D vs CNN1D_DA
- CNN vs LSTM para modelado temporal

### 2. Análisis de Domain Adaptation
- Efectividad del GRL
- Lambda warm-up
- Multi-task learning

### 3. Análisis Temporal
- Secuencias de diferentes longitudes
- Attention weights
- Dependencias temporales

## 📊 Resultados de Investigación

### Modelos Entrenados
- **CNN2D**: ~98.8% accuracy
- **CNN2D_DA**: TBD
- **CNN1D_DA**: TBD
- **LSTM_DA**: TBD

### Análisis de Incertidumbre
- **Epistémica**: MC Dropout
- **Aleatoria**: Heteroscedastic loss
- **GradCAM**: Visualización de explicabilidad

## 🚀 Uso de los Notebooks

### Prerequisitos
1. Ejecutar `notebooks/data_preprocessing.ipynb` primero
2. Tener cache generado en `cache/original/`

### Orden de Ejecución
1. **CNN2D**: `cnn_training.ipynb`
2. **CNN2D_DA**: `cnn_da_training.ipynb`
3. **CNN1D_DA**: `cnn1d_da_training.ipynb`
4. **LSTM_DA**: `lstm_da_training.ipynb`

### Configuración
- **GPU**: Recomendado para entrenamiento
- **Memoria**: Al menos 8GB RAM
- **Tiempo**: 10-20 minutos por modelo

## 📚 Referencias

### Papers Principales
- **Ibarra et al. (2023)**: "Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"
- **Selvaraju et al. (2017)**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **Kendall & Gal (2017)**: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

### Técnicas Implementadas
- **Domain Adaptation**: GRL para invarianza multi-corpus
- **Data Augmentation**: SpecAugment para robustez
- **Uncertainty Quantification**: MC Dropout + Heteroscedastic loss
- **Explainability**: GradCAM para interpretabilidad

## 🔧 Configuración Técnica

### Hiperparámetros
- **Learning Rate**: 0.001 (Adam) / 0.1 (SGD)
- **Batch Size**: 32
- **Epochs**: 100
- **Dropout**: 0.3 (conv) / 0.5 (fc)

### Arquitecturas
- **CNN2D**: 2 bloques Conv2D + MaxPool(3×3)
- **CNN1D**: 3 bloques Conv1D + atención temporal
- **LSTM**: Time-distributed CNN + BiLSTM

## 📈 Análisis de Resultados

### Métricas por Modelo
- **Accuracy**: Clasificación correcta
- **F1-Score**: Balance precisión/recall
- **Confusion Matrix**: Análisis de errores
- **ROC Curve**: Curva ROC

### Visualizaciones
- **Training Curves**: Loss y accuracy
- **Attention Maps**: Pesos de atención
- **GradCAM**: Mapas de activación
- **t-SNE**: Embeddings 2D

## 🎓 Contribuciones de Investigación

### Nuevas Técnicas
- **Temporal Attention**: En CNN1D para modelado temporal
- **Sequence Modeling**: BiLSTM para dependencias temporales
- **Uncertainty Quantification**: MC Dropout + Heteroscedastic loss

### Mejoras Implementadas
- **Modular Design**: Componentes reutilizables
- **Efficient Training**: K-fold cross-validation
- **Robust Evaluation**: Speaker-independent splits

## 📝 Notas de Desarrollo

### Versiones
- **v1.0**: Implementación inicial CNN2D
- **v2.0**: Domain Adaptation (GRL)
- **v3.0**: CNN1D con atención temporal
- **v4.0**: LSTM con modelado temporal

### Próximos Pasos
- **Ensemble Methods**: Combinación de modelos
- **Transfer Learning**: Pre-trained models
- **Multi-modal**: Audio + texto + metadata

---

**Autor**: PHD Research Team  
**Fecha**: 2025-01-21  
**Versión**: 4.0 (LSTM + Uncertainty)  
**Estado**: Investigación en progreso
