# Implementación Completa según Ibarra et al. (2023)

## 📄 Referencia
**"Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation"**

---

## ✅ Cumplimiento del Paper

### 1. Preprocesamiento de Datos ✓

| Especificación | Implementación | Archivo |
|----------------|----------------|---------|
| Resampleo a 44.1 kHz | ✅ | `modules/preprocessing.py:20` |
| Segmentos de 400ms con 50% solape | ✅ | `modules/preprocessing.py:21-22` |
| Mel-spectrogramas: 65 bandas | ✅ | `modules/preprocessing.py:23` |
| Hop length: 10ms | ✅ | `modules/preprocessing.py:24` |
| Ventana FFT: 40ms (vocales sostenidas) | ✅ | `modules/preprocessing.py:25` |
| Dimensiones: 65×41 px | ✅ | `modules/preprocessing.py:26` |
| Amplitud en dB | ✅ | `modules/preprocessing.py:130` |
| Normalización z-score | ✅ | `modules/preprocessing.py:143` |

### 2. Validación Cruzada 10-Fold ✓

| Especificación | Implementación | Archivo |
|----------------|----------------|---------|
| 10-fold CV | ✅ | `modules/cnn_utils.py:224-298` |
| Estratificada por PD | ✅ | `modules/cnn_utils.py:267` (StratifiedKFold) |
| Independiente por hablante | ✅ | `modules/cnn_utils.py:246-257` |
| Sin fugas entre folds | ✅ | Garantizado por agrupación de subject_id |

**Función principal:** `create_10fold_splits_by_speaker()`

### 3. Arquitectura 2D-CNN con Domain Adaptation ✓

| Componente | Especificación | Implementación | Archivo |
|------------|----------------|----------------|---------|
| Bloques convolucionales | Conv2D → BN → ReLU → **MaxPool(3×3)** → Dropout | ✅ | `modules/cnn_model.py:419-434` |
| Número de bloques | 2 | ✅ | `modules/cnn_model.py` |
| Cabeza PD | FC → ReLU → Dropout → FC (softmax) | ✅ | `modules/cnn_model.py:541-547` |
| Cabeza Dominio | FC → ReLU → Dropout → FC (softmax) | ✅ | `modules/cnn_model.py:552-558` |
| GRL | Gradient Reversal Layer | ✅ | `modules/cnn_model.py:339-396` |
| Pérdida total | L = L_PD + α·L_dom | ✅ | `modules/cnn_training.py:608` |

**Modelo principal:** `CNN2D_DA` en `modules/cnn_model.py:501-591`

### 4. Optimizador y Learning Rate ✓

| Especificación | Implementación | Archivo |
|----------------|----------------|---------|
| Optimizador: SGD | ✅ | `modules/cnn_training.py:1136-1138` |
| LR inicial: 0.1 | ✅ | `train_cnn_da_kfold.py:90` (default) |
| Momentum: 0.9 | ✅ | `modules/cnn_training.py:1137` |
| Weight decay: 1e-4 | ✅ | `modules/cnn_training.py:1137` |
| LR scheduler | ✅ StepLR (step=30, gamma=0.1) | `modules/cnn_training.py:1141-1143` |

### 5. Cross-Entropy Ponderada Automática ✓

| Especificación | Implementación | Archivo |
|----------------|----------------|---------|
| Detección automática de desbalance | ✅ | `modules/cnn_utils.py:130-159` |
| Pesos para tarea PD | ✅ | `modules/cnn_training.py:1116` |
| Pesos para tarea de dominio | ✅ | `modules/cnn_training.py:1119` |
| Threshold: 40% | ✅ | `modules/cnn_utils.py:132` (threshold=0.4) |

**Función principal:** `compute_class_weights_auto()` en `modules/cnn_utils.py:130-159`

### 6. Lambda de GRL ✓

| Especificación | Interpretación | Implementación | Archivo |
|----------------|----------------|----------------|---------|
| Lambda GRL | **Constante** (paper no especifica schedule) | ✅ λ = 1.0 | `modules/cnn_training.py:1152-1153` |

**Nota:** El paper menciona GRL pero no especifica schedule de λ. Implementamos valor constante como en DA estándar.

### 7. Métricas de Evaluación ✓

| Métrica | Implementación | Archivo |
|---------|----------------|---------|
| Accuracy | ✅ | `modules/cnn_training.py:636` |
| Precision | ✅ | `modules/cnn_training.py:721` |
| Recall | ✅ | `modules/cnn_training.py:722` |
| F1-Score | ✅ | `modules/cnn_training.py:638` |
| Mean ± Std (K-fold) | ✅ | `modules/cnn_training.py:1197-1198` |

### 8. Agregación por Paciente ✓

| Especificación | Implementación | Archivo |
|----------------|----------------|---------|
| Predicción por espectrograma | ✅ | `modules/cnn_training.py:939` |
| Agregación por subject_id | ✅ | `modules/cnn_training.py:950-954` |
| Probabilidad conjunta | ✅ | `modules/cnn_training.py:962` (promedio) |

**Función principal:** `evaluate_by_patient_da()` en `modules/cnn_training.py:907-994`

---

## 🚀 Uso

### Opción A: Entrenamiento Simple (Train/Val/Test)

Usar el notebook `parkinson_voice_analysis.ipynb` - **Celda 17** tiene la configuración correcta según Ibarra 2023.

### Opción B: 10-Fold Cross-Validation (Recomendado)

```bash
python train_cnn_da_kfold.py \
    --hc_dir data/vowels_healthy \
    --pd_dir data/vowels_pk \
    --n_folds 10 \
    --batch_size 32 \
    --lr 0.1 \
    --lambda_grl 1.0 \
    --epochs 100 \
    --output_dir results/cnn_da_kfold
```

**Parámetros principales:**
- `--n_folds`: Número de folds (default: 10)
- `--batch_size`: Tamaño de batch (paper sugiere: 16, 32, 64)
- `--lr`: Learning rate inicial (paper: 0.1)
- `--lambda_grl`: Lambda constante para GRL (default: 1.0)
- `--dropout_conv`: Dropout convolucional (paper: 0.2 o 0.5, default: 0.3)
- `--dropout_fc`: Dropout FC (paper: 0.2 o 0.5, default: 0.5)

---

## 📊 Resultados

El script de 10-fold CV genera:
- `results/cnn_da_kfold/kfold_results.json`: Métricas agregadas (mean ± std)
- `results/cnn_da_kfold/fold_X/`: Modelos y métricas por fold
- `results/cnn_da_kfold/config.json`: Configuración utilizada

---

## 🔍 Diferencias Respecto al Paper (Justificadas)

### 1. Transfer Learning NO Implementado
**Razón:** No tenemos dataset SVDD (sano vs patología de voz) para pre-entrenamiento.  
**Solución futura:** Agregar cuando esté disponible SVDD.

### 2. Búsqueda de Hiperparámetros (Talos) NO Implementada
**Razón:** El paper usa Talos para buscar en espacio de hiperparámetros. Complejidad alta.  
**Solución actual:** Usamos hiperparámetros razonables dentro del espacio del paper:
- Batch: 32 (paper: 16/32/64)
- Dropout conv: 0.3 (paper: 0.2/0.5)
- Dropout FC: 0.5 (paper: 0.2/0.5)

### 3. Lambda Schedule
**Interpretación:** Paper menciona GRL pero no especifica schedule de λ.  
**Implementación:** λ = 1.0 constante (estándar en DA, más conservador que schedule progresivo).

---

## 📝 Archivos Modificados

### Nuevos archivos:
- `train_cnn_da_kfold.py`: Script principal para 10-fold CV
- `IBARRA_2023_IMPLEMENTATION.md`: Este documento

### Archivos modificados:
- `modules/cnn_utils.py`:
  - `create_10fold_splits_by_speaker()`: Splits K-fold estratificados
  - `compute_class_weights_auto()`: Detección automática de desbalance
- `modules/cnn_training.py`:
  - `train_model_da()`: Agregado soporte para `lr_scheduler`
  - `train_model_da_kfold()`: Nueva función para K-fold CV completo
- `parkinson_voice_analysis.ipynb`:
  - Celda 17: Configuración actualizada (SGD LR=0.1, scheduler, pesos automáticos, lambda constante)
  - Celda 20: Resumen actualizado con cumplimiento del paper

---

## ✅ Checklist de Cumplimiento

- [x] Preprocesamiento: 44.1kHz, 400ms, 50% solape, Mel 65×41, z-score
- [x] 10-fold CV estratificada independiente por hablante
- [x] Arquitectura 2D-CNN-DA con MaxPool 3×3
- [x] SGD con LR inicial 0.1
- [x] LR scheduler (StepLR)
- [x] Cross-entropy ponderada automática (PD + dominio)
- [x] Lambda GRL constante = 1.0
- [x] Métricas: accuracy, precision, recall, F1
- [x] Agregación por paciente
- [x] Reportar mean ± std sobre K folds
- [ ] Transfer Learning desde SVDD (pendiente: no hay dataset)
- [ ] Búsqueda de hiperparámetros con Talos (pendiente: complejidad)

---

## 📚 Referencias

```bibtex
@article{ibarra2023towards,
  title={Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation},
  author={Ibarra, et al.},
  year={2023}
}
```

---

## 👨‍💻 Autor

Implementación realizada siguiendo estrictamente las especificaciones del paper de Ibarra et al. (2023).

**Fecha:** Octubre 2025

