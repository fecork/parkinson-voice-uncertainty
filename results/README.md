# 🎯 Resultados de Entrenamientos

Esta carpeta contiene los resultados de todos los entrenamientos realizados.

## 📁 Estructura

```
results/
├── cnn_no_da/              ← Resultados CNN2D (baseline)
│   ├── best_model.pth
│   ├── test_metrics.json
│   ├── training_progress.png
│   └── confusion_matrix_test.png
│
├── cnn_da/                 ← Resultados CNN2D_DA
│   ├── best_model_da.pth
│   ├── test_metrics_da.json
│   ├── training_progress_da.png
│   └── confusion_matrix_test_da.png
│
└── cnn_da_kfold/           ← Resultados K-fold (desde pipelines)
    ├── fold_*/
    ├── combined_results.json
    └── k_fold_results.png
```

## 📊 Contenido de Cada Carpeta

### Archivos Guardados

#### 🔹 Modelo Entrenado (`.pth`)
```python
# Cargar modelo
import torch
from modules.cnn_model import CNN2D

model = CNN2D(n_classes=2)
model.load_state_dict(torch.load('results/cnn_no_da/best_model.pth'))
model.eval()
```

#### 🔹 Métricas de Test (`.json`)
```json
{
  "accuracy": 0.9880,
  "f1_macro": 0.9878,
  "precision_macro": 0.9867,
  "recall_macro": 0.9893,
  "confusion_matrix": [[228, 5], [0, 183]],
  "classification_report": {...}
}
```

#### 🔹 Gráficas de Progreso (`.png`)
- Loss de entrenamiento y validación por época
- Accuracy por época
- Identificación de mejor época

#### 🔹 Matriz de Confusión (`.png`)
- Visualización de predicciones vs ground truth
- Healthy vs Parkinson

---

## 📂 Subcarpetas Detalladas

### `cnn_no_da/` - CNN2D Baseline

**Origen**: `cnn_training.ipynb`

**Modelo**: CNN2D sin Domain Adaptation

**Configuración**:
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100 (con early stopping)
- Split: 70/15/15 (train/val/test)

**Métricas Típicas**:
- Accuracy: ~98.8%
- F1-Score: ~98.8%

---

### `cnn_da/` - CNN2D_DA

**Origen**: `cnn_da_training.ipynb`

**Modelo**: CNN2D_DA con Gradient Reversal Layer

**Configuración**:
- Optimizer: SGD (según Ibarra 2023)
- Learning Rate: 0.1
- Alpha (peso domain): 1.0
- Lambda GRL: 1.0
- Epochs: 100 (con early stopping)

**Métricas Guardadas**:
```json
{
  "loss_pd": ...,
  "loss_domain": ...,
  "loss_total": ...,
  "acc_pd": ...,
  "f1_pd": ...,
  "precision_pd": ...,
  "recall_pd": ...,
  "acc_domain": ...
}
```

---

### `cnn_da_kfold/` - K-fold Cross-Validation

**Origen**: `pipelines/train_cnn_da_kfold.py`

**Modelo**: CNN2D_DA con 10-fold CV

**Estructura**:
```
cnn_da_kfold/
├── fold_0/
│   ├── best_model_fold_0.pth
│   ├── metrics_fold_0.json
│   └── ...
├── fold_1/
├── ...
├── fold_9/
└── combined_results.json    ← Métricas agregadas
```

**Métricas Agregadas**:
- Accuracy promedio ± std
- F1-Score promedio ± std
- Resultados por fold

---

## 📈 Comparación de Resultados

### Ver Métricas

```python
import json

# CNN2D
with open("results/cnn_no_da/test_metrics.json") as f:
    metrics_cnn = json.load(f)
    print(f"CNN2D Accuracy: {metrics_cnn['accuracy']:.4f}")

# CNN2D_DA
with open("results/cnn_da/test_metrics_da.json") as f:
    metrics_da = json.load(f)
    print(f"CNN2D_DA Accuracy: {metrics_da['acc_pd']:.4f}")
```

### Comparar Visualmente

```bash
# Abrir carpetas de resultados
explorer results/cnn_no_da/      # Windows
# o
open results/cnn_no_da/          # Mac
# o
xdg-open results/cnn_no_da/      # Linux

# Comparar:
# - training_progress.png (ambos modelos)
# - confusion_matrix_test.png (ambos modelos)
# - test_metrics.json (ambos modelos)
```

---

## 🔍 Análisis de Resultados

### Métricas Clave

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | % de predicciones correctas |
| **Precision** | De las predicciones positivas, % correctas |
| **Recall** | De los casos positivos reales, % detectados |
| **F1-Score** | Media armónica de Precision y Recall |

### Matriz de Confusión

```
              Pred HC  Pred PD
Real HC       [  TP  ] [ FP  ]
Real PD       [  FN  ] [ TP  ]

TP: True Positives
FP: False Positives
FN: False Negatives
```

---

## 🚨 Importante

### ⚠️ No Modificar
No modificar manualmente los archivos de resultados.

### ⚠️ Git Ignore
Los archivos `.pth` están en `.gitignore` por su tamaño.

### ⚠️ Backup
Hacer backup de resultados importantes antes de re-entrenar.

---

## 🎯 Organización de Experimentos

### Nombrado de Carpetas

Para múltiples experimentos, crear subcarpetas:

```
results/
├── cnn_no_da/
│   ├── exp1_lr_0.001/
│   ├── exp2_lr_0.01/
│   └── exp3_lr_0.0001/
│
└── cnn_da/
    ├── exp1_alpha_1.0/
    └── exp2_alpha_0.5/
```

### Registro de Experimentos

Mantener un log de experimentos:

```
experiments_log.txt
-------------------
2025-10-17 14:30 - CNN2D - LR=0.001 - Accuracy=98.8%
2025-10-17 15:00 - CNN2D - LR=0.01  - Accuracy=97.5%
...
```

---

**Última actualización**: 2025-10-17

