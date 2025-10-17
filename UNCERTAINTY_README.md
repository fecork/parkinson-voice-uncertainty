# 🧠 Sistema de Estimación de Incertidumbre para CNN

## 📋 Resumen

Este sistema implementa estimación de incertidumbre **epistémica** y **aleatoria** en un modelo CNN para clasificación de Parkinson vs Healthy usando señales de voz.

### 🎯 Tipos de Incertidumbre

1. **Epistémica (modelo)**: 
   - Capturada con **MC Dropout**
   - Representa incertidumbre en los parámetros del modelo
   - **Reducible** con más datos de entrenamiento
   
2. **Aleatoria (datos)**:
   - Capturada con **cabeza de varianza** (heteroscedástica)
   - Representa ruido inherente en los datos
   - **Irreducible** (ruido intrínseco)

---

## 🏗️ Arquitectura del Modelo

### Backbone (Feature Extractor)
```
Input [B, 1, 65, 41]
    ↓
Bloque 1: Conv2D(32) → BN → ReLU → MaxPool(3×3) → MCDropout(0.25)
    ↓
Bloque 2: Conv2D(64) → BN → ReLU → MaxPool(3×3) → MCDropout(0.25)
    ↓
AdaptiveAvgPool → Flatten
    ↓
Features [B, feat_size]
```

### Cabeza A: Predicción
```
Features → Linear(64) → ReLU → MCDropout(0.25) → Linear(C)
    ↓
logits [B, C]
```

### Cabeza B: Ruido de Datos
```
Features → Linear(64) → ReLU → MCDropout(0.25) → Linear(C)
    ↓
s_logit = clamp(x, -10, 3) [B, C]  # log-varianza
```

**Nota clave**: Se usa `MCDropout` que permanece activo incluso en `eval()` para hacer MC Dropout en inferencia.

---

## 🔬 Entrenamiento

### Pérdida Heteroscedástica

En lugar de Cross-Entropy estándar, usamos log-likelihood con ruido gaussiano:

```python
σ = exp(0.5 * s_logit)  # Desviación estándar por clase

# Para T_noise muestras:
for t in range(T_noise):
    ε ~ N(0, 1)
    x̂_t = logits + σ ⊙ ε  # Logits con ruido
    logp_t = log_softmax(x̂_t)[y]  # Log-prob de clase correcta

# Log-mean-exp estable:
m = max_t logp_y_t
loss = -mean_batch( m + log(mean_t exp(logp_y_t - m)) )
```

**Ventajas**:
- El modelo aprende σ² útil para cada clase
- Más robusto a ruido en etiquetas
- Mejor calibración

### Hiperparámetros de Entrenamiento

```python
N_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = AdamW
T_NOISE = 5  # Muestras de ruido en training
DROPOUT_P = 0.25
S_CLAMP = [-10.0, 3.0]
BATCH_SIZE = 128
EARLY_STOPPING = 15 épocas
```

---

## 🔮 Inferencia con MC Dropout

### Procedimiento

```python
model.eval()  # Pero MCDropout sigue activo

# T_test pasadas (30-50)
for t in range(T_test):
    logits_t, s_logit_t = model(x)
    p_t = softmax(logits_t)
    σ²_t = exp(s_logit_t)
    
    guardar(p_t, σ²_t)

# Agregación
p̄ = mean_t(p_t)  # Probabilidades promedio
pred = argmax(p̄)   # Predicción final
```

### Cálculo de Incertidumbres

#### 1. Entropía Total (Predictiva)
```python
H(p̄) = -Σ p̄_c log(p̄_c)
```
Mide incertidumbre total en la predicción.

#### 2. Epistémica (BALD)
```python
BALD = H(p̄) - mean_t(H(p_t))
```
- `H(p̄)`: Entropía del promedio
- `mean H(p_t)`: Promedio de entropías individuales
- **Interpretación**: Desacuerdo entre diferentes "versiones" del modelo (MC Dropout)

#### 3. Aleatoria
```python
σ²_aleatoric = mean_t( mean_c(σ²_t_c) )
```
Promedio de la varianza estimada por la cabeza B.

---

## 📊 Métricas de Evaluación

### Clasificación
- **Accuracy**: % aciertos
- **Precision, Recall, F1**: Por clase y macro-promedio

### Calibración
- **NLL** (Negative Log-Likelihood): `- mean log(p̄[y])`
- **Brier Score**: `mean((p̄ - y_one_hot)²)`
- **ECE** (Expected Calibration Error): Diferencia entre confianza y accuracy por bins

### Incertidumbre
- **H_total**: Entropía promedio
- **Epistémica**: BALD promedio
- **Aleatoria**: σ² promedio
- **Separación correcto/incorrecto**: ¿Los errores tienen mayor incertidumbre?

---

## 📈 Visualizaciones

### 1. Histogramas de Incertidumbres
```
[Entropía]     [Epistémica]    [Aleatoria]
Correcto ✅    Correcto ✅     Correcto ✅
Incorrecto ❌  Incorrecto ❌   Incorrecto ❌
```
**Esperado**: Los incorrectos tienen **mayor** incertidumbre.

### 2. Reliability Diagram
```
Accuracy vs Confidence
Perfecto: diagonal 45°
```
Mide si el modelo está bien calibrado (confianza = accuracy real).

### 3. Scatter Epistémica vs Aleatoria
```
Y: Aleatoria
X: Epistémica
Color: Correcto (verde) / Incorrecto (rojo)
```
Permite ver si errores se deben a modelo o datos.

### 4. Matriz de Confusión
Confusión tradicional HC vs PD.

---

## 🚀 Uso del Sistema

### 1. Pre-requisito
```bash
# Ejecutar primero el notebook de preprocesamiento
jupyter notebook data_preprocessing.ipynb
```

### 2. Entrenar modelo con incertidumbre
```bash
jupyter notebook cnn_uncertainty_training.ipynb
```

### 3. Resultados guardados en `results/cnn_uncertainty/`:
```
best_model_uncertainty.pth
test_metrics_uncertainty.json
training_history.png
uncertainty_histograms.png
reliability_diagram.png
uncertainty_scatter.png
confusion_matrix_test.png
```

---

## 📁 Estructura de Archivos

```
modules/
├── uncertainty_model.py           # Modelo UncertaintyCNN + MCDropout
├── uncertainty_loss.py            # Pérdida heteroscedástica
├── uncertainty_training.py        # Train/eval con MC Dropout
└── uncertainty_visualization.py   # Plots de incertidumbres

cnn_uncertainty_training.ipynb     # Notebook principal
UNCERTAINTY_README.md              # Este archivo
```

---

## 🔬 Interpretación de Resultados

### Caso Ideal

| Métrica | Valor Esperado | Significado |
|---------|----------------|-------------|
| H(correctos) | **Bajo** | Predicciones seguras en aciertos |
| H(incorrectos) | **Alto** | Modelo "sabe que no sabe" |
| Epistémica alta | ➜ **Más datos** | Modelo incierto |
| Aleatoria alta | ➜ **Datos ruidosos** | Límite intrínseco |
| ECE | **< 0.05** | Bien calibrado |

### Ejemplo de Salida

```
📊 MÉTRICAS DE CLASIFICACIÓN:
  Accuracy:  0.9850
  F1-Score:  0.9845

📈 CALIBRACIÓN:
  NLL:  0.0523
  ECE:  0.0234  ✅ Bien calibrado

🎲 INCERTIDUMBRES:
  Total:      0.1234
  Epistémica: 0.0456  (reducible con más datos)
  Aleatoria:  0.0778  (ruido intrínseco)

✅ ❌ SEPARACIÓN:
  H(correctos):   0.0892  ✅ Baja
  H(incorrectos): 0.4567  ✅ Alta (modelo detecta errores)
```

---

## 🎯 Ventajas del Sistema

1. **Cuantifica confianza**: No solo predicción, también incertidumbre
2. **Detecta errores**: Modelo "sabe cuando no sabe"
3. **Diagnóstico**: Separa problemas de modelo vs datos
4. **Mejor calibración**: Pérdida heteroscedástica mejora confianza
5. **OOD Detection**: Alto H(p̄) indica muestra fuera de distribución

---

## 💡 Trucos y Gotchas

### ⚠️ Estabilidad Numérica

```python
# ✅ BIEN
σ = exp(0.5 * s_logit)  # Desviación estándar

# ❌ MAL
σ = exp(s_logit)  # Explota numéricamente

# ✅ Clamp de s_logit
s = clamp(s, min=-10, max=3)
```

### ⚠️ MC Dropout

```python
# ✅ BIEN: Usar MCDropout personalizado
class MCDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)  # Siempre activo

# ❌ MAL: Dropout normal no funciona en eval()
```

### ⚠️ Entropía Estable

```python
# ✅ Añadir epsilon
H = -(p * torch.log(p + 1e-12)).sum()

# ❌ Sin epsilon da NaN
H = -(p * torch.log(p)).sum()  # log(0) = -inf
```

---

## 📚 Referencias

### Teóricas
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities"

### Implementación
- Código base: `cnn_training.ipynb` (sin incertidumbre)
- Arquitectura backbone: Ibarra et al. (2023)

---

## 🔄 Comparación con Modelo Base

| Aspecto | CNN Base | CNN Incertidumbre |
|---------|----------|-------------------|
| Cabezas | 1 (logits) | 2 (logits + σ²) |
| Pérdida | CrossEntropy | Heteroscedástica |
| Inferencia | 1 pase | T_test pases (MC) |
| Output | `pred, probs` | `pred, H, epi, ale` |
| Calibración | Estándar | **Mejorada** |
| Tiempo train | ~5 min | ~7-8 min (+40%) |
| Tiempo test | <1 s | ~10-15 s (MC×30) |

---

## ✅ Checklist de Validación

- [ ] Entropía incorrectos > correctos
- [ ] ECE < 0.10 (idealmente < 0.05)
- [ ] Reliability diagram cerca de diagonal
- [ ] NLL razonable (< 0.15)
- [ ] Accuracy similar al modelo base
- [ ] Scatter muestra separación errores
- [ ] Pérdida converge sin NaN

---

**Creado**: 2024  
**Autor**: Sistema de estimación de incertidumbre para diagnóstico de Parkinson  
**Licencia**: [Tu licencia aquí]

