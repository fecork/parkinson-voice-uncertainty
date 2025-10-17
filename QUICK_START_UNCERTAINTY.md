# 🚀 Quick Start: Sistema de Incertidumbre

## ¿Qué hemos construido?

Un sistema completo de **estimación de incertidumbre epistémica + aleatoria** para clasificación de Parkinson basado en CNN.

---

## 📦 Componentes Creados

### 1. Módulos Python (`modules/`)

#### `uncertainty_model.py`
- **UncertaintyCNN**: Modelo con 2 cabezas
  - Cabeza A: logits (predicción)
  - Cabeza B: s_logit (log-varianza)
- **MCDropout/MCDropout2d**: Dropout activo en inferencia
- Función de resumen del modelo

#### `uncertainty_loss.py`
- **heteroscedastic_classification_loss**: Pérdida con ruido gaussiano
- **compute_nll**: Negative Log-Likelihood
- **compute_brier_score**: Error cuadrático de probabilidades
- **compute_ece**: Expected Calibration Error

#### `uncertainty_training.py`
- **train_uncertainty_model**: Loop de entrenamiento completo
- **evaluate_with_uncertainty**: Evaluación con MC Dropout (T_test pases)
- **print_uncertainty_results**: Imprime métricas bonitas

#### `uncertainty_visualization.py`
- **plot_uncertainty_histograms**: 3 histogramas (total, epistémica, aleatoria)
- **plot_reliability_diagram**: Diagrama de calibración
- **plot_uncertainty_scatter**: Scatter epistémica vs aleatoria
- **plot_training_history_uncertainty**: Progreso de entrenamiento

### 2. Notebook Principal

#### `cnn_uncertainty_training.ipynb`
Pipeline completo:
1. Setup
2. Carga de datos (cache)
3. Split (70/15/15)
4. Creación del modelo
5. Entrenamiento
6. Evaluación con MC Dropout
7. Visualizaciones

### 3. Documentación

- **UNCERTAINTY_README.md**: Documentación técnica completa
- **QUICK_START_UNCERTAINTY.md**: Este archivo

---

## ⚡ Uso Rápido

### Paso 1: Preprocesar datos (si no lo has hecho)
```bash
jupyter notebook data_preprocessing.ipynb
# Ejecutar todas las celdas
```

### Paso 2: Entrenar modelo con incertidumbre
```bash
jupyter notebook cnn_uncertainty_training.ipynb
# Ejecutar todas las celdas
```

### Paso 3: Ver resultados
Los archivos se guardan en `results/cnn_uncertainty/`:
- `best_model_uncertainty.pth`: Modelo entrenado
- `test_metrics_uncertainty.json`: Métricas JSON
- Varias gráficas PNG

---

## 🔬 Detalles Técnicos Clave

### Entrenamiento
```python
# T_noise = 5 (rápido, estable)
for batch in train_loader:
    logits, s_logit = model(x)
    loss = heteroscedastic_loss(logits, s_logit, y, T_noise=5)
    loss.backward()
```

### Inferencia
```python
# T_test = 30 (más muestras = mejor estimación)
results = model.predict_with_uncertainty(x, n_samples=30)

# Outputs:
# - pred: Predicción final
# - probs_mean: Probabilidades promedio
# - entropy_total: H(p̄)
# - epistemic: BALD
# - aleatoric: σ²
# - confidence: max(p̄)
```

---

## 📊 Interpretando Resultados

### Buena Señal ✅
```
Entropy(correctos):   0.08  ← Bajo
Entropy(incorrectos): 0.45  ← Alto
ECE: 0.03  ← Bien calibrado
```
➜ El modelo **sabe cuándo está inseguro**

### Mala Señal ❌
```
Entropy(correctos):   0.25  ← Similar
Entropy(incorrectos): 0.28  ← Similar
ECE: 0.15  ← Mal calibrado
```
➜ Modelo no distingue certeza

### Diagnóstico

| Alta Epistémica | ➜ Necesita más datos |
| Alta Aleatoria | ➜ Datos ruidosos (límite intrínseco) |

---

## 🎯 Próximos Pasos

### Experimentación
1. Ajustar `T_noise` (probar 3, 5, 10)
2. Ajustar `T_test` (probar 30, 50, 100)
3. Probar `dropout_p` (0.2, 0.25, 0.3)
4. Ajustar clamp de `s_logit`

### Comparación
- Comparar con modelo base (`cnn_training.ipynb`)
- Comparar con modelo DA (`cnn_da_training.ipynb`)
- Añadir incertidumbre al modelo DA

### Aplicación
- Usar incertidumbre para detectar casos difíciles
- Filtrar predicciones por confianza
- Active learning (seleccionar samples con alta epistémica)

---

## ⚙️ Hiperparámetros Recomendados

| Parámetro | Valor Inicial | Rango | Notas |
|-----------|---------------|-------|-------|
| `dropout_p` | 0.25 | 0.2-0.3 | Mayor = más epistémica |
| `T_noise` | 5 | 3-10 | Más = estable pero lento |
| `T_test` | 30 | 30-100 | Más = mejor estimación |
| `s_min` | -10.0 | -15 a -5 | exp(-10/2) ≈ 0.0067 |
| `s_max` | 3.0 | 2 a 5 | exp(3/2) ≈ 4.48 |
| `lr` | 1e-3 | 1e-4 a 1e-2 | AdamW |
| `weight_decay` | 1e-4 | 1e-5 a 1e-3 | Regularización |

---

## 🐛 Troubleshooting

### Pérdida diverge (NaN)
- Reducir `T_noise` a 3
- Ajustar `s_min/s_max` a [-15, 2]
- Reducir `lr` a 5e-4
- Verificar que no haya NaN en datos

### Epistémica muy baja
- Aumentar `dropout_p` a 0.3
- Verificar que MCDropout esté activo en eval

### Aleatoria muy alta
- Revisar calidad de datos
- Puede ser normal si datos tienen ruido intrínseco
- Considerar limpieza de datos

### ECE alto (> 0.10)
- Aumentar `T_noise`
- Ajustar `weight_decay`
- Considerar temperature scaling post-training

---

## 📞 Contacto

Para preguntas o issues sobre el sistema de incertidumbre, consultar:
- `UNCERTAINTY_README.md` para detalles técnicos
- Código fuente en `modules/uncertainty_*.py`
- Notebook de ejemplo: `cnn_uncertainty_training.ipynb`

---

**Happy Uncertainty Estimation! 🎲**

