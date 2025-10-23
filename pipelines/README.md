# 🚀 Pipelines de Entrenamiento

Esta carpeta contiene **pipelines completos end-to-end** para entrenar modelos CNN de detección de Parkinson.

## 🎯 ¿Qué son los Pipelines?

Los pipelines son programas automatizados que ejecutan un flujo completo de trabajo desde la carga de datos hasta la generación de resultados finales. A diferencia de los módulos (que son componentes individuales) o notebooks (que son interactivos), los pipelines:

- ✅ Se ejecutan desde línea de comandos
- ✅ Procesan todo automáticamente (sin intervención)
- ✅ Son reproducibles y configurables
- ✅ Ideales para producción, experimentos batch, y HPC

---

## 📁 Pipelines Disponibles

### 1. `train_cnn.py` - CNN2D sin Domain Adaptation

**Pipeline completo para modelo baseline con MC Dropout**

**Flujo**:
```
Datos → Preprocesar → Split por Speaker → Entrenar CNN2D → 
Evaluar → MC Dropout Inference → Agregar por Archivo/Paciente → 
Visualizaciones
```

**Características**:
- Modelo: CNN2D simple (single-head)
- Validación: Train/Val/Test split speaker-independent
- Incluye: MC Dropout para cuantificación de incertidumbre
- Salida: `results/cnn_training/`

**Uso** (desde la raíz del proyecto):
```bash
# Básico
python pipelines/train_cnn.py

# Personalizado
python pipelines/train_cnn.py \
    --hc_dir data/vowels_healthy \
    --pd_dir data/vowels_pk \
    --epochs 100 \
    --lr 0.001 \
    --mc_samples 30 \
    --output_dir results/my_experiment
```

**Parámetros principales**:
- `--hc_dir`: Directorio con datos Healthy
- `--pd_dir`: Directorio con datos Parkinson
- `--epochs`: Número máximo de épocas (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Tamaño de batch (default: 32)
- `--mc_samples`: Muestras para MC Dropout (default: 30)
- `--patience`: Early stopping patience (default: 10)
- `--output_dir`: Directorio de salida (default: results/cnn_training)

**Tiempo estimado**: ~15-20 minutos (CPU)

---

### 2. `train_cnn_da_kfold.py` - CNN2D_DA con 10-Fold CV

**Pipeline completo según Ibarra et al. (2023) con validación cruzada**

**Flujo**:
```
Datos → Preprocesar → 10-Fold Split por Speaker → 
Para cada Fold: Entrenar CNN2D_DA con GRL → 
Agregar Métricas → Reportar Mean ± Std
```

**Características**:
- Modelo: CNN2D_DA (dual-head con Gradient Reversal Layer)
- Validación: 10-fold cross-validation speaker-independent
- Según paper: SGD, LR=0.1, StepLR scheduler
- Salida: `results/cnn_da_kfold/`

**Uso** (desde la raíz del proyecto):
```bash
# Básico
python pipelines/train_cnn_da_kfold.py

# Personalizado
python pipelines/train_cnn_da_kfold.py \
    --hc_dir data/vowels_healthy \
    --pd_dir data/vowels_pk \
    --n_folds 10 \
    --epochs 100 \
    --lr 0.1 \
    --alpha 1.0 \
    --lambda_grl 1.0 \
    --output_dir results/my_kfold_experiment
```

**Parámetros principales**:
- `--hc_dir`: Directorio con datos Healthy
- `--pd_dir`: Directorio con datos Parkinson
- `--n_folds`: Número de folds (default: 10)
- `--epochs`: Número máximo de épocas por fold (default: 100)
- `--lr`: Learning rate inicial (default: 0.1, según paper)
- `--alpha`: Peso de pérdida de dominio (default: 1.0)
- `--lambda_grl`: Lambda constante para GRL (default: 1.0)
- `--batch_size`: Tamaño de batch (default: 32)
- `--output_dir`: Directorio de salida (default: results/cnn_da_kfold)

**Tiempo estimado**: ~2-3 horas (CPU, 10 folds × 100 epochs)

---

## 🔄 Pipelines vs Notebooks vs Modules

| Aspecto | Pipelines | Notebooks | Modules |
|---------|-----------|-----------|---------|
| **Ubicación** | `pipelines/` | Raíz del proyecto | `modules/` |
| **Tipo** | Scripts CLI | Jupyter | Librerías Python |
| **Ejecución** | `python pipelines/xxx.py` | Celda por celda | Se importan |
| **Propósito** | Automatización completa | Exploración interactiva | Código reutilizable |
| **Cuándo usar** | Producción, batch, HPC | Desarrollo, análisis | Siempre (base) |
| **Interacción** | No (desatendido) | Sí (paso a paso) | N/A (componentes) |

---

## 💡 ¿Cuál Usar?

### Usa Pipelines cuando:
- ✅ Quieres entrenar sin supervisión (ej. dejar corriendo toda la noche)
- ✅ Necesitas ejecutar múltiples experimentos en batch
- ✅ Estás en un servidor/cluster sin interfaz gráfica
- ✅ Quieres reproducibilidad exacta con parámetros guardados
- ✅ Necesitas el pipeline completo: datos → modelo → resultados → visualizaciones

### Usa Notebooks cuando:
- ✅ Estás explorando datos o probando ideas
- ✅ Quieres ver resultados intermedios paso a paso
- ✅ Estás haciendo debugging
- ✅ Necesitas presentar resultados de forma visual
- ✅ Quieres modificar código y re-ejecutar rápidamente

---

## 📊 Ejemplos de Uso

### Experimento Rápido (Baseline)
```bash
# Entrenar CNN2D baseline con configuración por defecto
python pipelines/train_cnn.py

# Ver resultados
ls results/cnn_training/
```

### Experimento con Diferentes Hiperparámetros
```bash
# Probar diferentes learning rates
python pipelines/train_cnn.py --lr 0.001 --output_dir results/lr_001
python pipelines/train_cnn.py --lr 0.01  --output_dir results/lr_01
python pipelines/train_cnn.py --lr 0.1   --output_dir results/lr_1

# Comparar resultados
cat results/lr_*/test_metrics.json
```

### Validación Cruzada Completa (Paper)
```bash
# 10-fold CV según Ibarra et al. (2023)
python pipelines/train_cnn_da_kfold.py \
    --n_folds 10 \
    --epochs 100 \
    --lr 0.1 \
    --output_dir results/paper_replication

# Ver métricas agregadas
cat results/paper_replication/kfold_results.json
```

### Batch de Experimentos
```bash
# Probar múltiples configuraciones
for dropout in 0.3 0.5 0.7; do
    python pipelines/train_cnn.py \
        --dropout_conv $dropout \
        --dropout_fc $dropout \
        --output_dir results/dropout_$dropout
done
```

---

## 🔍 Estructura de Salida

### `train_cnn.py` genera:
```
results/cnn_training/
├── config.json                     # Configuración usada
├── best_model.pth                  # Mejor modelo entrenado
├── training_history.json           # Historial de entrenamiento
├── test_metrics.json               # Métricas en test set
├── mc_dropout_results.npz          # Resultados MC Dropout
├── file_level_results.npz          # Agregación por archivo
├── patient_level_results.npz       # Agregación por paciente
├── uncertainty_analysis.json       # Análisis de incertidumbre
└── visualizations/                 # Gráficas
    ├── training_history.png
    ├── confusion_matrix.png
    ├── uncertainty_distribution.png
    └── ...
```

### `train_cnn_da_kfold.py` genera:
```
results/cnn_da_kfold/
├── config.json                     # Configuración usada
├── kfold_results.json              # Métricas agregadas (mean ± std)
├── fold_1/
│   ├── best_model_da.pth
│   └── training_history.json
├── fold_2/
│   └── ...
└── fold_10/
    └── ...
```

---

## 📚 Más Información

- **Modules**: Ver `modules/README.md` para componentes individuales
- **Notebooks**: Ver notebooks en raíz para exploración interactiva
- **Documentación general**: Ver `README.md` en raíz del proyecto

---

## 🆘 Ayuda

Para ver todos los parámetros disponibles:
```bash
python pipelines/train_cnn.py --help
python pipelines/train_cnn_da_kfold.py --help
```

Para reportar problemas o sugerencias, consulta la documentación principal del proyecto.

