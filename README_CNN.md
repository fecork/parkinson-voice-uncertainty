# CNN 2D para Detección de Parkinson

Pipeline completo de CNN 2D con MC Dropout y Grad-CAM siguiendo el estilo Ibarra/MARTA.

## 🎯 Características

- ✅ **Reutiliza pipeline existente**: No reprocesa, usa espectrogramas ya calculados
- ✅ **Split speaker-independent**: Evita data leakage
- ✅ **SpecAugment on-the-fly**: Solo durante entrenamiento
- ✅ **MC Dropout**: Inferencia con incertidumbre (30 muestras)
- ✅ **Grad-CAM**: Mapas de activación para explicabilidad
- ✅ **Agregación multinivel**: Segmento → Archivo → Paciente
- ✅ **Class weights**: Balanceo automático HC/PD
- ✅ **Early stopping**: Detiene cuando no mejora

## 📦 Módulos Creados

### `modules/cnn_utils.py`
Utilidades que extienden el pipeline existente:
- `SpecAugment`: Transform on-the-fly
- `split_by_speaker()`: Split speaker-independent
- `create_dataloaders_from_existing()`: Crea loaders desde dataset existente
- `compute_class_weights_from_dataset()`: Calcula pesos para loss

### `modules/cnn_model.py`
Modelo y técnicas de incertidumbre/explicabilidad:
- `CNN2D`: Red compacta (32→64 filtros, dropout 0.3/0.5)
- `mc_dropout_predict()`: Inferencia con MC Dropout
- `GradCAM`: Generación de mapas de atención
- `get_last_conv_layer()`: Helper para Grad-CAM

### `modules/cnn_training.py`
Pipeline de entrenamiento:
- `EarlyStopping`: Detección de convergencia
- `train_one_epoch()`: Entrenamiento por época
- `evaluate()`: Evaluación sin dropout
- `train_model()`: Pipeline completo con checkpoints
- `detailed_evaluation()`: Métricas y matriz de confusión

### `modules/cnn_inference.py`
Inferencia con incertidumbre:
- `mc_dropout_inference()`: MC Dropout sobre dataset completo
- `aggregate_by_file()`: Agrega segmentos por archivo
- `aggregate_by_patient()`: Agrega por paciente (speaker)
- `analyze_uncertainty()`: Estadísticas de entropía/varianza
- `find_interesting_cases()`: Casos para análisis (aciertos/errores con alta/baja confianza)

### `modules/cnn_visualization.py`
Visualizaciones:
- `visualize_gradcam()`: Visualiza Grad-CAM individual
- `visualize_multiple_gradcam()`: Múltiples casos
- `visualize_interesting_cases_with_gradcam()`: Casos interesantes
- `plot_uncertainty_distribution()`: Distribución de incertidumbre
- `plot_aggregated_results()`: Resultados por archivo
- `plot_training_history()`: Curvas de entrenamiento
- `generate_visual_report()`: Reporte completo

## 🚀 Uso Rápido

```bash
# Entrenamiento básico
python train_cnn.py

# Con configuración personalizada
python train_cnn.py \
  --hc_dir data/vowels_healthy \
  --pd_dir data/vowels_pk \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --mc_samples 30 \
  --use_class_weights \
  --output_dir results/cnn_experiment_1
```

## ⚙️ Parámetros Principales

### Datos
- `--hc_dir`: Directorio con audios HC (default: `data/vowels_healthy`)
- `--pd_dir`: Directorio con audios PD (default: `data/vowels_pk`)
- `--train_ratio`: Proporción train (default: 0.6)
- `--val_ratio`: Proporción val (default: 0.15)
- `--test_ratio`: Proporción test (default: 0.25)

### Modelo
- `--dropout_conv`: Dropout convolucional (default: 0.3)
- `--dropout_fc`: Dropout FC (default: 0.5)

### SpecAugment
- `--freq_mask`: Frequency masking (default: 8 bins)
- `--time_mask`: Time masking (default: 6 frames)
- `--spec_augment_prob`: Probabilidad (default: 0.5)

### Entrenamiento
- `--batch_size`: Tamaño de batch (default: 32)
- `--epochs`: Épocas máximas (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--patience`: Early stopping (default: 10)
- `--use_class_weights`: Flag para usar class weights

### MC Dropout
- `--mc_samples`: Muestras estocásticas (default: 30)

## 📊 Output

El script genera en `output_dir`:

```
results/cnn_training/
├── config.json                    # Configuración del experimento
├── best_model.pth                 # Checkpoint del mejor modelo
├── training_history.json          # Métricas por época
├── mc_dropout_results.npz         # Resultados MC Dropout (nivel segmento)
├── file_level_results.npz         # Resultados agregados por archivo
├── patient_level_results.npz      # Resultados agregados por paciente
├── uncertainty_analysis.json      # Análisis de incertidumbre
└── visualizations/
    ├── training_history.png       # Curvas loss/acc/F1
    ├── uncertainty_distribution.png  # Distribución de incertidumbre
    ├── aggregated_results.png     # Resultados por archivo
    └── gradcam/
        ├── gradcam_correct_confident.png
        ├── gradcam_incorrect_uncertain.png
        └── gradcam_incorrect_confident.png
```

## 🔬 Pipeline Detallado

### 1. Carga de Datos (Reutiliza preprocesamiento existente)
```python
hc_result = build_full_pipeline(hc_files)  # modules/dataset.py
pd_result = build_full_pipeline(pd_files)
combined_dataset = ConcatDataset([hc_result['torch_ds'], pd_result['torch_ds']])
```
**No reprocesa**: Usa `preprocessing.preprocess_audio_paper()` que ya genera espectrogramas (65×41).

### 2. Split Speaker-Independent
```python
split_indices = split_by_speaker(all_metas, train_ratio=0.6, val_ratio=0.15)
```
Agrupa segmentos por `subject_id` y divide por speaker, no por archivo individual.

### 3. DataLoaders con SpecAugment
```python
loaders = create_dataloaders_from_existing(
    combined_dataset,
    split_indices,
    spec_augment_params={'freq_mask_param': 8, 'time_mask_param': 6, 'prob': 0.5}
)
```
SpecAugment se aplica **solo en train** como transform on-the-fly.

### 4. Entrenamiento
```python
training_results = train_model(
    model, train_loader, val_loader,
    optimizer, criterion, device,
    n_epochs=100, early_stopping_patience=10
)
```
Con early stopping por `val_loss`.

### 5. MC Dropout Inference
```python
mc_results = mc_dropout_inference(model, test_loader, device, n_samples=30)
# mc_results contiene: predictions, probabilities_mean, probabilities_std, entropy, variance
```

### 6. Agregación Multinivel
```python
file_results = aggregate_by_file(mc_results, aggregation_method='mean')
patient_results = aggregate_by_patient(mc_results, aggregation_method='mean')
```

### 7. Visualizaciones
```python
generate_visual_report(model, test_loader, mc_results, file_results, history, save_dir)
```
Genera todas las gráficas incluyendo Grad-CAM de casos interesantes.

## 🧪 Ejemplo de Uso en Notebook

```python
# En tu notebook después de tener los espectrogramas
from modules.dataset import build_full_pipeline
from modules.cnn_utils import split_by_speaker, create_dataloaders_from_existing
from modules.cnn_model import CNN2D
from modules.cnn_training import train_model
from modules.cnn_inference import mc_dropout_inference, aggregate_by_file
import torch

# 1. Cargar datos (reutiliza pipeline existente)
hc_result = build_full_pipeline(audio_files_hc)
pd_result = build_full_pipeline(audio_files_pd)

# 2. Combinar
from torch.utils.data import ConcatDataset
combined = ConcatDataset([hc_result['torch_ds'], pd_result['torch_ds']])
all_metas = hc_result['metadata'] + pd_result['metadata']

# 3. Split speaker-independent
splits = split_by_speaker(all_metas, seed=42)

# 4. DataLoaders con SpecAugment
loaders = create_dataloaders_from_existing(combined, splits, batch_size=32)

# 5. Modelo
model = CNN2D(n_classes=2, p_drop_conv=0.3, p_drop_fc=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 6. Entrenar
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

results = train_model(
    model, loaders['train'], loaders['val'],
    optimizer, criterion, device,
    n_epochs=50, early_stopping_patience=10
)

# 7. MC Dropout en test
mc_results = mc_dropout_inference(model, loaders['test'], device, n_samples=30)

# 8. Agregación
file_results = aggregate_by_file(mc_results)

# 9. Métricas
from sklearn.metrics import classification_report
print(classification_report(
    file_results['file_labels'],
    file_results['file_predictions'],
    target_names=['HC', 'PD']
))
```

## 📋 Checklist Papers Ibarra/MARTA

- ✅ Segmentos 400ms con 50% overlap
- ✅ Mel-spectrograms 65 bins × 41 frames
- ✅ Ventana FFT 40ms para vocales (10ms hop)
- ✅ Z-score normalización por espectrograma
- ✅ SpecAugment (freq mask + time mask)
- ✅ Split speaker-independent
- ✅ Agregación por archivo/paciente
- ✅ MC Dropout para incertidumbre
- ✅ Grad-CAM para explicabilidad

## 🔧 Troubleshooting

### Error: "No module named 'modules.cnn_dataloader'"
**Solución**: El módulo `cnn_dataloader.py` fue eliminado para evitar duplicación. Ahora usamos `cnn_utils.py` que reutiliza el pipeline existente.

### Advertencia: "Solo 1 speaker PD detectado"
**Causa**: El dataset PD solo tiene 1 speaker (1580).
**Impacto**: Causa data leakage porque el mismo speaker aparece en train/val/test.
**Solución ideal**: Conseguir más speakers PD para split real.
**Workaround actual**: Se usa el mismo speaker en todos los splits pero se reporta la advertencia.

### VRAM insuficiente
**Solución**: Reducir `--batch_size` (probar 16 u 8).

### Entrenamiento lento en CPU
**Solución**: 
- Reducir `--mc_samples` (probar 10-20)
- Usar GPU si está disponible
- Reducir `--epochs`

## 🎓 Referencias

- Ibarra et al.: Domain adaptation para Parkinson
- MARTA: Multi-task learning para speech disorders
- SpecAugment: Park et al., 2019
- MC Dropout: Gal & Ghahramani, 2016
- Grad-CAM: Selvaraju et al., 2017

