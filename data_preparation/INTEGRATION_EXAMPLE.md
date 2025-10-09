# 📝 Integración con Jupyter Notebook

## 🎯 Workflow Completo: Parkinson + Healthy

### Paso 1: Ejecutar Muestreo de Datos Saludables

```bash
# En terminal
python sample_healthy_data.py --target-spectrograms 1219

# Verificar resultado
python verify_sampling.py
```

### Paso 2: Modificar Notebook para Procesar Ambos Datasets

#### A. Procesar Dataset Parkinson (Existente)

```python
# Celda: Data Loading - Parkinson
DATA_PATH_PARKINSON = "./vowels"

if os.path.exists(DATA_PATH_PARKINSON):
    audio_files_parkinson = list(Path(DATA_PATH_PARKINSON).glob("*.nsp"))
    print(f"✅ Archivos Parkinson (.nsp): {len(audio_files_parkinson)}")
```

```python
# Celda: Dataset Generation - Parkinson
results_parkinson = build_full_pipeline(
    audio_files=audio_files_parkinson,
    preprocess_fn=None,
    max_files=None
)

dataset_parkinson = results_parkinson["dataset"]
(X_parkinson, y_task_parkinson, y_domain_parkinson) = results_parkinson["tensors"]

print(f"✅ Parkinson: {len(dataset_parkinson)} muestras base")
print(f"   Shape: {X_parkinson.shape}")
```

```python
# Celda: Augmentation - Parkinson
augmented_dataset_parkinson = create_augmented_dataset(
    audio_files_parkinson,
    augmentation_types=["original", "pitch_shift", "time_stretch", "noise"],
    apply_spec_augment=True,
    num_spec_augment_versions=2,
    use_cache=True,
    cache_dir="./cache/parkinson"
)

X_aug_pk, y_task_aug_pk, y_domain_aug_pk, meta_aug_pk = to_pytorch_tensors(
    augmented_dataset_parkinson
)

print(f"✅ Parkinson con augmentation: {len(augmented_dataset_parkinson)} muestras")
print(f"   Shape: {X_aug_pk.shape}")
```

#### B. Procesar Dataset Healthy (Nuevo)

```python
# Celda: Data Loading - Healthy
DATA_PATH_HEALTHY = "./vowels_healthy"

if os.path.exists(DATA_PATH_HEALTHY):
    audio_files_healthy = list(Path(DATA_PATH_HEALTHY).glob("*.nsp"))
    print(f"✅ Archivos Healthy (.nsp): {len(audio_files_healthy)}")
```

```python
# Celda: Dataset Generation - Healthy
results_healthy = build_full_pipeline(
    audio_files=audio_files_healthy,
    preprocess_fn=None,
    max_files=None
)

dataset_healthy = results_healthy["dataset"]
(X_healthy, y_task_healthy, y_domain_healthy) = results_healthy["tensors"]

print(f"✅ Healthy: {len(dataset_healthy)} muestras base")
print(f"   Shape: {X_healthy.shape}")
```

```python
# Celda: Augmentation - Healthy
augmented_dataset_healthy = create_augmented_dataset(
    audio_files_healthy,
    augmentation_types=["original", "pitch_shift", "time_stretch", "noise"],
    apply_spec_augment=True,
    num_spec_augment_versions=2,
    use_cache=True,
    cache_dir="./cache/healthy"
)

X_aug_hl, y_task_aug_hl, y_domain_aug_hl, meta_aug_hl = to_pytorch_tensors(
    augmented_dataset_healthy
)

print(f"✅ Healthy con augmentation: {len(augmented_dataset_healthy)} muestras")
print(f"   Shape: {X_aug_hl.shape}")
```

#### C. Combinar Datasets

```python
# Celda: Combinar Parkinson + Healthy
import torch

# Combinar espectrogramas
X_combined = torch.cat([X_aug_pk, X_aug_hl], dim=0)

# Crear labels: 0=Control (Healthy), 1=Parkinson
y_task_combined = torch.cat([
    torch.ones(len(X_aug_pk)),   # Parkinson = 1
    torch.zeros(len(X_aug_hl))   # Healthy = 0
], dim=0)

# Combinar domains (ajustar IDs para evitar colisiones)
max_domain_pk = y_domain_aug_pk.max().item()
y_domain_combined = torch.cat([
    y_domain_aug_pk,
    y_domain_aug_hl + max_domain_pk + 1  # Offset para healthy
], dim=0)

print("="*70)
print("📊 DATASET COMBINADO:")
print("="*70)
print(f"   • Total muestras: {len(X_combined)}")
print(f"   • Shape: {X_combined.shape}")
print(f"   • Parkinson: {(y_task_combined == 1).sum().item()}")
print(f"   • Healthy: {(y_task_combined == 0).sum().item()}")
print(f"   • Dominios únicos: {len(torch.unique(y_domain_combined))}")
print(f"   • Balance: {(y_task_combined == 1).sum() / len(y_task_combined) * 100:.1f}% Parkinson")
print("="*70)
```

#### D. Visualizar Balance

```python
# Celda: Visualización de Balance
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Distribución de clases
class_counts = {
    'Parkinson': (y_task_combined == 1).sum().item(),
    'Healthy': (y_task_combined == 0).sum().item()
}

axes[0].bar(class_counts.keys(), class_counts.values(), color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Distribución de Clases', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Número de Muestras')
axes[0].grid(axis='y', alpha=0.3)

for i, (k, v) in enumerate(class_counts.items()):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Gráfico 2: Distribución de dominios
domain_counts_pk = torch.bincount(y_domain_aug_pk)
domain_counts_hl = torch.bincount(y_domain_aug_hl)

axes[1].bar(
    range(len(domain_counts_pk)), 
    domain_counts_pk.numpy(), 
    color='#e74c3c', 
    alpha=0.7, 
    label='Parkinson'
)
axes[1].bar(
    range(len(domain_counts_pk), len(domain_counts_pk) + len(domain_counts_hl)), 
    domain_counts_hl.numpy(), 
    color='#2ecc71', 
    alpha=0.7, 
    label='Healthy'
)
axes[1].set_title('Distribución de Dominios', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Domain ID')
axes[1].set_ylabel('Muestras por Dominio')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Estadísticas adicionales
print("\n📊 ESTADÍSTICAS:")
print(f"   • Ratio Parkinson/Healthy: {class_counts['Parkinson'] / class_counts['Healthy']:.2f}")
print(f"   • Balance ideal (50/50): {abs(0.5 - class_counts['Parkinson'] / len(y_task_combined)) * 100:.1f}% desviación")
```

#### E. Split Train/Val/Test

```python
# Celda: Split estratificado
from sklearn.model_selection import train_test_split

# Convertir a numpy para sklearn
X_np = X_combined.numpy()
y_task_np = y_task_combined.numpy()
y_domain_np = y_domain_combined.numpy()

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp, d_train, d_temp = train_test_split(
    X_np, y_task_np, y_domain_np,
    test_size=0.3,
    stratify=y_task_np,
    random_state=42
)

X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
    X_temp, y_temp, d_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# Convertir de vuelta a tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

print("="*70)
print("📊 SPLITS DEL DATASET:")
print("="*70)
print(f"   • Train: {len(X_train)} muestras ({len(X_train)/len(X_combined)*100:.1f}%)")
print(f"      - Parkinson: {(y_train == 1).sum().item()}")
print(f"      - Healthy: {(y_train == 0).sum().item()}")
print(f"\n   • Validation: {len(X_val)} muestras ({len(X_val)/len(X_combined)*100:.1f}%)")
print(f"      - Parkinson: {(y_val == 1).sum().item()}")
print(f"      - Healthy: {(y_val == 0).sum().item()}")
print(f"\n   • Test: {len(X_test)} muestras ({len(X_test)/len(X_combined)*100:.1f}%)")
print(f"      - Parkinson: {(y_test == 1).sum().item()}")
print(f"      - Healthy: {(y_test == 0).sum().item()}")
print("="*70)
```

#### F. Crear DataLoaders

```python
# Celda: PyTorch DataLoaders
from torch.utils.data import TensorDataset, DataLoader

# Crear TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Crear DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("✅ DataLoaders creados:")
print(f"   • Train: {len(train_loader)} batches")
print(f"   • Val: {len(val_loader)} batches")
print(f"   • Test: {len(test_loader)} batches")
print(f"   • Batch size: {BATCH_SIZE}")

# Verificar un batch
X_batch, y_batch = next(iter(train_loader))
print(f"\n🔍 Batch de ejemplo:")
print(f"   • X shape: {X_batch.shape}")
print(f"   • y shape: {y_batch.shape}")
print(f"   • Device: {X_batch.device}")
```

---

## 📋 Resumen del Pipeline Completo

```
1. Terminal: Muestreo de datos
   └─> python sample_healthy_data.py --target-spectrograms 1219
   └─> python verify_sampling.py

2. Notebook: Procesar Parkinson
   └─> Cargar audios ./vowels/*.nsp
   └─> Pipeline + Augmentation
   └─> Resultado: X_aug_pk (1219 muestras)

3. Notebook: Procesar Healthy
   └─> Cargar audios ./vowels_healthy/*.nsp
   └─> Pipeline + Augmentation
   └─> Resultado: X_aug_hl (1313 muestras)

4. Notebook: Combinar y Balancear
   └─> Concatenar datasets
   └─> Crear labels (0=Healthy, 1=Parkinson)
   └─> Split train/val/test estratificado
   └─> DataLoaders de PyTorch

5. Notebook: Entrenar Modelo
   └─> CNN con Domain Adaptation
   └─> Evaluación con métricas
   └─> Análisis de incertidumbre
```

---

## 💡 Tips de Integración

### Cache Separado
```python
# Usar carpetas de cache diferentes para evitar conflictos
cache_parkinson = "./cache/parkinson"
cache_healthy = "./cache/healthy"
```

### Reproducibilidad
```python
# Fijar seeds en todas partes
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### Monitoreo de Balance
```python
# Verificar balance en cada epoch
def check_balance(y_pred, y_true):
    correct_pk = ((y_pred == 1) & (y_true == 1)).sum()
    correct_hl = ((y_pred == 0) & (y_true == 0)).sum()
    total_pk = (y_true == 1).sum()
    total_hl = (y_true == 0).sum()
    
    print(f"   Parkinson: {correct_pk}/{total_pk} ({correct_pk/total_pk*100:.1f}%)")
    print(f"   Healthy: {correct_hl}/{total_hl} ({correct_hl/total_hl*100:.1f}%)")
```

---

## 🎯 Resultado Final Esperado

```
Dataset Balanceado:
├─ Parkinson: ~1,219 muestras (48%)
├─ Healthy:   ~1,313 muestras (52%)
└─ Total:     ~2,532 muestras

Split Estratificado:
├─ Train:      1,772 (70%)
├─ Validation:   380 (15%)  
└─ Test:         380 (15%)

Listo para:
✅ Entrenar CNN
✅ Domain Adaptation
✅ Análisis de Incertidumbre
```

