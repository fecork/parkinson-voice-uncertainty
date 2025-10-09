# 🎯 Muestreo de Datos Saludables

Script para seleccionar una muestra aleatoria de sujetos saludables y preparar sus datos para balancear el dataset de Parkinson.

## 📋 Descripción

Este script toma una muestra aleatoria de sujetos del directorio `data/healthy/` y copia sus archivos de vocales (`.nsp` o `.egg`) a un directorio de salida para su posterior procesamiento con augmentation.

## 🚀 Uso Rápido

### 1. Ver sujetos disponibles
```bash
python sample_healthy_data.py --list-subjects
```

### 2. Simular muestreo (sin copiar archivos)
```bash
python sample_healthy_data.py --dry-run --target-spectrograms 1219
```

### 3. Ejecutar muestreo real
```bash
python sample_healthy_data.py --target-spectrograms 1219
```

## 📊 Resultados Esperados

Con la configuración default:
- **Sujetos disponibles**: 687
- **Sujetos necesarios**: 10 (para ~1219 espectrogramas)
- **Archivos copiados**: 130 (10 sujetos × 13 archivos/sujeto)
- **Espectrogramas finales**: ~1313 (con augmentation 10.1x)

## 🔧 Opciones Disponibles

### Argumentos Principales

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--target-spectrograms` | Número objetivo de espectrogramas | 1219 |
| `--num-subjects` | Número de sujetos a seleccionar | (auto-calculado) |
| `--seed` | Semilla aleatoria para reproducibilidad | 42 |
| `--extension` | Extensión de archivos (`nsp` o `egg`) | `nsp` |
| `--source-dir` | Directorio fuente | `./data/healthy` |
| `--output-dir` | Directorio de salida | `./vowels_healthy` |
| `--dry-run` | Simular sin copiar archivos | False |
| `--list-subjects` | Solo listar sujetos y salir | False |

### Ejemplos de Uso

#### Seleccionar manualmente número de sujetos
```bash
python sample_healthy_data.py --num-subjects 15
```

#### Usar archivos .egg en lugar de .nsp
```bash
python sample_healthy_data.py --extension egg
```

#### Cambiar semilla para obtener muestra diferente
```bash
python sample_healthy_data.py --seed 123
```

#### Cambiar directorios de entrada/salida
```bash
python sample_healthy_data.py \
  --source-dir ./data/healthy \
  --output-dir ./custom_output
```

#### Combinar opciones
```bash
python sample_healthy_data.py \
  --target-spectrograms 2000 \
  --seed 999 \
  --extension nsp \
  --output-dir ./vowels_healthy_large
```

## 📁 Estructura Generada

```
vowels_healthy/
├── 26-a_h.nsp
├── 26-a_l.nsp
├── 26-a_lhl.nsp
├── ...
├── 2248-u_n.nsp
└── sampling_metadata.json
```

### Metadata (sampling_metadata.json)

El script genera un archivo JSON con información de la muestra:
```json
{
  "config": {
    "source_dir": "data/healthy",
    "output_dir": "vowels_healthy",
    "target_spectrograms": 1219,
    "num_subjects": 10,
    "seed": 42,
    "extension": "nsp"
  },
  "subject_ids": [26, 94, 117, 135, 697, 968, 990, 1022, 1865, 2248],
  "num_subjects": 10,
  "stats": {
    "subjects_processed": 10,
    "files_copied": 130,
    "files_skipped": 0,
    "num_errors": 0
  },
  "expected_spectrograms": 1313
}
```

## 🔄 Workflow Completo

### 1. Muestreo de datos (este script)
```bash
# Tomar muestra de 10 sujetos aleatorios
python sample_healthy_data.py --target-spectrograms 1219
```

### 2. Procesamiento en Jupyter Notebook
```python
# En parkinson_voice_analysis.ipynb

# Modificar ruta de datos
DATA_PATH = "./vowels_healthy"  # En lugar de "./vowels"

# Ejecutar las celdas de:
# - Data Loading
# - Preprocessing  
# - Dataset Generation con Augmentation
```

### 3. Verificar balance
```python
# Verificar que tenemos cantidades similares
print(f"Parkinson: {len(X_aug_parkinson)} espectrogramas")
print(f"Healthy: {len(X_aug_healthy)} espectrogramas")
```

## ⚙️ Cálculo de Sujetos Necesarios

El script calcula automáticamente cuántos sujetos se necesitan:

```
espectrogramas_por_sujeto = archivos_por_sujeto × augmentation_factor
                         = 13 × 10.1
                         = 131.3

sujetos_necesarios = target_spectrograms / espectrogramas_por_sujeto
                   = 1219 / 131.3
                   = 9.3 → 10 sujetos
```

## 📊 Consideraciones

### Archivos .nsp vs .egg

- **`.nsp`** (Nasal Speech): Audio capturado por micrófono → **RECOMENDADO**
- **`.egg`** (Electroglotógrafo): Señal de vibraciones de cuerdas vocales

Para detección de Parkinson desde voz, normalmente se usa `.nsp`.

### Reproducibilidad

El parámetro `--seed` garantiza que siempre se seleccionen los mismos sujetos:
- `seed=42`: Sujetos [26, 94, 117, 135, 697, 968, 990, 1022, 1865, 2248]
- `seed=123`: Sujetos diferentes pero siempre los mismos con ese seed

### Augmentation Factor

El factor de 10.1x incluye:
- Original (1x)
- Pitch Shift: ±1, ±2 semitonos (4x)
- Time Stretch: 0.9x, 1.1x (2x)
- Noise (1x)
- SpecAugment: 2 versiones (2x)

Total: 1 + 4 + 2 + 1 + 2 = 10x (con variaciones → 10.1x)

## ❓ Troubleshooting

### Error: "No se encontraron sujetos"
```bash
# Verificar ruta correcta
ls -la data/healthy/

# O especificar ruta absoluta
python sample_healthy_data.py --source-dir /ruta/completa/data/healthy
```

### Error: "No existe el directorio"
```bash
# Crear estructura de directorios
mkdir -p data/healthy
```

### Sujetos insuficientes
```bash
# Ver cuántos hay disponibles
python sample_healthy_data.py --list-subjects

# Ajustar target o usar todos
python sample_healthy_data.py --num-subjects 5
```

## 💡 Tips

1. **Siempre usar `--dry-run` primero** para verificar antes de copiar
2. **Documentar el seed usado** para reproducibilidad
3. **Guardar `sampling_metadata.json`** para referencia futura
4. **Verificar balance** después del augmentation en notebook

## 📝 Notas

- El script es **idempotente**: puede ejecutarse múltiples veces
- Los archivos se **copian** (no se mueven), los originales permanecen intactos
- La metadata incluye la lista exacta de sujetos para trazabilidad
- Tiempo de ejecución: ~1 segundo (solo copia, no procesa)

## 🔗 Próximos Pasos

Después de ejecutar este script:

1. ✅ Archivos copiados a `vowels_healthy/`
2. → Abrir `parkinson_voice_analysis.ipynb`
3. → Cambiar `DATA_PATH` a `"./vowels_healthy"`
4. → Ejecutar pipeline de augmentation
5. → Combinar datasets Parkinson + Healthy
6. → Entrenar modelo con datos balanceados

---

**Autor**: Script de utilidad para PhD Parkinson - Incertidumbre  
**Versión**: 1.0  
**Fecha**: Octubre 2025

