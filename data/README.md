# 📊 Datos de Audio - Vocales de Parkinson

Esta carpeta contiene los archivos de audio raw utilizados para el entrenamiento de los modelos.

## 📁 Estructura

```
data/
├── vowels_healthy/          ← Sujetos sanos
│   ├── *.egg               ← Archivos de audio
│   └── healthy_diverse_metadata.json
│
└── vowels_pk/               ← Pacientes con Parkinson
    ├── *.egg               ← Archivos de audio
    └── *.nsp               ← Metadatos asociados
```

## 🎵 Formato de Datos

### Archivos .egg
- **Formato**: Señales de audio de vocales sostenidas
- **Vocales**: a, i, u, iau (diptongo)
- **Condiciones**: 
  - `_h`: high pitch (tono alto)
  - `_l`: low pitch (tono bajo)
  - `_n`: normal pitch (tono normal)
  - `_lhl`: low-high-low (variación tonal)

### Nomenclatura
Ejemplo: `1022-a_lhl-egg.egg`
- `1022`: ID del sujeto
- `a`: Vocal pronunciada
- `lhl`: Condición tonal
- `egg`: Extensión del archivo

## 📈 Dataset

### Healthy (vowels_healthy/)
- **Sujetos**: 13 archivos
- **Origen**: Sujetos sanos sin diagnóstico de Parkinson

### Parkinson (vowels_pk/)
- **Sujetos**: 13 archivos
- **Origen**: Pacientes diagnosticados con Parkinson

### Balance
- Ratio: 1:1 (perfecto balance)
- Total archivos raw: 26

## 🔄 Procesamiento

Estos archivos raw son procesados por `data_preprocessing.ipynb`:

1. **Carga**: Lee archivos .egg
2. **Preprocesamiento**: 
   - Resampling a 16kHz
   - Segmentación en ventanas de 100ms
   - Generación de Mel spectrograms (65 bandas)
3. **Augmentation**:
   - Pitch shift
   - Time stretch
   - Noise injection
   - SpecAugment
4. **Cache**: Guarda en `cache/healthy/` y `cache/parkinson/`

### Output del Procesamiento
- **Healthy**: ~1553 espectrogramas augmentados
- **Parkinson**: ~1219 espectrogramas augmentados
- **Factor de multiplicación**: ~119x (de 26 → 2772 muestras)

## 📝 Notas Importantes

### ⚠️ No Modificar
Los archivos en esta carpeta son datos raw originales. No modificar o eliminar.

### 📥 Agregar Más Datos
Para agregar nuevos datos:
1. Colocar archivos .egg en la carpeta correspondiente
2. Seguir nomenclatura: `{ID}-{vocal}_{condicion}-egg.egg`
3. Re-ejecutar `data_preprocessing.ipynb` con `FORCE_REGENERATE = True`

### 🔍 Verificación
Para verificar la integridad de los datos:
```bash
cd data_preparation
python verify_sampling.py
```

## 🎯 Uso

### Ver Metadatos
```python
import json

with open('data/vowels_healthy/healthy_diverse_metadata.json') as f:
    metadata = json.load(f)
    print(metadata)
```

### Cargar Audio Raw
```python
import librosa

audio, sr = librosa.load('data/vowels_healthy/1022-a_lhl-egg.egg', sr=None)
print(f"Duración: {len(audio)/sr:.2f}s")
print(f"Sample rate: {sr} Hz")
```

## 📚 Referencias

Ver `data_preparation/` para:
- `INTEGRATION_EXAMPLE.md`: Ejemplo de integración
- `QUICK_START.md`: Guía rápida
- `sample_healthy_data.py`: Script de ejemplo
- `verify_sampling.py`: Verificación de datos

---

**Última actualización**: 2025-10-17

