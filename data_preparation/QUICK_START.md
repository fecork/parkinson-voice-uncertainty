# 🚀 Quick Start - Muestreo de Datos Saludables

## ✅ Lo que se ha creado

### 1. Scripts Python
- **`sample_healthy_data.py`**: Script principal para muestreo aleatorio
- **`verify_sampling.py`**: Script de verificación de resultados
- **`README_SAMPLING.md`**: Documentación detallada
- **`INTEGRATION_EXAMPLE.md`**: Ejemplos de integración con notebook

### 2. Datos Generados
- **`vowels_healthy/`**: 130 archivos .nsp de 10 sujetos saludables
- **`sampling_metadata.json`**: Metadata completa del muestreo

---

## 🎯 Resultado Actual

```
✅ MUESTREO COMPLETADO

Datos Parkinson (existente):
├─ Archivos: 13 (.egg en ./vowels/)
└─ Espectrogramas (con augmentation): ~1,219

Datos Healthy (nuevo):
├─ Archivos: 130 (.nsp en ./vowels_healthy/)
├─ Sujetos: 10 (IDs: 26, 94, 117, 135, 697, 968, 990, 1022, 1865, 2248)
└─ Espectrogramas estimados (con augmentation): ~1,313

Balance: 48% Parkinson / 52% Healthy ✅
```

---

## 📝 Nota Importante: .nsp vs .egg

**Se detectó que los archivos originales de Parkinson son `.egg`**

### ¿Qué son cada uno?
- **`.egg`** (Electroglotógrafo): Señal de vibraciones de cuerdas vocales
- **`.nsp`** (Nasal Speech): Audio capturado por micrófono

### ⚠️ Acción Requerida

Necesitas decidir cuál usar:

#### Opción A: Usar `.nsp` en ambos (RECOMENDADO)
```bash
# Ya tienes healthy con .nsp ✅
# Necesitas cambiar Parkinson a .nsp

# En el notebook, cambiar:
audio_files_parkinson = list(Path("./vowels").glob("*.nsp"))
```

#### Opción B: Usar `.egg` en ambos
```bash
# Regenerar healthy con .egg
python sample_healthy_data.py --extension egg --target-spectrograms 1219

# En el notebook:
audio_files_healthy = list(Path("./vowels_healthy").glob("*.egg"))
```

**💡 Recomendación**: Usar `.nsp` (audio de micrófono) porque:
- Es más común para detección de Parkinson desde voz
- Es lo que normalmente capturan aplicaciones móviles
- Tiene más aplicabilidad clínica

---

## 🔄 Próximos Pasos

### 1. Verificar tipo de archivo correcto
```bash
# Ver qué archivos tienes en Parkinson
ls vowels/*.egg vowels/*.nsp

# Si solo tienes .egg, regenerar healthy:
python sample_healthy_data.py --extension egg --target-spectrograms 1219
```

### 2. Abrir notebook y modificar
```python
# Celda de configuración
DATA_PATH_PARKINSON = "./vowels"
DATA_PATH_HEALTHY = "./vowels_healthy"

# Decidir extensión consistente
AUDIO_EXTENSION = "nsp"  # o "egg"

# Cargar archivos
audio_files_pk = list(Path(DATA_PATH_PARKINSON).glob(f"*.{AUDIO_EXTENSION}"))
audio_files_hl = list(Path(DATA_PATH_HEALTHY).glob(f"*.{AUDIO_EXTENSION}"))
```

### 3. Procesar ambos datasets
Ver ejemplos completos en `INTEGRATION_EXAMPLE.md`

---

## 📊 Comandos Útiles

### Regenerar con diferentes parámetros
```bash
# Más sujetos (más datos)
python sample_healthy_data.py --num-subjects 15

# Diferentes sujetos (cambiar seed)
python sample_healthy_data.py --seed 123

# Usar .egg en lugar de .nsp
python sample_healthy_data.py --extension egg
```

### Verificar resultados
```bash
# Verificar muestreo
python verify_sampling.py

# Contar archivos
ls vowels_healthy/*.nsp | wc -l

# Ver metadata
cat vowels_healthy/sampling_metadata.json
```

### Limpiar y reiniciar
```bash
# Eliminar archivos generados
rm -rf vowels_healthy/

# Regenerar desde cero
python sample_healthy_data.py --target-spectrograms 1219
```

---

## 🐛 Troubleshooting

### Los archivos de Parkinson son .egg pero healthy es .nsp
```bash
# Solución 1: Regenerar healthy con .egg
rm -rf vowels_healthy/
python sample_healthy_data.py --extension egg --target-spectrograms 1219

# Solución 2: Cambiar Parkinson a .nsp (si existen)
# Verificar primero si existen archivos .nsp en ./vowels/
```

### No se encuentran archivos .nsp en ./vowels/
```bash
# Los archivos de Parkinson son .egg
# Usa .egg para ambos:
python sample_healthy_data.py --extension egg --target-spectrograms 1219
```

### Quiero diferentes sujetos
```bash
# Cambiar seed
python sample_healthy_data.py --seed 999
```

---

## 📚 Documentación Completa

- **README_SAMPLING.md**: Guía detallada del script
- **INTEGRATION_EXAMPLE.md**: Código completo para notebook
- **sampling_metadata.json**: Metadata del muestreo actual

---

## ✨ Resumen

```
✅ Scripts creados y funcionando
✅ 130 archivos .nsp copiados (10 sujetos)
✅ Balance ~50/50 con Parkinson
⚠️  VERIFICAR: Usar mismo tipo de archivo (.nsp o .egg) en ambos datasets
→  Siguiente: Integrar en notebook y entrenar modelo
```

**Seed usado**: 42 (reproducible)
**Fecha**: Octubre 2025

