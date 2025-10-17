# 🧪 Tests del Proyecto

Esta carpeta contiene scripts de prueba y validación del proyecto.

## 📁 Archivos de Test

```
test/
├── test_grl_completo.py           ← Test de GRL (Gradient Reversal Layer)
├── test_ibarra_implementation.py  ← Test implementación Ibarra (2023)
├── verify_labels.py               ← Verificación de labels
└── Feature_Extractor_2D_CNN_Visualization.ipynb  ← Visualización features
```

---

## 🔬 Tests Disponibles

### `test_grl_completo.py`

**Propósito**: Verificar implementación correcta del Gradient Reversal Layer

**Ejecutar**:
```bash
cd test
python test_grl_completo.py
```

**Verifica**:
- ✓ Forward pass de GRL
- ✓ Backward pass con reversión de gradiente
- ✓ Integración con CNN2D_DA

**Output esperado**:
```
✓ GRL forward pass: OK
✓ GRL backward pass: OK
✓ Gradient reversal: OK
✓ Integration with CNN2D_DA: OK
```

---

### `test_ibarra_implementation.py`

**Propósito**: Verificar que la implementación sigue el paper Ibarra et al. (2023)

**Ejecutar**:
```bash
cd test
python test_ibarra_implementation.py
```

**Verifica**:
- ✓ Arquitectura CNN2D_DA
- ✓ Parámetros de entrenamiento
- ✓ Loss functions multi-task
- ✓ Lambda scheduling

**Output esperado**:
```
✓ Architecture: OK
✓ Training config: OK
✓ Loss functions: OK
✓ Lambda scheduler: OK
```

---

### `verify_labels.py`

**Propósito**: Verificar correctitud de labels y metadata

**Ejecutar**:
```bash
cd test
python verify_labels.py
```

**Verifica**:
- ✓ Labels de tarea (Healthy/Parkinson)
- ✓ Labels de dominio (Subject ID)
- ✓ Consistencia de metadata

---

### `Feature_Extractor_2D_CNN_Visualization.ipynb`

**Propósito**: Visualizar features extraídas por la CNN

**Ejecutar**:
```bash
jupyter notebook test/Feature_Extractor_2D_CNN_Visualization.ipynb
```

**Contenido**:
- Visualización de activaciones de capas
- Feature maps de convoluciones
- Análisis de representaciones aprendidas

---

## 🎯 Ejecutar Todos los Tests

### Script Completo

```bash
cd test

# Test GRL
python test_grl_completo.py

# Test implementación Ibarra
python test_ibarra_implementation.py

# Verificar labels
python verify_labels.py

echo "✅ Todos los tests completados"
```

---

## 📊 Interpretación de Resultados

### ✅ Test Exitoso
```
✓ All tests passed
✓ Implementation correct
```
→ Todo funcionando correctamente

### ❌ Test Fallido
```
✗ Test failed: [descripción del error]
```
→ Revisar implementación según mensaje de error

---

## 🔧 Agregar Nuevos Tests

### Template de Test

```python
# test/test_nueva_funcionalidad.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.nueva_funcionalidad import funcion_a_testear

def test_funcion():
    """Test de nueva funcionalidad"""
    # Setup
    input_data = ...
    expected_output = ...
    
    # Ejecutar
    result = funcion_a_testear(input_data)
    
    # Verificar
    assert result == expected_output, f"Expected {expected_output}, got {result}"
    print("✓ Test passed")

if __name__ == "__main__":
    test_funcion()
    print("✅ All tests completed")
```

---

## 📝 Buenas Prácticas

### ✅ Hacer
1. Ejecutar tests antes de hacer cambios importantes
2. Agregar tests para nuevas funcionalidades
3. Documentar qué verifica cada test

### ❌ Evitar
1. Modificar código de producción sin ejecutar tests
2. Ignorar tests fallidos
3. Tests sin documentación

---

## 🎓 Tipos de Tests

### Unit Tests
Verifican funciones individuales en aislamiento

### Integration Tests
Verifican interacción entre módulos

### Validation Tests
Verifican correctitud de datos y labels

---

## 💡 Debugging

### Si un Test Falla

1. **Leer mensaje de error**
   ```
   ✗ Test failed: Expected X, got Y
   ```

2. **Verificar implementación**
   - Revisar código en `modules/`
   - Comparar con paper/especificación

3. **Ejecutar con verbose**
   ```bash
   python test/test_grl_completo.py --verbose
   ```

4. **Revisar logs**
   - Imprimir valores intermedios
   - Verificar shapes de tensores

---

**Última actualización**: 2025-10-17

