# Pruebas Unitarias del Proyecto

Este directorio contiene las pruebas unitarias para verificar la implementación correcta de los modelos.

## 📋 Pruebas Disponibles

### `test_lstm_da_implementation.py` (NUEVO)

Pruebas completas para el modelo Time-CNN-BiLSTM-DA.

**Ejecutar:**
```bash
python test/test_lstm_da_implementation.py
```

**Cobertura (14 tests):**

#### 1. TestSequenceGeneration (3 tests)
- ✅ Verifica forma correcta de secuencias `(n_frames, 1, 65, 41)`
- ✅ Verifica que padding contiene ceros
- ✅ Verifica filtrado de secuencias con muy pocos frames

#### 2. TestLSTMModel (4 tests)
- ✅ Verifica dimensiones de outputs del modelo
- ✅ Verifica forward con masking (secuencias de longitud variable)
- ✅ Verifica retorno de embeddings para visualización
- ✅ Verifica que forward es determinístico en modo eval

#### 3. TestGradientReversal (3 tests)
- ✅ Verifica que GRL no modifica valores en forward
- ✅ Verifica que GRL invierte gradientes en backward
- ✅ Verifica actualización de lambda

#### 4. TestBiLSTMTemporal (2 tests)
- ✅ Verifica procesamiento correcto de secuencias temporales
- ✅ Verifica efecto del masking en resultados

#### 5. TestIntegration (1 test)
- ✅ Pipeline completo: dataset → secuencias → predicciones

#### 6. TestParameterCount (1 test)
- ✅ Verifica número razonable de parámetros (~1.67M)

**Resultado:** ✅ 14/14 tests pasando

## 🔧 Otras Pruebas del Proyecto

### `test_uncertainty_math.py`
Verifica la implementación matemática del modelo de incertidumbre.

### `test_grl_completo.py`
Verifica la implementación del Gradient Reversal Layer.

### `test_yarin_gal_implementation.py`
Verifica implementación de MC Dropout según Yarin Gal.

### `test_cnn1d_implementation.py`
Verifica implementación del modelo CNN1D-DA.

## 📊 Cómo Ejecutar Todas las Pruebas

```bash
# Ejecutar todas las pruebas del proyecto
python -m unittest discover test

# Ejecutar una prueba específica
python test/test_lstm_da_implementation.py

# Ejecutar con verbose
python test/test_lstm_da_implementation.py -v
```

## ✅ Validaciones Realizadas

Las pruebas verifican:

1. **Arquitectura Correcta**
   - Dimensiones de tensores en cada capa
   - Conectividad entre componentes
   - Número de parámetros

2. **Matemática Correcta**
   - Forward pass determinístico
   - Backward pass con gradientes invertidos (GRL)
   - Masking en LSTM funcional

3. **Procesamiento de Secuencias**
   - Zero-padding correcto
   - Agrupación por archivo de audio
   - Filtrado de secuencias muy cortas

4. **Integración End-to-End**
   - Pipeline completo funcional
   - No hay NaNs en outputs
   - Probabilidades suman 1.0

## 🎯 Buenas Prácticas Aplicadas

- ✅ Tests aislados e independientes
- ✅ Fixtures con setUp() para datos de prueba
- ✅ Nombres descriptivos de tests
- ✅ Assertions claras y específicas
- ✅ Cobertura de casos edge (secuencias cortas, padding, etc.)
- ✅ Tests de integración además de unitarios

## 📝 Agregar Nuevas Pruebas

Para agregar nuevas pruebas:

1. Crear clase que herede de `unittest.TestCase`
2. Implementar `setUp()` si necesitas fixtures
3. Nombrar tests como `test_descripcion_clara()`
4. Usar `self.assertEqual()`, `self.assertTrue()`, etc.
5. Agregar a `run_tests()` en el archivo principal

Ejemplo:
```python
class TestNuevaFuncionalidad(unittest.TestCase):
    def setUp(self):
        self.model = MiModelo()
    
    def test_nueva_funcionalidad(self):
        """Test: Descripción clara de qué verifica."""
        result = self.model.forward(datos_prueba)
        self.assertEqual(result.shape, (4, 2))
```

