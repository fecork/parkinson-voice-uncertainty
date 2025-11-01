# Pruebas Deprecadas - Talos

Este directorio contiene las pruebas unitarias para **Talos**, que ya no se usa en el proyecto.

## ⚠️ Estado: DEPRECADO

**Fecha de deprecación:** Octubre 2025  
**Razón:** Migración de Talos a Optuna

## ¿Por qué están deprecadas?

El proyecto migró de **Talos** a **Optuna** para la optimización de hiperparámetros debido a:

1. ❌ **Talos no está mantenido** - Sin actualizaciones desde 2020
2. ❌ **Problemas de instalación** - Errores con scipy y dependencias
3. ❌ **Solo funciona con Keras** - No está diseñado para PyTorch
4. ✅ **Optuna es superior** - Activamente mantenido, mejor integración con PyTorch

## Archivos en este directorio

- `test_talos_basic.py` - Tests básicos de Talos
- `test_talos_file_generation.py` - Tests de generación de archivos
- `test_talos_integration.py` - Tests de integración
- `test_talos_optimization.py` - Tests de optimización
- `test_talos_real_files.py` - Tests con archivos reales
- `test_core_talos.py` - Tests del core de Talos
- `evaluate_talos.py` - Script de evaluación

## ⚠️ No Ejecutar

**Estas pruebas NO deben ejecutarse** ya que:
- Requieren Talos que ya no está instalado
- No son compatibles con el código actual
- Han sido reemplazadas por pruebas de Optuna

## Pruebas Actuales

Las nuevas pruebas de Optuna están en:
- `test/test_optuna_basic.py` - ✅ Tests básicos de Optuna
- `test/test_cnn2d_optuna.py` - ✅ Tests de integración CNN2D con Optuna

## Información de Migración

Para más información sobre la migración, consulta:
- `docs/MIGRATION_TALOS_TO_OPTUNA.md` - Guía completa de migración
- `README.md` - Documentación principal del proyecto

## ¿Se pueden eliminar?

Estos archivos se mantienen temporalmente por:
1. **Referencia histórica** - Documentación de cómo funcionaba el sistema anterior
2. **Comparación** - Para entender las diferencias con Optuna
3. **Recuperación** - En caso de necesitar revisar la implementación original

**Pueden ser eliminados en futuras versiones** una vez que Optuna esté completamente establecido.

---

**Última actualización:** Octubre 2025  
**Proyecto:** Parkinson Voice Uncertainty

