# Módulos Deprecados - Talos

Este directorio contiene los módulos de **Talos** que ya no se usan en el proyecto.

## ⚠️ Estado: DEPRECADO

**Fecha de deprecación:** Octubre 2025  
**Razón:** Migración de Talos a Optuna

## Archivos en este directorio

### Core Talos
- `talos_optimization.py` - Optimización de hiperparámetros con Talos
- `talos_evaluator.py` - Evaluador de resultados de Talos
- `talos_analysis.py` - Análisis de experimentos de Talos
- `talos_visualization.py` - Visualización de resultados de Talos

### Wrappers
- `cnn2d_talos_wrapper.py` - Wrapper de CNN2D para Talos
- `talos_optimization_models.py` - Optimización desde módulo models

## ⚠️ No Importar

**Estos módulos NO deben ser importados** ya que:
- Requieren Talos que ya no está instalado
- No son compatibles con el código actual
- Han sido reemplazados por módulos de Optuna

## Módulos Actuales

Los nuevos módulos de Optuna están en:
- `modules/core/optuna_optimization.py` - ✅ Optimización con Optuna
- `modules/core/cnn2d_optuna_wrapper.py` - ✅ Wrapper CNN2D para Optuna

## Información de Migración

Para más información sobre la migración, consulta:
- `docs/MIGRATION_TALOS_TO_OPTUNA.md` - Guía completa de migración
- `README.md` - Documentación principal del proyecto

---

**Última actualización:** Octubre 2025  
**Proyecto:** Parkinson Voice Uncertainty

