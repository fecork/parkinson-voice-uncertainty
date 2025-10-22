"""
Parkinson Voice Analysis - Modular Pipeline
============================================
Módulos para análisis de voz y detección de Parkinson.

Estructura:
- core: Módulos base (dataset, preprocessing, utils, visualization)
- data: Manejo de datos (augmentation, cache)
- models: Modelos de ML (cnn2d, cnn1d, uncertainty)
"""

__version__ = "4.0.0"
__author__ = "PHD Research Team"

from . import core
from . import data
from . import models

__all__ = ["core", "data", "models"]
