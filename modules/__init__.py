"""
Parkinson Voice Analysis - Modular Pipeline
============================================
Módulos para análisis de voz y detección de Parkinson.
"""

__version__ = "1.0.0"
__author__ = "PHD Research Team"

from . import preprocessing
from . import augmentation
from . import dataset
from . import utils
from . import visualization

__all__ = ["preprocessing", "augmentation", "dataset", "utils", "visualization"]
