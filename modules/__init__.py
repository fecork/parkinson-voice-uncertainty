"""
Parkinson Voice Analysis - Modular Pipeline
============================================
Módulos para análisis de voz y detección de Parkinson.
"""

__version__ = "2.0.0"
__author__ = "PHD Research Team"

from . import preprocessing
from . import augmentation
from . import dataset
from . import utils
from . import visualization
from . import cache_utils
from . import cnn_model
from . import cnn_training
from . import cnn_utils
from . import cnn_visualization

__all__ = [
    "preprocessing",
    "augmentation",
    "dataset",
    "utils",
    "visualization",
    "cache_utils",
    "cnn_model",
    "cnn_training",
    "cnn_utils",
    "cnn_visualization",
]
