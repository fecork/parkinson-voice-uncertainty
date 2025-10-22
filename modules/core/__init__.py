"""
Core Module
===========
Módulos base compartidos: dataset, preprocessing, utils, visualization.
"""

from . import dataset
from . import preprocessing
from . import sequence_dataset
from . import utils
from . import visualization
from .dataset import DictDataset

__all__ = [
    "dataset",
    "preprocessing",
    "sequence_dataset",
    "utils",
    "visualization",
    "DictDataset",
]
