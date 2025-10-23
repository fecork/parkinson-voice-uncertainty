"""
Common Model Components
=======================
Componentes compartidos entre diferentes arquitecturas de modelos.
"""

from .layers import (
    FeatureExtractor,
    GradientReversalFunction,
    GradientReversalLayer,
    ClassifierHead,
)

__all__ = [
    "FeatureExtractor",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "ClassifierHead",
]

