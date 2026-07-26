"""
NeuroGenesis Features Package
================================
Computes morphometric and intensity-based features from segmented ROI patches.

Modules:
    feature_extractor — Volume, grey matter volume, voxel count, mean/max/min
                        intensity, standard deviation, approximate surface area.
                        Provides get_feature_vector() hook for NeuroProp-X.
"""

__version__ = "1.0.0"

from .feature_extractor import FeatureExtractor

__all__ = ["FeatureExtractor"]
