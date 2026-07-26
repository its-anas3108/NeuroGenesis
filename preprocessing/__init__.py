"""
NeuroGenesis Preprocessing Package
===================================
Provides MRI data loading, quality control, intensity normalization,
skull stripping, spatial resampling, and ROI-centric patch extraction.

Modules:
    loader            — MRI file discovery and loading (nibabel)
    artifact_detector — Automated MRI quality control (QC scoring)
    normalization     — N4 bias correction, CLAHE, anisotropic diffusion,
                        WM-peak normalization, Gaussian smoothing
    skull_strip       — Brain extraction (Nilearn / SimpleITK)
    resize            — Spatial resampling to target shape (SimpleITK)
    roi_crop          — ROI-centric 3D patch extraction for NeuroProp-X

Phase 2 Integration:
    roi_crop.get_neuroprox_tensor() → (5, 48, 48, 48) numpy array
"""

__version__ = "1.0.0"
__author__ = "NeuroGenesis Research Team"

from .loader import MRILoader
from .artifact_detector import ArtifactDetector
from .normalization import MRINormalizer
from .skull_strip import SkullStripper
from .resize import MRIResizer
from .roi_crop import ROICropper

__all__ = [
    "MRILoader",
    "ArtifactDetector",
    "MRINormalizer",
    "SkullStripper",
    "MRIResizer",
    "ROICropper",
]
