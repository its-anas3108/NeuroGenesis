"""
NeuroGenesis Segmentation Package
===================================
Provides atlas-based ROI extraction for speech-related brain regions.

Modules:
    roi_extraction — Extracts Broca, Wernicke, Insula, IFG, STG masks
                     using the Nilearn Harvard-Oxford cortical atlas.
"""

__version__ = "1.0.0"

from .roi_extraction import ROIExtractor

__all__ = ["ROIExtractor"]
