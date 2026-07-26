"""
NeuroGenesis Visualization Package
=====================================
Produces publication-quality figures for all pipeline stages.

Modules:
    visualize — NeuroGenesisVisualizer class. Generates tri-plane views,
                preprocessing before/after comparisons, ROI overlays,
                feature tables, connectivity graphs, and pipeline summaries.
                All figures saved at 300 DPI.
"""

__version__ = "1.0.0"

from .visualize import NeuroGenesisVisualizer

__all__ = ["NeuroGenesisVisualizer"]
