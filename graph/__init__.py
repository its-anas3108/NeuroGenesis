"""
NeuroGenesis Graph Package
============================
Constructs and visualizes a brain connectivity graph for speech-related regions.

Modules:
    graph_builder — NetworkX-based directed graph with anatomical edge weights.
                    Provides get_graph_data() hook for NeuroProp-X graph encoder.
"""

__version__ = "1.0.0"

from .graph_builder import BrainConnectivityGraph

__all__ = ["BrainConnectivityGraph"]
