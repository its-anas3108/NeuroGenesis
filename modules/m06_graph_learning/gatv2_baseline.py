"""
Re-export of the GATv2 baseline (Section 22 file layout).

The GAT and GATv2 baselines share one implementation in
:mod:`modules.m06_graph_learning.gat_baseline`, because the only difference
between them is whether the scoring vector is applied before or after the
nonlinearity. Duplicating the surrounding layer stack, readout and buffers into
a second file would create two copies to keep in sync, and any drift between
them would silently invalidate the baseline comparison.

This module exists so that the file layout matches the research design
document, and simply re-exports the class.
"""

from modules.m06_graph_learning.gat_baseline import GATv2Baseline

__all__ = ["GATv2Baseline"]
