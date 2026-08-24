"""
M11 — NeuroProp-X: Stage-Aware Regional Vulnerability and Adaptive Graph
Propagation.

The proposed algorithmic contribution of the framework. Transforms an ordinary
subject-specific brain graph ``G`` into a disease-aware enhanced graph
``G* = (V, X*, A*, P)`` through four stages:

===========  ========================================================
M11.1 SRVE   Stage-Aware Regional Vulnerability Estimation
M11.2 AP-LAF Anatomical-Prior and Learned-Attention Fusion
M11.3 ANP    Adaptive Neurodegeneration Propagation
M11.4 SAGR   Stage-Aware Graph Representation
===========  ========================================================
"""

from modules.m06_neuropropx.anp import ANP, IdentityPropagation
from modules.m06_neuropropx.ap_laf import APLAF, row_normalize
from modules.m06_neuropropx.neuropropx_engine import (
    NeuroPropX,
    NeuroPropXConfigFlags,
    save_neuropropx_output,
)
from modules.m06_neuropropx.sagr import CENTRALITY_FEATURES, SAGR, graph_centrality
from modules.m06_neuropropx.srve import SRVE, ConstantVulnerability
from modules.m06_neuropropx.types import (
    ANPOutput,
    APLAFOutput,
    NeuroPropXOutput,
    SAGROutput,
    SRVEOutput,
)

__all__ = [
    "SRVE",
    "ConstantVulnerability",
    "APLAF",
    "row_normalize",
    "ANP",
    "IdentityPropagation",
    "SAGR",
    "CENTRALITY_FEATURES",
    "graph_centrality",
    "NeuroPropX",
    "NeuroPropXConfigFlags",
    "save_neuropropx_output",
    "SRVEOutput",
    "APLAFOutput",
    "ANPOutput",
    "SAGROutput",
    "NeuroPropXOutput",
]
