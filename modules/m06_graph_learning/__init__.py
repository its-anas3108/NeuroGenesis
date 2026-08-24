"""
M12/M13 — Graph learning: SAEG-GATv2, baselines, fusion and the stage classifier.

All graph attention is implemented in dense pure PyTorch. Every graph in this
study has exactly five nodes, so sparse message-passing machinery would add a
heavy dependency without benefit, while dense tensors keep attention matrices
and edge gates directly extractable for inspection.
"""

from modules.m06_graph_learning.attention import (
    ADJ_EPS,
    AttentionTrace,
    DenseGraphAttentionLayer,
    GraphReadout,
)
from modules.m06_graph_learning.classifier import (
    ClassificationOutput,
    StageClassifier,
)
from modules.m06_graph_learning.fusion import (
    FusionOutput,
    MultimodalFusion,
    SpatialBranchProjection,
)
from modules.m06_graph_learning.gat_baseline import GATBaseline, GATv2Baseline
from modules.m06_graph_learning.saeg_gatv2 import GraphEncoderOutput, SAEGGATv2

__all__ = [
    "ADJ_EPS",
    "AttentionTrace",
    "DenseGraphAttentionLayer",
    "GraphReadout",
    "SAEGGATv2",
    "GraphEncoderOutput",
    "GATBaseline",
    "GATv2Baseline",
    "SpatialBranchProjection",
    "MultimodalFusion",
    "FusionOutput",
    "StageClassifier",
    "ClassificationOutput",
]
