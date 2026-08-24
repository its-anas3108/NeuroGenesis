"""
M14 — Stage-Temporal Graph Transformer.

Learns the ordered CN -> MCI -> AD stage geometry and estimates model-derived
stage-transition propensity from a single cross-sectional scan. Requires no
longitudinal data and produces no forecast of future imaging.
"""

from modules.m07_stage_tgt.propensity_head import PropensityHead, PropensityOutput
from modules.m07_stage_tgt.stage_prototypes import PrototypeOutput, StagePrototypes
from modules.m07_stage_tgt.stage_transformer import (
    SUBJECT_TOKEN_INDEX,
    StageTransformer,
    StageTransformerOutput,
)

__all__ = [
    "StagePrototypes",
    "PrototypeOutput",
    "StageTransformer",
    "StageTransformerOutput",
    "SUBJECT_TOKEN_INDEX",
    "PropensityHead",
    "PropensityOutput",
]
