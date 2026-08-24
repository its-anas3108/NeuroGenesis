"""M15/M16 — Explainable AI: attribution, attention, vulnerability, ROI ranking."""

from modules.m08_xai.graph_explainer import (
    GraphExplanation,
    NeuroPropXExplanation,
    cnn_occlusion_importance,
    explain_graph,
    explain_neuropropx,
)
from modules.m08_xai.roi_ranking import (
    ROIRanking,
    StabilityReport,
    compute_roi_ranking,
    normalize_signal,
    ranking_stability,
    stagewise_rankings,
)
from modules.m08_xai.shap_explainer import (
    AttributionResult,
    FeatureAttributor,
    shap_available,
)

__all__ = [
    "shap_available",
    "AttributionResult",
    "FeatureAttributor",
    "GraphExplanation",
    "NeuroPropXExplanation",
    "explain_graph",
    "explain_neuropropx",
    "cnn_occlusion_importance",
    "ROIRanking",
    "compute_roi_ranking",
    "normalize_signal",
    "StabilityReport",
    "ranking_stability",
    "stagewise_rankings",
]
