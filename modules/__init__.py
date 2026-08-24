"""
NeuroGenesis — Stage-Aware Speech-Network NeuroAI Framework
==========================================================

NeuroGenesis is a multimodal NeuroAI framework for stage-wise analysis of
Alzheimer's disease using structural MRI of speech-related brain networks. The
system localizes five speech-related regions using the Harvard-Oxford atlas,
learns local three-dimensional representations using a lightweight 3D CNN,
constructs a subject-specific brain graph, and applies the proposed NeuroProp-X
framework to estimate stage-relevant regional vulnerability, fuse anatomical
priors with learned graph attention, and derive adaptive propagation
representations. The resulting enriched disease graph is processed by an
edge-gated GATv2 model for CN/MCI/AD classification. A Stage-Temporal Graph
Transformer learns the ordered CN->MCI->AD representation to estimate
model-derived stage-transition propensity. Explainable AI, ROI ranking,
statistical analysis, and ablation studies provide interpretable and rigorous
evaluation.

Module map (numbers follow the research design document):

    common                  configuration, seeds, logging, paths, run state
    m01_dataset             OASIS-1 cohort, CDR label mapping, subject splits
    m02_preprocessing       MRI preprocessing chain (wraps ``preprocessing/``)
    m03_segmentation        Harvard-Oxford speech ROI localization
    m04_feature_extraction  ROI morphometric / textural features + scaler
    m05_graph_construction  anatomical prior + subject-specific brain graph
    m06_spatial_encoder     lightweight 3D CNN ROI patch encoder
    m06_neuropropx          SRVE / AP-LAF / ANP / SAGR
    m06_graph_learning      GAT + GATv2 baselines, SAEG-GATv2, fusion, head
    m07_stage_tgt           stage prototypes, transformer, propensity head
    m08_xai                 SHAP, attention, vulnerability, ROI ranking
    m09_statistics          stage-wise statistical comparison
    m10_results             research tables and figures
    m11_report              subject report generation
    m12_future_extensions   quarantined legacy longitudinal / digital twin code
    training                losses, trainer, evaluation, baselines, ablation
    inference               single-subject inference pipeline
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
