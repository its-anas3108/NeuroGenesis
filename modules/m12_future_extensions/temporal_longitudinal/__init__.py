"""
Quarantined legacy code — NOT part of the validated current experiment.

Legacy closed-form atrophy extrapolation, previously mislabelled a
'Temporal Graph Transformer'. The ACTIVE Stage-TGT lives in
modules.m07_stage_tgt and is unrelated: it is a genuine transformer over
ordered stage representations and needs no longitudinal data.

Load via ``modules.m12_future_extensions.load_legacy(...)`` with an explicit
acknowledgement; this package intentionally re-exports nothing.
"""

IS_VALIDATED = False
