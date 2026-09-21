"""
M15 — Unified ROI ranking and ranking stability (Section 14).
============================================================

Combines the three independent ROI importance signals into one score:

.. math::

    \\text{ROI score}_i = \\lambda_1 RV_i + \\lambda_2 \\text{Attention}_i
                         + \\lambda_3 \\text{SHAP}_i

Normalisation before combining
------------------------------

The three signals live on incompatible scales: ``RV`` is a sigmoid output in
``[0, 1]``, attention is incoming softmax mass summing to ``N_ROI`` across
regions, and attribution magnitudes are unbounded and depend on the model's
probability scale. Combining them raw would let whichever signal happens to have
the largest numeric range dominate, and the ``lambda`` weights would silently
become scale corrections rather than importance weights.

Each signal is therefore rescaled to ``[0, 1]`` **within the subject** before
combining, using min-max over the five ROIs. That makes the combination a
weighted average of ranks-in-context and keeps the lambdas meaningful. A signal
that is constant across all five ROIs carries no ranking information and is
mapped to a uniform 0.5 with a recorded note, rather than to an arbitrary 0 or 1.

Missing signals
---------------

A model variant without NeuroProp-X has no ``RV``; one without attention has no
attention signal. Rather than substituting zeros — which would systematically
penalise every ROI equally and quietly change what the score means — the
available weights are **renormalised to sum to 1** over the signals that exist,
and the omission is recorded in :attr:`ROIRanking.signals_used`.

Stability
---------

:func:`ranking_stability` aggregates rankings across the subjects a
``--mode xai`` pass processed into mean rank, rank standard deviation,
selection frequency (how often the ROI appears in the top-``k``) and a
composite stability score. One "run" in this table's output (``n_runs``) is
one *subject* the ranking was computed for, not a repeated experiment
execution — the name is inherited from the design document's more general
framing, but every caller in this codebase supplies one :class:`ROIRanking`
per subject. With a small stage (e.g. AD n=6-30 depending on the cohort), a
single subject's ROI ranking is not a finding; the stability table, computed
across every available subject, is what makes the ranking reportable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.common.config import RankingConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER, roi_short

logger = get_logger(__name__)


def normalize_signal(values: Dict[str, float],
                     eps: float = 1e-12) -> Tuple[Dict[str, float], bool]:
    """Min-max rescale a per-ROI signal to ``[0, 1]``.

    Args:
        values: ``{roi: value}``.
        eps: Range below which the signal is treated as constant.

    Returns:
        ``(normalised, was_constant)``. A constant signal maps every ROI to 0.5,
        which carries no ranking preference in either direction.
    """
    if not values:
        return {}, True
    keys = list(values)
    array = np.array([values[k] for k in keys], dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return {k: 0.5 for k in keys}, True
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if (hi - lo) < eps:
        return {k: 0.5 for k in keys}, True
    out = {}
    for k, v in zip(keys, array):
        out[k] = float((v - lo) / (hi - lo)) if np.isfinite(v) else 0.5
    return out, False


@dataclass
class ROIRanking:
    """Unified ROI importance ranking for one subject or one group."""

    #: ``{roi: combined score}`` in ``[0, 1]``.
    scores: Dict[str, float] = field(default_factory=dict)
    #: Ordered ranking, most important first.
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    #: Signal name -> the normalised per-ROI values that went into the score.
    components: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Signal name -> effective weight actually applied after renormalisation.
    weights_used: Dict[str, float] = field(default_factory=dict)
    #: Which signals were available.
    signals_used: List[str] = field(default_factory=list)
    #: Which signals were requested but unavailable.
    signals_missing: List[str] = field(default_factory=list)
    #: Optional label, e.g. a stage name for a stage-specific ranking.
    group: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def top(self, k: int = 3) -> List[str]:
        """Return the top-``k`` ROI names."""
        return [entry["roi"] for entry in self.ranking[:k]]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "group": self.group,
            "scores": dict(self.scores),
            "ranking": list(self.ranking),
            "components": {k: dict(v) for k, v in self.components.items()},
            "weights_used": dict(self.weights_used),
            "signals_used": list(self.signals_used),
            "signals_missing": list(self.signals_missing),
            "notes": list(self.notes),
        }


def compute_roi_ranking(
    vulnerability: Optional[Dict[str, float]] = None,
    attention: Optional[Dict[str, float]] = None,
    attribution: Optional[Dict[str, float]] = None,
    cnn_importance: Optional[Dict[str, float]] = None,
    cfg: Optional[RankingConfig] = None,
    group: Optional[str] = None,
) -> ROIRanking:
    """Combine ROI importance signals into a unified ranking.

    Args:
        vulnerability: ``{roi: RV}`` from NeuroProp-X SRVE.
        attention: ``{roi: incoming attention}`` from SAEG-GATv2.
        attribution: ``{roi: aggregated attribution magnitude}`` from SHAP or
            permutation attribution.
        cnn_importance: ``{roi: occlusion importance}`` from the 3D CNN branch.
            Optional fourth signal; when supplied it shares the attribution
            weight equally with ``attribution``, so adding it does not silently
            inflate the total weight on model-attribution evidence.
        cfg: Ranking weights.
        group: Optional group label.

    Returns:
        An :class:`ROIRanking`.
    """
    cfg = cfg or RankingConfig()
    result = ROIRanking(group=group)

    raw: Dict[str, Tuple[Optional[Dict[str, float]], float]] = {
        "vulnerability": (vulnerability, cfg.lambda_vulnerability),
        "attention": (attention, cfg.lambda_attention),
    }
    if cnn_importance is not None and attribution is not None:
        raw["attribution"] = (attribution, cfg.lambda_shap / 2.0)
        raw["cnn_occlusion"] = (cnn_importance, cfg.lambda_shap / 2.0)
    elif cnn_importance is not None:
        raw["cnn_occlusion"] = (cnn_importance, cfg.lambda_shap)
    else:
        raw["attribution"] = (attribution, cfg.lambda_shap)

    available: Dict[str, float] = {}
    for name, (values, weight) in raw.items():
        if values:
            normalised, was_constant = normalize_signal(values)
            result.components[name] = normalised
            available[name] = weight
            if was_constant:
                result.notes.append(
                    f"The {name} signal is constant across all ROIs and "
                    "therefore contributes no ranking information; it was "
                    "mapped to a uniform 0.5."
                )
        else:
            result.signals_missing.append(name)

    if not available:
        result.notes.append(
            "No ROI importance signal was available, so no ranking could be "
            "produced."
        )
        return result

    total = sum(available.values())
    if total <= 0:
        result.notes.append("All available signal weights are zero.")
        return result

    result.weights_used = {k: v / total for k, v in available.items()}
    result.signals_used = sorted(available)
    if result.signals_missing:
        result.notes.append(
            f"Signals unavailable in this configuration: "
            f"{result.signals_missing}. The remaining weights were renormalised "
            "to sum to 1 rather than substituting zeros."
        )

    for roi in ROI_ORDER:
        result.scores[roi] = float(sum(
            result.weights_used[name] * result.components[name].get(roi, 0.5)
            for name in result.weights_used
        ))

    ordered = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)
    result.ranking = [
        {
            "rank": r + 1,
            "roi": roi,
            "roi_short": roi_short(roi),
            "score": score,
            **{
                f"{name}_normalized": result.components[name].get(roi)
                for name in result.weights_used
            },
        }
        for r, (roi, score) in enumerate(ordered)
    ]
    return result


@dataclass
class StabilityReport:
    """Ranking stability across subjects (Table 8).

    ``n_runs`` counts the subjects a ranking was computed for, not repeated
    experiment executions — see :func:`ranking_stability`.
    """

    n_runs: int
    top_k: int
    rows: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "n_runs": self.n_runs,
            "top_k": self.top_k,
            "rows": list(self.rows),
            "notes": list(self.notes),
        }


def ranking_stability(
    rankings: Sequence[ROIRanking],
    top_k: int = 3,
) -> StabilityReport:
    """Aggregate per-subject rankings into a stability table (Section 14, Table 8).

    Args:
        rankings: One :class:`ROIRanking` per subject the caller processed
            (every current call site — ``run.py``'s ``mode_xai`` — supplies
            exactly this: one ranking per subject, not one per repeated
            pipeline execution).
        top_k: Cutoff for the selection-frequency column.

    Returns:
        A :class:`StabilityReport` with, per ROI: mean rank, rank standard
        deviation, selection frequency and a composite stability score.

        The stability score combines being ranked highly with being ranked
        *consistently*::

            stability = 0.5 * (1 - (mean_rank - 1) / (N_ROI - 1))
                      + 0.5 * (1 - rank_sd / max_possible_sd)

        An ROI that is always first scores 1.0; one whose rank varies wildly
        scores low even if its mean rank is good. Both halves are needed: mean
        rank alone would let an ROI that alternates between first and last look
        as good as one that is consistently third.
    """
    report = StabilityReport(n_runs=len(rankings), top_k=top_k)
    if not rankings:
        report.notes.append("No rankings were supplied.")
        return report

    per_roi: Dict[str, List[int]] = {roi: [] for roi in ROI_ORDER}
    per_roi_scores: Dict[str, List[float]] = {roi: [] for roi in ROI_ORDER}
    selections: Dict[str, int] = {roi: 0 for roi in ROI_ORDER}

    usable = 0
    for ranking in rankings:
        if not ranking.ranking:
            continue
        usable += 1
        top = set(ranking.top(top_k))
        for entry in ranking.ranking:
            roi = entry["roi"]
            if roi in per_roi:
                per_roi[roi].append(int(entry["rank"]))
                per_roi_scores[roi].append(float(entry["score"]))
        for roi in top:
            if roi in selections:
                selections[roi] += 1

    if usable == 0:
        report.notes.append("Every supplied ranking was empty.")
        return report
    report.n_runs = usable

    # Largest achievable SD for a rank confined to [1, N_ROI]: alternating
    # between the extremes.
    max_sd = (N_ROI - 1) / 2.0

    for roi in ROI_ORDER:
        ranks = np.array(per_roi[roi], dtype=np.float64)
        if ranks.size == 0:
            report.rows.append({
                "roi": roi, "roi_short": roi_short(roi),
                "mean_rank": None, "rank_sd": None,
                "selection_frequency": 0.0, "stability_score": None,
                "n_runs": 0,
            })
            continue
        mean_rank = float(ranks.mean())
        rank_sd = float(ranks.std(ddof=1)) if ranks.size > 1 else 0.0
        rank_component = 1.0 - (mean_rank - 1.0) / max(N_ROI - 1, 1)
        consistency = 1.0 - min(rank_sd / max_sd, 1.0) if max_sd > 0 else 1.0
        report.rows.append({
            "roi": roi,
            "roi_short": roi_short(roi),
            "mean_rank": mean_rank,
            "rank_sd": rank_sd,
            "selection_frequency": float(selections[roi] / usable),
            "stability_score": float(0.5 * rank_component + 0.5 * consistency),
            "mean_score": float(np.mean(per_roi_scores[roi])),
            "n_runs": int(ranks.size),
        })

    report.rows.sort(
        key=lambda r: (r["mean_rank"] if r["mean_rank"] is not None else 999)
    )
    if usable < 5:
        report.notes.append(
            f"Stability is computed over only {usable} subject(s); rank "
            "standard deviations from so few subjects are themselves unstable."
        )
    return report


def stagewise_rankings(
    per_subject: Dict[str, Tuple[str, ROIRanking]],
) -> Dict[str, ROIRanking]:
    """Aggregate per-subject rankings into one ranking per stage (Table 6).

    Scores are averaged within each stage, then re-ranked. Averaging normalised
    per-subject scores rather than raw signals keeps every subject's
    contribution equally weighted, so a single subject with unusually large
    attribution magnitudes cannot dominate its group's ranking.

    Args:
        per_subject: ``{session_id: (stage, ranking)}``.

    Returns:
        ``{stage: ROIRanking}`` for every stage present.
    """
    grouped: Dict[str, List[ROIRanking]] = {}
    for stage, ranking in per_subject.values():
        if ranking.ranking:
            grouped.setdefault(stage, []).append(ranking)

    out: Dict[str, ROIRanking] = {}
    for stage, rankings in grouped.items():
        mean_scores = {
            roi: float(np.mean([r.scores.get(roi, 0.5) for r in rankings]))
            for roi in ROI_ORDER
        }
        aggregated = ROIRanking(group=stage, scores=mean_scores)
        aggregated.signals_used = sorted(
            {s for r in rankings for s in r.signals_used}
        )
        aggregated.weights_used = dict(rankings[0].weights_used)
        aggregated.notes.append(
            f"Mean of {len(rankings)} per-subject ranking(s) in stage {stage}."
        )
        ordered = sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True)
        aggregated.ranking = [
            {"rank": r + 1, "roi": roi, "roi_short": roi_short(roi),
             "score": score}
            for r, (roi, score) in enumerate(ordered)
        ]
        out[stage] = aggregated
    return out


__all__ = [
    "normalize_signal",
    "ROIRanking",
    "compute_roi_ranking",
    "StabilityReport",
    "ranking_stability",
    "stagewise_rankings",
]
