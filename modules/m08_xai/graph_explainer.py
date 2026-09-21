"""
Graph and NeuroProp-X explanation (Sections 15B, 15C, 15D).
==========================================================

Extracts the explanation signals the model actually computes, rather than
re-deriving them from a formula:

**B. SAEG-GATv2 attention** — the per-layer, per-head attention matrices and
edge-gate matrices recorded during the forward pass. Node importance is incoming
attention mass; edge importance is the gated attention. These are read from
:class:`~modules.m06_graph_learning.attention.AttentionTrace`, so they are the
values used for the prediction, not a reconstruction of them.

**C. NeuroProp-X** — regional vulnerability ``RV`` and the propagation
representation ``P``, taken directly from the module outputs, plus the AP-LAF
``alpha`` and adjacency triple.

Everything connects back to the CN/MCI/AD prediction (Section 60): each signal is
reported together with the predicted stage and its probability, and the
class-specific variants are computed per target class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER, STAGE_ORDER, roi_short
from modules.m05_graph_construction.anatomical_prior import tract_of
from modules.model import ModelOutput

logger = get_logger(__name__)


@dataclass
class GraphExplanation:
    """Attention-based node and edge importance for one subject."""

    #: ``{roi: incoming attention mass}``.
    node_importance: Dict[str, float] = field(default_factory=dict)
    #: ``(N_ROI, N_ROI)`` head-averaged gated attention.
    edge_attention: Optional[np.ndarray] = None
    #: ``(N_ROI, N_ROI)`` head-averaged edge gate.
    edge_gate: Optional[np.ndarray] = None
    #: Per-layer metadata (heads, widths, gate status).
    layers: List[Dict[str, Any]] = field(default_factory=list)
    #: ``{roi: weight}`` from attention pooling, when that readout is used.
    pooling_weights: Optional[Dict[str, float]] = None
    notes: List[str] = field(default_factory=list)

    def top_edges(self, k: int = 5) -> List[Dict[str, Any]]:
        """Return the ``k`` most attended directed pathways, with tract names."""
        if self.edge_attention is None:
            return []
        matrix = self.edge_attention
        entries = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    continue
                entries.append({
                    "source": ROI_ORDER[i],
                    "target": ROI_ORDER[j],
                    "pathway": f"{roi_short(ROI_ORDER[i])} -> "
                               f"{roi_short(ROI_ORDER[j])}",
                    "attention": float(matrix[i, j]),
                    "gate": (
                        float(self.edge_gate[i, j])
                        if self.edge_gate is not None else None
                    ),
                    "anatomical_tract": tract_of(ROI_ORDER[i], ROI_ORDER[j]),
                })
        entries.sort(key=lambda e: e["attention"], reverse=True)
        return entries[:k]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "node_importance": dict(self.node_importance),
            "edge_attention": (
                self.edge_attention.tolist()
                if self.edge_attention is not None else None
            ),
            "edge_gate": (
                self.edge_gate.tolist() if self.edge_gate is not None else None
            ),
            "top_edges": self.top_edges(5),
            "layers": list(self.layers),
            "pooling_weights": self.pooling_weights,
            "roi_order": list(ROI_ORDER),
            "notes": list(self.notes),
        }


@dataclass
class NeuroPropXExplanation:
    """NeuroProp-X explanation signals for one subject."""

    regional_vulnerability: Dict[str, float] = field(default_factory=dict)
    vulnerability_ranking: List[Dict[str, Any]] = field(default_factory=list)
    propagation_matrix: Optional[np.ndarray] = None
    adaptive_adjacency: Optional[np.ndarray] = None
    prior_adjacency: Optional[np.ndarray] = None
    attention_adjacency: Optional[np.ndarray] = None
    alpha: Optional[float] = None
    beta: Optional[Dict[str, float]] = None
    top_pathways: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-feature SRVE weight decomposition for the highest-vulnerability ROI.
    srve_trace: Optional[Dict[str, Any]] = None
    disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        def arr(a: Optional[np.ndarray]) -> Optional[List[List[float]]]:
            return a.tolist() if a is not None else None

        return {
            "regional_vulnerability": dict(self.regional_vulnerability),
            "vulnerability_ranking": list(self.vulnerability_ranking),
            "propagation_matrix": arr(self.propagation_matrix),
            "adaptive_adjacency": arr(self.adaptive_adjacency),
            "prior_adjacency": arr(self.prior_adjacency),
            "attention_adjacency": arr(self.attention_adjacency),
            "alpha": self.alpha,
            "beta": self.beta,
            "top_pathways": list(self.top_pathways),
            "srve_trace": self.srve_trace,
            "roi_order": list(ROI_ORDER),
            "disclaimer": self.disclaimer,
        }


def explain_graph(output: ModelOutput, batch_index: int = 0
                  ) -> GraphExplanation:
    """Extract attention-based explanation from a traced forward pass.

    Args:
        output: A :class:`~modules.model.ModelOutput` produced with
            ``return_trace=True``.
        batch_index: Subject within the batch.

    Returns:
        A :class:`GraphExplanation`. When the model variant has no attention
        (A0 morphometry-only, A1 prior propagation) the result is empty and
        carries an explanatory note rather than zeros.
    """
    explanation = GraphExplanation()

    if output.graph is None:
        explanation.notes.append("This model variant has no graph encoder.")
        return explanation
    if not output.graph.layer_traces:
        explanation.notes.append(
            "No attention was recorded. Either the forward pass ran without "
            "return_trace=True, or this encoder has no attention to record "
            "(the prior-propagation and morphometry-only variants do not)."
        )
        return explanation

    last = output.graph.layer_traces[-1]
    attention = last.head_mean_attention()[batch_index].cpu().numpy()
    explanation.edge_attention = attention

    if last.edge_gate is not None:
        explanation.edge_gate = (
            last.edge_gate[batch_index].mean(dim=0).cpu().numpy()
        )
    else:
        explanation.notes.append(
            "The edge gate is disabled in this variant, so no gate matrix "
            "exists."
        )

    incoming = attention.sum(axis=0)
    explanation.node_importance = {
        roi: float(incoming[i]) for i, roi in enumerate(ROI_ORDER)
    }

    for trace in output.graph.layer_traces:
        explanation.layers.append({
            "layer": trace.meta.get("layer"),
            "attention_type": trace.meta.get("attention_type"),
            "heads": trace.meta.get("heads"),
            "head_dim": trace.meta.get("head_dim"),
            "in_dim": trace.meta.get("in_dim"),
            "out_dim": trace.meta.get("out_dim"),
            "edge_gate": trace.meta.get("edge_gate"),
            "attention_shape": list(trace.attention.shape),
            "gate_range": (
                [float(trace.edge_gate.min()), float(trace.edge_gate.max())]
                if trace.edge_gate is not None else None
            ),
        })

    if output.graph.pooling_weights is not None:
        weights = output.graph.pooling_weights[batch_index].cpu().numpy()
        explanation.pooling_weights = {
            roi: float(weights[i]) for i, roi in enumerate(ROI_ORDER)
        }

    return explanation


def explain_neuropropx(output: ModelOutput, model: Any = None,
                       batch_index: int = 0) -> NeuroPropXExplanation:
    """Extract NeuroProp-X explanation signals.

    Args:
        output: Model output.
        model: The model, used to trace the SRVE weight decomposition for the
            most vulnerable ROI. Optional.
        batch_index: Subject within the batch.

    Returns:
        A :class:`NeuroPropXExplanation`. Empty with a disclaimer when the
        variant has no NeuroProp-X.
    """
    from modules.m06_neuropropx.types import NeuroPropXOutput

    explanation = NeuroPropXExplanation(disclaimer=NeuroPropXOutput.disclaimer())

    npx = output.neuropropx
    if npx is None:
        explanation.disclaimer = (
            "This model variant does not include NeuroProp-X, so no regional "
            "vulnerability or propagation representation exists."
        )
        return explanation

    arrays = npx.to_numpy(batch_index)
    explanation.regional_vulnerability = npx.srve.as_dict(batch_index)
    explanation.vulnerability_ranking = [
        {"rank": r + 1, "roi": roi, "roi_short": roi_short(roi),
         "vulnerability": v}
        for r, (roi, v) in enumerate(npx.srve.ranking(batch_index))
    ]
    explanation.propagation_matrix = arrays["propagation_score"]
    explanation.adaptive_adjacency = arrays["adaptive_adjacency"]
    explanation.prior_adjacency = arrays["prior_adjacency"]
    explanation.attention_adjacency = arrays["attention_adjacency"]
    explanation.alpha = npx.ap_laf.alpha_value()
    explanation.beta = npx.anp.beta
    explanation.top_pathways = npx.anp.top_pathways(batch_index, k=5)

    # Trace the highest-vulnerability ROI back to its feature contributions.
    if (model is not None and getattr(model, "neuropropx", None) is not None
            and npx.srve.normalized_features is not None
            and hasattr(model.neuropropx.srve, "explain")):
        top_roi = explanation.vulnerability_ranking[0]["roi"]
        try:
            explanation.srve_trace = model.neuropropx.srve.explain(
                npx.srve.normalized_features,
                batch_index=batch_index,
                roi_index=ROI_ORDER.index(top_roi),
            )
        except (ValueError, IndexError, RuntimeError) as exc:
            logger.debug("SRVE trace unavailable: %s", exc)

    return explanation


__all__ = [
    "GraphExplanation",
    "NeuroPropXExplanation",
    "explain_graph",
    "explain_neuropropx",
]
