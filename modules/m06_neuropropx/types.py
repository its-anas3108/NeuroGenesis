"""
NeuroProp-X data contracts.
===========================

Typed containers passed between SRVE, AP-LAF, ANP and SAGR, and consumed by
SAEG-GATv2, the XAI layer, the report and the dashboard.

Naming discipline (Sections 8, 32, 33) is enforced at the type level. The
variable that carries the ANP output is named ``propagation_score`` — never
``spread_probability`` — and :meth:`NeuroPropXOutput.disclaimer` returns the
wording that must accompany any rendering of these quantities. These are
model-derived discriminative quantities, not biological probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from modules.common.roi_constants import ROI_ORDER, roi_short


@dataclass
class SRVEOutput:
    """Stage-Aware Regional Vulnerability Estimation result.

    Attributes:
        regional_vulnerability: ``(B, N_ROI)`` in ``[0, 1]``. ``RV_i`` is a
            model-derived measure of how informative/vulnerable ROI ``i`` is for
            the current CN/MCI/AD discrimination — **not** a probability of
            future biological degeneration.
        logits: ``(B, N_ROI)`` pre-sigmoid values ``w_v^T x_i + b_v``.
        normalized_features: ``(B, N_ROI, F)`` the ``xhat_i`` actually consumed,
            retained so the dashboard can trace feature -> calculation -> RV.
    """

    regional_vulnerability: torch.Tensor
    logits: torch.Tensor
    normalized_features: Optional[torch.Tensor] = None

    def as_dict(self, batch_index: int = 0) -> Dict[str, float]:
        """Return ``{roi_name: RV}`` for one subject in the batch."""
        rv = self.regional_vulnerability[batch_index].detach().cpu().numpy()
        return {roi: float(rv[i]) for i, roi in enumerate(ROI_ORDER)}

    def ranking(self, batch_index: int = 0) -> List[Tuple[str, float]]:
        """Return ROIs ordered by descending vulnerability."""
        items = self.as_dict(batch_index).items()
        return sorted(items, key=lambda kv: kv[1], reverse=True)


@dataclass
class APLAFOutput:
    """Anatomical-Prior and Learned-Attention Fusion result.

    Attributes:
        adaptive_adjacency: ``(B, N, N)`` the fused ``A_star``.
        prior_adjacency: ``(B, N, N)`` the ``A_prior`` actually mixed in (row-
            normalised; see :mod:`modules.m06_neuropropx.ap_laf` for why).
        attention_adjacency: ``(B, N, N)`` the learned ``A_att``.
        alpha: Scalar mixing weight in ``[0, 1]``; ``alpha = sigmoid(a)``.
        raw_prior: ``(N, N)`` un-normalised prior, for display only.
    """

    adaptive_adjacency: torch.Tensor
    prior_adjacency: torch.Tensor
    attention_adjacency: torch.Tensor
    alpha: torch.Tensor
    raw_prior: Optional[torch.Tensor] = None
    #: ``(B, N, N)`` subject-specific structural-covariance adjacency,
    #: or ``None`` when the structural term is disabled.
    structural_adjacency: Optional[torch.Tensor] = None
    #: ``(3,)`` simplex weights ``[w_prior, w_att, w_struct]``.
    mix_weights: Optional[torch.Tensor] = None

    def alpha_value(self) -> float:
        """Return ``alpha`` (the prior's share) as a Python float."""
        return float(self.alpha.detach().cpu().reshape(-1)[0])

    def weights(self) -> Dict[str, float]:
        """Return the mixing weights by name."""
        if self.mix_weights is None:
            a = self.alpha_value()
            return {"prior": a, "attention": 1.0 - a, "structural": 0.0}
        w = self.mix_weights.detach().cpu().reshape(-1).tolist()
        return {"prior": float(w[0]), "attention": float(w[1]),
                "structural": float(w[2])}


@dataclass
class ANPOutput:
    """Adaptive Neurodegeneration Propagation result.

    Attributes:
        propagation_score: ``(B, N, N)`` in ``[0, 1]``. A learned relational
            representation combining endpoint vulnerability with adaptive
            connectivity. **Not** a validated biological disease-spread
            probability.
        logits: ``(B, N, N)`` pre-sigmoid values.
        topology_term: ``(B, N, N)`` the optional topology contribution, or
            ``None`` when disabled.
        beta: The learned coefficients ``(beta1, beta2, beta3, beta4, bias)``.
    """

    propagation_score: torch.Tensor
    logits: torch.Tensor
    topology_term: Optional[torch.Tensor] = None
    beta: Optional[Dict[str, float]] = None

    def top_pathways(self, batch_index: int = 0, k: int = 5
                     ) -> List[Dict[str, Any]]:
        """Return the ``k`` highest-scoring directed pathways for one subject.

        Self-loops are excluded: a node's relation to itself is not a pathway
        and would otherwise dominate the ranking.
        """
        P = self.propagation_score[batch_index].detach().cpu().numpy()
        n = P.shape[0]
        entries = [
            {
                "source": ROI_ORDER[i],
                "target": ROI_ORDER[j],
                "source_short": roi_short(ROI_ORDER[i]),
                "target_short": roi_short(ROI_ORDER[j]),
                "propagation_score": float(P[i, j]),
            }
            for i in range(n) for j in range(n) if i != j
        ]
        entries.sort(key=lambda e: e["propagation_score"], reverse=True)
        return entries[:k]


@dataclass
class SAGROutput:
    """Stage-Aware Graph Representation result.

    Attributes:
        node_features: ``(B, N, F_star)`` the enriched ``X_star``.
        component_slices: Name -> ``(start, stop)`` column range within
            ``node_features``, so XAI can attribute an importance value back to
            the block it came from (morphometry, CNN embedding, RV, centrality).
        centrality: ``(B, N, C)`` graph centrality features, or ``None``.
    """

    node_features: torch.Tensor
    component_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    centrality: Optional[torch.Tensor] = None

    @property
    def feature_dim(self) -> int:
        """Width of the enriched node representation ``F_star``."""
        return int(self.node_features.shape[-1])


@dataclass
class NeuroPropXOutput:
    """Complete NeuroProp-X output: the enhanced graph ``G*`` and its parts.

    ``G* = (V, X_star, A_star, P)``
    """

    srve: SRVEOutput
    ap_laf: APLAFOutput
    anp: ANPOutput
    sagr: SAGROutput
    #: Which submodules were actually active (ablation bookkeeping).
    active_components: Dict[str, bool] = field(default_factory=dict)

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def regional_vulnerability(self) -> torch.Tensor:
        """``(B, N)`` vulnerability vector ``RV``."""
        return self.srve.regional_vulnerability

    @property
    def adaptive_adjacency(self) -> torch.Tensor:
        """``(B, N, N)`` adaptive adjacency ``A_star``."""
        return self.ap_laf.adaptive_adjacency

    @property
    def propagation_score(self) -> torch.Tensor:
        """``(B, N, N)`` propagation representation ``P``."""
        return self.anp.propagation_score

    @property
    def node_features(self) -> torch.Tensor:
        """``(B, N, F_star)`` enriched node features ``X_star``."""
        return self.sagr.node_features

    def edge_features(self) -> torch.Tensor:
        """Stack the per-edge feature vector ``E_ij`` consumed by SAEG-GATv2.

        Returns:
            ``(B, N, N, 3)`` where the last axis is
            ``[A_prior_ij, A_att_ij, P_ij]`` — exactly the edge feature defined
            in Section 9.
        """
        return torch.stack(
            [
                self.ap_laf.prior_adjacency,
                self.ap_laf.attention_adjacency,
                self.anp.propagation_score,
            ],
            dim=-1,
        )

    # ── Export ────────────────────────────────────────────────────────────

    def to_numpy(self, batch_index: int = 0) -> Dict[str, np.ndarray]:
        """Extract one subject's matrices as NumPy arrays for persistence."""
        def npy(t: torch.Tensor) -> np.ndarray:
            return t[batch_index].detach().cpu().numpy()

        out = {
            "regional_vulnerability": npy(self.srve.regional_vulnerability),
            "adaptive_adjacency": npy(self.ap_laf.adaptive_adjacency),
            "prior_adjacency": npy(self.ap_laf.prior_adjacency),
            "attention_adjacency": npy(self.ap_laf.attention_adjacency),
            "propagation_score": npy(self.anp.propagation_score),
            "node_features": npy(self.sagr.node_features),
            "edge_features": npy(self.edge_features()),
        }
        if self.sagr.centrality is not None:
            out["centrality"] = npy(self.sagr.centrality)
        return out

    def metadata(self, batch_index: int = 0) -> Dict[str, Any]:
        """Return a JSON-serialisable description of ``G*`` for one subject."""
        return {
            "roi_order": list(ROI_ORDER),
            "n_nodes": len(ROI_ORDER),
            "node_feature_dim": self.sagr.feature_dim,
            "component_slices": {
                k: list(v) for k, v in self.sagr.component_slices.items()
            },
            "alpha": self.ap_laf.alpha_value(),
            "beta": self.anp.beta,
            "active_components": dict(self.active_components),
            "regional_vulnerability": self.srve.as_dict(batch_index),
            "vulnerability_ranking": [
                {"rank": r + 1, "roi": roi, "roi_short": roi_short(roi),
                 "vulnerability": v}
                for r, (roi, v) in enumerate(self.srve.ranking(batch_index))
            ],
            "top_pathways": self.anp.top_pathways(batch_index, k=5),
            "interpretation_note": self.disclaimer(),
        }

    @staticmethod
    def disclaimer() -> str:
        """Return the mandatory interpretation note for these quantities.

        Any figure, table, report section or dashboard panel that renders
        regional vulnerability or propagation scores must display this text.
        """
        return (
            "Regional vulnerability and propagation scores are model-derived "
            "quantities that describe how informative each speech-related "
            "region and inter-regional pathway is for the current CN/MCI/AD "
            "discrimination. They are not biological probabilities of future "
            "degeneration and not measurements of disease spread."
        )


__all__ = [
    "SRVEOutput",
    "APLAFOutput",
    "ANPOutput",
    "SAGROutput",
    "NeuroPropXOutput",
]
