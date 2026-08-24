"""
Baseline graph encoders (Section 18, baselines 3 and 4).
=======================================================

Two reference encoders that share SAEG-GATv2's interface so the comparison in
Table 2 is like-for-like:

* :class:`GATBaseline` — standard GAT (Velickovic et al., 2018), *static*
  attention, **no** edge gate, operating on the anatomical prior only.
* :class:`GATv2Baseline` — GATv2 (Brody et al., 2022), *dynamic* attention,
  **no** edge gate, anatomical prior only.

Both take exactly the same node features and produce a graph embedding of the
same width as SAEG-GATv2 under the same configuration, so a measured difference
is attributable to the attention mechanism and the edge gate rather than to
capacity or input differences. Neither receives NeuroProp-X edge features, and
neither sees the learned adjacency — that is precisely what they are baselines
*for*.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.config import GraphLearningConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI
from modules.m05_graph_construction.anatomical_prior import build_prior_matrix
from modules.m06_graph_learning.attention import (
    AttentionTrace,
    DenseGraphAttentionLayer,
    GraphReadout,
)
from modules.m06_graph_learning.saeg_gatv2 import GraphEncoderOutput

logger = get_logger(__name__)


class _AttentionBaseline(nn.Module):
    """Shared implementation for the GAT and GATv2 baselines.

    Args:
        in_dim: Node feature width.
        version: ``"v1"`` for GAT or ``"v2"`` for GATv2.
        cfg: Graph-learning hyper-parameters, shared with SAEG-GATv2.
        n_roi: Number of nodes.
    """

    def __init__(
        self,
        in_dim: int,
        version: str,
        cfg: Optional[GraphLearningConfig] = None,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GraphLearningConfig()
        self.in_dim = in_dim
        self.version = version
        self.n_roi = n_roi

        # The baselines are defined on the anatomical prior alone. It is stored
        # as a buffer so the encoder is self-contained: a caller cannot
        # accidentally feed it the learned adaptive adjacency and turn the
        # baseline into a partial version of the proposed model.
        prior = torch.from_numpy(
            build_prior_matrix(include_self_loops=True, self_loop_weight=1.0)
        )
        self.register_buffer("prior", prior, persistent=True)

        hidden = self.cfg.hidden_dim
        layers: List[DenseGraphAttentionLayer] = []
        width = in_dim
        for depth in range(self.cfg.n_layers):
            is_last = depth == self.cfg.n_layers - 1
            layers.append(
                DenseGraphAttentionLayer(
                    in_dim=width,
                    out_dim=hidden,
                    heads=self.cfg.heads,
                    version=version,
                    edge_dim=None,
                    edge_gate=False,
                    dropout=self.cfg.dropout,
                    negative_slope=self.cfg.negative_slope,
                    concat=not is_last,
                    residual=True,
                    bias=True,
                )
            )
            # See SAEGGATv2: both head-combination modes emit `hidden`.
            width = hidden
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(l.out_dim if l.concat else l.head_dim) for l in layers]
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.readout = GraphReadout(width, mode=self.cfg.readout)
        self.out_dim = self.readout.out_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> GraphEncoderOutput:
        """Encode node features over the anatomical prior.

        Args:
            x: ``(B, N, in_dim)`` node features.
            adj: Ignored. Accepted only so the baseline is a drop-in replacement
                for :class:`~modules.m06_graph_learning.saeg_gatv2.SAEGGATv2`;
                the prior buffer is always used.
            edge_features: Ignored — baselines have no edge channel.
            return_trace: Record per-layer attention.

        Returns:
            A :class:`GraphEncoderOutput`.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        a = self.prior.unsqueeze(0).expand(b, self.n_roi, self.n_roi)

        h = x
        traces: List[AttentionTrace] = []
        for depth, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h, trace = layer(h, a, None, return_trace=return_trace)
            h = norm(h)
            if depth < len(self.layers) - 1:
                h = self.dropout(self.act(h))
            if trace is not None:
                trace.meta["layer"] = depth + 1
                traces.append(trace)

        z_g, pool_w = self.readout(h, return_weights=return_trace)
        return GraphEncoderOutput(
            graph_embedding=z_g,
            node_embeddings=h,
            layer_traces=traces,
            pooling_weights=pool_w,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the baseline."""
        name = "GAT" if self.version == "v1" else "GATv2"
        citation = ("Velickovic et al., 2018" if self.version == "v1"
                    else "Brody et al., 2022")
        return {
            "name": f"{name} baseline",
            "attention": ("static (GAT)" if self.version == "v1"
                          else "dynamic (GATv2)"),
            "citation": citation,
            "edge_gate_active": False,
            "adjacency": "anatomical prior only (row-normalised, self-loops)",
            "uses_neuropropx_edge_features": False,
            "n_layers": self.cfg.n_layers,
            "heads": self.cfg.heads,
            "hidden_dim": self.cfg.hidden_dim,
            "readout": self.cfg.readout,
            "graph_embedding_dim": self.out_dim,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"in_dim={self.in_dim}, version={self.version}, "
            f"out_dim={self.out_dim}, layers={self.cfg.n_layers}"
        )


class GATBaseline(_AttentionBaseline):
    """Baseline 3: standard GAT with static attention over the anatomical prior."""

    def __init__(self, in_dim: int, cfg: Optional[GraphLearningConfig] = None,
                 n_roi: int = N_ROI) -> None:
        super().__init__(in_dim, version="v1", cfg=cfg, n_roi=n_roi)


class GATv2Baseline(_AttentionBaseline):
    """Baseline 4: GATv2 with dynamic attention over the anatomical prior."""

    def __init__(self, in_dim: int, cfg: Optional[GraphLearningConfig] = None,
                 n_roi: int = N_ROI) -> None:
        super().__init__(in_dim, version="v2", cfg=cfg, n_roi=n_roi)


__all__ = ["GATBaseline", "GATv2Baseline"]
