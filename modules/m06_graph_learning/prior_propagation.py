"""
Fixed anatomical-prior propagation encoder (ablation A1).
=========================================================

A graph-convolution-style encoder with **no learned attention at all**:

.. math::

    H^{(l+1)} = \\mathrm{ELU}\\!\\left( \\hat{A}_{\\text{prior}} H^{(l)} W^{(l)} \\right)

where :math:`\\hat{A}_{\\text{prior}}` is the row-normalised anatomical prior with
self-loops, held as a fixed buffer.

This is the A1 rung of the ablation ladder — "morphometry + anatomical prior".
Without it, A1 would have to be built from an attention layer with attention
switched off, which is not the same experiment: the ladder is meant to establish
that *any* use of the anatomical graph helps before asking whether *learned*
attention helps on top of it. A2 (standard GAT) and A3 (prior + learned
attention) then measure exactly that increment.

The layer stack, readout width and hidden dimension follow the same
:class:`~modules.common.config.GraphLearningConfig` as every other encoder, so
the comparison is not confounded by capacity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.config import GraphLearningConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI
from modules.m05_graph_construction.anatomical_prior import build_prior_matrix
from modules.m06_graph_learning.attention import GraphReadout
from modules.m06_graph_learning.saeg_gatv2 import GraphEncoderOutput

logger = get_logger(__name__)


class PriorPropagationEncoder(nn.Module):
    """Fixed-adjacency graph propagation over the anatomical prior.

    Args:
        in_dim: Node feature width.
        cfg: Graph-learning hyper-parameters (shared with the other encoders).
        n_roi: Number of nodes.

    Shape:
        ``(B, N, in_dim)`` -> ``Z_G (B, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: Optional[GraphLearningConfig] = None,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GraphLearningConfig()
        self.in_dim = in_dim
        self.n_roi = n_roi

        prior = build_prior_matrix(
            include_self_loops=True, self_loop_weight=1.0, normalize=True
        )
        self.register_buffer("prior", torch.from_numpy(prior), persistent=True)

        hidden = self.cfg.hidden_dim
        widths = [in_dim] + [hidden] * self.cfg.n_layers
        self.linears = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1], bias=True)
             for i in range(self.cfg.n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(self.cfg.n_layers)]
        )
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.act = nn.ELU()
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.readout = GraphReadout(hidden, mode=self.cfg.readout)
        self.out_dim = self.readout.out_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> GraphEncoderOutput:
        """Propagate node features over the fixed prior.

        Args:
            x: ``(B, N, in_dim)`` node features.
            adj: Ignored — the fixed prior buffer is always used. Accepted so
                this encoder is a drop-in replacement for the others.
            edge_features: Ignored.
            return_trace: Record the (fixed) propagation matrix, so the
                dashboard can show what was used even though nothing was learned.

        Returns:
            A :class:`GraphEncoderOutput` with an empty ``layer_traces`` list —
            there is no attention to trace.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        a = self.prior.unsqueeze(0).expand(b, self.n_roi, self.n_roi)

        h = x
        for depth, (lin, norm) in enumerate(zip(self.linears, self.norms)):
            h = torch.bmm(a, lin(h))
            h = norm(h)
            if depth < len(self.linears) - 1:
                h = self.dropout(self.act(h))

        z_g, pool_w = self.readout(h, return_weights=return_trace)
        return GraphEncoderOutput(
            graph_embedding=z_g,
            node_embeddings=h,
            layer_traces=[],
            pooling_weights=pool_w,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the encoder."""
        return {
            "name": "Prior propagation encoder",
            "formula": "H' = ELU(A_prior_hat @ H @ W)",
            "attention": "none — fixed anatomical adjacency",
            "edge_gate_active": False,
            "adjacency": "anatomical prior only (row-normalised, self-loops), "
                         "held fixed",
            "uses_neuropropx_edge_features": False,
            "n_layers": self.cfg.n_layers,
            "hidden_dim": self.cfg.hidden_dim,
            "readout": self.cfg.readout,
            "graph_embedding_dim": self.out_dim,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"layers={self.cfg.n_layers} (no attention)")


class MorphometryMLP(nn.Module):
    """Graph-free encoder over flattened per-ROI features (ablation A0, baseline 2).

    Concatenates every ROI's feature vector and passes it through a two-layer
    MLP. There is no graph, no attention and no adjacency — the model can still
    combine information across regions, but only through dense weights, with no
    notion of which regions are anatomically connected.

    Args:
        in_dim: Per-ROI feature width.
        hidden_dim: Hidden width.
        out_dim: Output width, matched to a graph encoder's ``out_dim`` so the
            downstream fusion and classifier are unchanged.
        dropout: Dropout rate.
        n_roi: Number of ROIs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        dropout: float = 0.2,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_roi = n_roi
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(n_roi * in_dim),
            nn.Dropout(dropout),
            nn.Linear(n_roi * in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> GraphEncoderOutput:
        """Encode flattened per-ROI features.

        Args:
            x: ``(B, N, in_dim)`` node features.
            adj: Ignored — this encoder has no graph.
            edge_features: Ignored.
            return_trace: Ignored; nothing is traced.

        Returns:
            A :class:`GraphEncoderOutput`. ``node_embeddings`` returns the input
            unchanged, since no per-node representation is learned.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b, n, f = x.shape
        if n != self.n_roi or f != self.in_dim:
            raise ValueError(
                f"Expected (B, {self.n_roi}, {self.in_dim}); got {tuple(x.shape)}"
            )
        z = self.net(x.reshape(b, n * f))
        return GraphEncoderOutput(
            graph_embedding=z, node_embeddings=x, layer_traces=[],
            pooling_weights=None,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the encoder."""
        return {
            "name": "Morphometry MLP",
            "formula": "Z = MLP(concat_i X_i)",
            "attention": "none",
            "graph": "none — no adjacency is used",
            "edge_gate_active": False,
            "uses_neuropropx_edge_features": False,
            "graph_embedding_dim": self.out_dim,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (f"in_dim={self.in_dim}, n_roi={self.n_roi}, "
                f"out_dim={self.out_dim} (no graph)")


__all__ = ["PriorPropagationEncoder", "MorphometryMLP"]
