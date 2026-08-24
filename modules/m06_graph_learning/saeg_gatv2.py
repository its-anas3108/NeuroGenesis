"""
M12 — SAEG-GATv2: Stage-Aware Edge-Gated Graph Attention Network v2 (Section 9).
==============================================================================

Learns a graph-level representation ``Z_G`` from the NeuroProp-X enhanced graph
``G* = (V, X*, A*, P)``.

.. code-block:: text

    G*  ->  SAEG-GATv2 layer 1 (multi-head, edge-gated)
        ->  SAEG-GATv2 layer 2 (multi-head, edge-gated)
        ->  graph readout (mean+max, or attention pooling)
        ->  Z_G

Edge features and the gate
--------------------------

Each directed edge carries the three-dimensional NeuroProp-X feature

.. math:: E_{ij} = \\left[\\, A^{\\text{prior}}_{ij},\\; A^{\\text{att}}_{ij},\\;
          P_{ij} \\,\\right]

from which a gate is computed and applied to the GATv2 attention:

.. math::

    g_{ij} = \\sigma\\!\\left(w_g^\\top E_{ij} + b_g\\right), \\qquad
    \\tilde{\\alpha}_{ij} = g_{ij}\\, \\alpha_{ij}

The gate lets the model *suppress* a message that node-feature attention would
otherwise pass, when the anatomical prior, the learned adjacency and the
propagation representation jointly argue the pathway is uninformative. Because
the gated attention is not re-normalised by default, a node's total incoming
message can be attenuated rather than merely redistributed — which is the point
of a gate rather than a reweighting.

What is and is not claimed as novel (Section 33)
------------------------------------------------

GATv2 is established prior work (Brody et al., 2022) and is used as the
foundation. The proposed elements are: the NeuroProp-X-enriched node input
``X*``, the three-channel NeuroProp-X edge feature, the edge gate applied to
dynamic attention, and the resulting stage-aware graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.config import GraphLearningConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER, roi_short
from modules.m06_graph_learning.attention import (
    AttentionTrace,
    DenseGraphAttentionLayer,
    GraphReadout,
)

logger = get_logger(__name__)


@dataclass
class GraphEncoderOutput:
    """Output of a graph encoder, with inspectable internals."""

    #: ``(B, out_dim)`` graph-level embedding ``Z_G``.
    graph_embedding: torch.Tensor
    #: ``(B, N, hidden)`` final node representations.
    node_embeddings: torch.Tensor
    #: Per-layer attention traces, when requested.
    layer_traces: List[AttentionTrace] = field(default_factory=list)
    #: ``(B, N)`` attention-pooling weights, when that readout is used.
    pooling_weights: Optional[torch.Tensor] = None

    def node_importance(self) -> Optional[torch.Tensor]:
        """Return ``(B, N)`` node importance from the last layer's attention.

        Incoming attention mass per node, averaged over heads. Returns ``None``
        when no trace was recorded.
        """
        if not self.layer_traces:
            return None
        return self.layer_traces[-1].node_importance()

    def edge_importance(self) -> Optional[torch.Tensor]:
        """Return ``(B, N, N)`` head-averaged gated attention from the last layer."""
        if not self.layer_traces:
            return None
        return self.layer_traces[-1].head_mean_attention()

    def top_edges(self, batch_index: int = 0, k: int = 5) -> List[Dict[str, Any]]:
        """Return the ``k`` most attended directed pathways for one subject.

        Self-loops are excluded — a node attending to itself is not a pathway.
        """
        imp = self.edge_importance()
        if imp is None:
            return []
        matrix = imp[batch_index].detach().cpu().numpy()
        entries = [
            {
                "source": ROI_ORDER[i],
                "target": ROI_ORDER[j],
                "source_short": roi_short(ROI_ORDER[i]),
                "target_short": roi_short(ROI_ORDER[j]),
                "attention": float(matrix[i, j]),
            }
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i != j
        ]
        entries.sort(key=lambda e: e["attention"], reverse=True)
        return entries[:k]


class SAEGGATv2(nn.Module):
    """Stage-Aware Edge-Gated Graph Attention Network v2.

    Args:
        in_dim: Width of the enriched node representation ``X*``.
        edge_dim: Width of the per-edge feature ``E_ij`` (3 for NeuroProp-X).
        cfg: Graph-learning hyper-parameters.
        n_roi: Number of nodes.

    Shape:
        ``x (B, N, in_dim)``, ``adj (B, N, N)``, ``edge (B, N, N, edge_dim)`` ->
        ``Z_G (B, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int = 3,
        cfg: Optional[GraphLearningConfig] = None,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GraphLearningConfig()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.n_roi = n_roi

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
                    version="v2",
                    edge_dim=edge_dim,
                    edge_gate=self.cfg.use_edge_gate,
                    gate_per_head=False,
                    renormalize_after_gate=False,
                    dropout=self.cfg.dropout,
                    negative_slope=self.cfg.negative_slope,
                    # Average heads on the final layer (conventional) and
                    # concatenate on earlier layers to preserve head diversity.
                    concat=not is_last,
                    residual=True,
                    bias=True,
                )
            )
            # Output width is `hidden` either way: a concatenating layer emits
            # heads * (hidden // heads) = hidden, and an averaging layer gives
            # each head the full `hidden` width and then averages. Recomputing
            # this by hand is how the declared out_dim drifts from reality.
            width = hidden
        self.layers = nn.ModuleList(layers)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(layer.out_dim if layer.concat else layer.head_dim)
             for layer in layers]
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(self.cfg.dropout)

        self.readout = GraphReadout(width, mode=self.cfg.readout)
        self.out_dim = self.readout.out_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> GraphEncoderOutput:
        """Encode the enhanced graph into a graph-level embedding.

        Args:
            x: ``(B, N, in_dim)`` enriched node features ``X*``.
            adj: ``(B, N, N)`` adaptive adjacency ``A*``.
            edge_features: ``(B, N, N, edge_dim)`` NeuroProp-X edge features.
            return_trace: Record per-layer attention and gate matrices.

        Returns:
            A :class:`GraphEncoderOutput`.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        h = x
        traces: List[AttentionTrace] = []
        for depth, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h, trace = layer(h, adj, edge_features, return_trace=return_trace)
            h = norm(h)
            # No activation after the final layer: the readout and fusion head
            # operate on the raw representation.
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

    # ── Introspection (Section 56) ────────────────────────────────────────

    def architecture(self) -> List[Dict[str, Any]]:
        """Return a per-stage description with tensor widths and head counts."""
        stages: List[Dict[str, Any]] = [
            {
                "stage": "input G*",
                "shape": [self.n_roi, self.in_dim],
                "detail": f"X_star ({self.in_dim}-d nodes), A_star, "
                          f"E_ij ({self.edge_dim}-d edges)",
            }
        ]
        for i, layer in enumerate(self.layers):
            stages.append({
                "stage": f"SAEG-GATv2 layer {i + 1}",
                "shape": [self.n_roi,
                          layer.out_dim if layer.concat else layer.head_dim],
                "heads": layer.heads,
                "head_dim": layer.head_dim,
                "attention": "dynamic (GATv2)",
                "edge_gate": layer.edge_gate,
                "head_combination": "concat" if layer.concat else "mean",
                "n_params": sum(p.numel() for p in layer.parameters()),
            })
        stages.append({
            "stage": f"graph readout ({self.cfg.readout})",
            "shape": [self.out_dim],
            "detail": ("mean pooling || max pooling"
                       if self.cfg.readout == "mean_max"
                       else "learned attention pooling"),
        })
        return stages

    def summary(self) -> Dict[str, Any]:
        """Return a full description of the encoder."""
        return {
            "name": "SAEG-GATv2",
            "full_name": "Stage-Aware Edge-Gated Graph Attention Network v2",
            "foundation": "GATv2 (Brody et al., 2022) — established prior work, "
                          "not claimed as novel",
            "proposed_elements": [
                "NeuroProp-X enriched node input X*",
                "three-channel NeuroProp-X edge feature "
                "E_ij = [A_prior_ij, A_att_ij, P_ij]",
                "edge gate g_ij = sigmoid(w_g^T E_ij + b_g) applied to dynamic "
                "attention as alpha_tilde_ij = g_ij * alpha_ij",
            ],
            "edge_gate_active": self.cfg.use_edge_gate,
            "n_layers": self.cfg.n_layers,
            "heads": self.cfg.heads,
            "hidden_dim": self.cfg.hidden_dim,
            "readout": self.cfg.readout,
            "graph_embedding_dim": self.out_dim,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "architecture": self.architecture(),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"in_dim={self.in_dim}, edge_dim={self.edge_dim}, "
            f"out_dim={self.out_dim}, layers={self.cfg.n_layers}, "
            f"heads={self.cfg.heads}, edge_gate={self.cfg.use_edge_gate}"
        )


__all__ = ["GraphEncoderOutput", "SAEGGATv2"]
