"""
Dense graph attention primitives.
=================================

A single dense attention layer covering the three variants the study needs:

======================  ==================================================
``version="v1"``        GAT (Velickovic et al., 2018) — *static* attention:
                        ``e_ij = LeakyReLU(a^T [W h_i || W h_j])``
``version="v2"``        GATv2 (Brody et al., 2022) — *dynamic* attention:
                        ``e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)``
``edge_gate=True``      The proposed edge-gated extension (SAEG-GATv2).
======================  ==================================================

Why dense
---------

Every graph in this study has exactly five nodes and a dense 5x5 adjacency.
Sparse message-passing machinery would add a heavy dependency and buy nothing at
this size, while dense tensors make the attention matrix, the edge-gate matrix
and the message flow directly extractable for the dashboard panels required by
Sections 56 and 60.

Why the v1/v2 distinction matters here
--------------------------------------

In GAT-v1 the scoring vector ``a`` is applied *after* the nonlinearity, which
makes the ranking of keys identical for every query — Brody et al. call this
*static* attention. On a 5-node graph that degenerates into a single global node
ranking shared by every subject, so a "learned attention" branch built on v1
could not express subject-specific connectivity at all. GATv2 moves the
nonlinearity inside, making attention genuinely conditional on the query node.
This is why v2 is the basis of the proposed model and v1 is retained only as a
baseline.

Claim discipline
----------------

GATv2 itself is established prior work and is not claimed as novel. The proposed
element is the edge gate driven by NeuroProp-X edge features — see
:class:`~modules.m06_graph_learning.saeg_gatv2.SAEGGATv2Layer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Adjacency entries at or below this value are treated as absent, so a node
#: never attends across a pathway the adaptive graph assigns no weight.
ADJ_EPS = 1e-8


@dataclass
class AttentionTrace:
    """Inspectable internals of one attention layer's forward pass."""

    #: ``(B, heads, N, N)`` post-softmax attention, before gating.
    attention: torch.Tensor
    #: ``(B, heads, N, N)`` edge gate values, or ``None`` when ungated.
    edge_gate: Optional[torch.Tensor] = None
    #: ``(B, heads, N, N)`` the gated attention actually used for messages.
    gated_attention: Optional[torch.Tensor] = None
    #: ``(B, N, out_dim)`` layer output.
    output: Optional[torch.Tensor] = None
    #: ``(B, N, N)`` boolean support mask applied to the softmax.
    support_mask: Optional[torch.Tensor] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def head_mean_attention(self) -> torch.Tensor:
        """Return ``(B, N, N)`` attention averaged over heads."""
        source = self.gated_attention if self.gated_attention is not None \
            else self.attention
        return source.mean(dim=1)

    def node_importance(self) -> torch.Tensor:
        """Return ``(B, N)`` incoming attention mass per node.

        Column sums of the head-averaged attention: how much the other regions
        attend *to* this region. This is the node-importance signal consumed by
        the ROI ranking.
        """
        return self.head_mean_attention().sum(dim=1)


class DenseGraphAttentionLayer(nn.Module):
    """Multi-head dense graph attention with an optional edge gate.

    Args:
        in_dim: Input node feature width.
        out_dim: **Total** output width across all heads when ``concat=True``.
            Must be divisible by ``heads``.
        heads: Number of attention heads.
        version: ``"v1"`` (GAT, static) or ``"v2"`` (GATv2, dynamic).
        edge_dim: Width of the per-edge feature vector. Required when
            ``edge_gate`` is ``True``.
        edge_gate: Apply the NeuroProp-X edge gate
            ``alpha_tilde_ij = g_ij * alpha_ij``.
        gate_per_head: Learn a separate gate per attention head. Default
            ``False``, which matches the single scalar gate
            ``g_ij = sigmoid(w_g^T E_ij + b_g)`` of the research design; the
            per-head variant is available as a capacity extension.
        renormalize_after_gate: Re-normalise gated attention to sum 1 per row.
            Default ``False``: the gate is meant to be able to attenuate a
            node's total incoming message, and renormalising would cancel
            exactly that effect, leaving only a redistribution.
        dropout: Dropout applied to attention weights.
        negative_slope: LeakyReLU slope.
        concat: Concatenate heads (``True``) or average them (``False``).
            Averaging is conventional for the final layer.
        residual: Add a (projected) residual connection.
        bias: Add an output bias.

    Shape:
        ``x (B, N, in_dim)``, ``adj (B, N, N)``, ``edge_features (B, N, N,
        edge_dim)`` -> ``(B, N, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        version: str = "v2",
        edge_dim: Optional[int] = None,
        edge_gate: bool = False,
        gate_per_head: bool = False,
        renormalize_after_gate: bool = False,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        concat: bool = True,
        residual: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if version not in ("v1", "v2"):
            raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
        if concat and out_dim % heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by heads ({heads}) when "
                "concatenating heads."
            )
        if edge_gate and not edge_dim:
            raise ValueError(
                "edge_gate=True requires edge_dim; the gate is computed from "
                "the per-edge feature vector E_ij."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.version = version
        self.edge_dim = edge_dim or 0
        self.edge_gate = edge_gate
        self.gate_per_head = gate_per_head
        self.renormalize_after_gate = renormalize_after_gate
        self.negative_slope = negative_slope
        self.concat = concat
        self.head_dim = out_dim // heads if concat else out_dim

        if version == "v2":
            # GATv2: separate left/right projections summed before the
            # nonlinearity, so the score depends on the query node.
            self.lin_l = nn.Linear(in_dim, heads * self.head_dim, bias=False)
            self.lin_r = nn.Linear(in_dim, heads * self.head_dim, bias=False)
            self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        else:
            # GAT-v1: one shared projection, scoring vector over the
            # concatenated pair, applied after the nonlinearity.
            self.lin_l = nn.Linear(in_dim, heads * self.head_dim, bias=False)
            self.lin_r = None
            self.att = nn.Parameter(torch.empty(heads, 2 * self.head_dim))

        if edge_gate:
            gate_out = heads if gate_per_head else 1
            self.gate = nn.Linear(self.edge_dim, gate_out, bias=True)
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)

        if residual:
            need_proj = in_dim != (out_dim if concat else self.head_dim)
            self.res_proj = (
                nn.Linear(in_dim, out_dim if concat else self.head_dim, bias=False)
                if need_proj else nn.Identity()
            )
        else:
            self.res_proj = None

        self.bias = (
            nn.Parameter(torch.zeros(out_dim if concat else self.head_dim))
            if bias else None
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-initialise projections and the scoring vector."""
        nn.init.xavier_uniform_(self.lin_l.weight)
        if self.lin_r is not None:
            nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        if self.gate is not None:
            nn.init.xavier_uniform_(self.gate.weight)
            # Bias +1 starts every gate near sigmoid(1) ~ 0.73, i.e. mostly open.
            # A zero bias would halve every message at initialisation and slow
            # the first epochs for no reason.
            nn.init.constant_(self.gate.bias, 1.0)

    # ── Attention scores ──────────────────────────────────────────────────

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised attention logits ``(B, heads, N, N)``."""
        b, n, _ = x.shape
        h, d = self.heads, self.head_dim

        if self.version == "v2":
            hl = self.lin_l(x).view(b, n, h, d).permute(0, 2, 1, 3)
            hr = self.lin_r(x).view(b, n, h, d).permute(0, 2, 1, 3)
            # (B, H, N, 1, D) + (B, H, 1, N, D) -> (B, H, N, N, D)
            pair = hl.unsqueeze(3) + hr.unsqueeze(2)
            pair = F.leaky_relu(pair, negative_slope=self.negative_slope)
            # Nonlinearity applied before the scoring vector -> dynamic attention.
            return torch.einsum("bhijd,hd->bhij", pair, self.att)

        hp = self.lin_l(x).view(b, n, h, d).permute(0, 2, 1, 3)
        a_src, a_dst = self.att[:, :d], self.att[:, d:]
        # Scoring vector applied before the nonlinearity -> static attention.
        s_src = torch.einsum("bhnd,hd->bhn", hp, a_src).unsqueeze(3)
        s_dst = torch.einsum("bhnd,hd->bhn", hp, a_dst).unsqueeze(2)
        return F.leaky_relu(s_src + s_dst, negative_slope=self.negative_slope)

    def _projected_values(self, x: torch.Tensor) -> torch.Tensor:
        """Project node features to per-head value vectors ``(B, H, N, D)``."""
        b, n, _ = x.shape
        source = self.lin_r if (self.version == "v2" and self.lin_r is not None) \
            else self.lin_l
        return source(x).view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[AttentionTrace]]:
        """Run one attention layer.

        Args:
            x: ``(B, N, in_dim)`` node features.
            adj: ``(B, N, N)`` adjacency. Only its **support** is used, as a
                softmax mask; the numeric weights reach the layer through
                ``edge_features`` so that the gate can learn how to use them.
            edge_features: ``(B, N, N, edge_dim)`` per-edge features. Required
                when the edge gate is enabled.
            return_trace: Also return an :class:`AttentionTrace`.

        Returns:
            ``(output, trace)`` where ``trace`` is ``None`` unless requested.

        Raises:
            ValueError: On a shape mismatch, or if the gate is enabled but no
                edge features were supplied.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        b, n, f = x.shape
        if f != self.in_dim:
            raise ValueError(
                f"Node feature width {f} != layer in_dim {self.in_dim}"
            )
        if adj.shape != (b, n, n):
            raise ValueError(
                f"adj shape {tuple(adj.shape)} does not match nodes "
                f"(B={b}, N={n})"
            )

        scores = self._scores(x)  # (B, H, N, N)

        # Mask the softmax to the adjacency support, always allowing self-loops:
        # a node must be able to retain its own representation even if the
        # adaptive graph gives it no incoming edges.
        support = adj > ADJ_EPS
        eye = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0)
        support = support | eye
        mask = support.unsqueeze(1).expand(b, self.heads, n, n)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        # A fully masked row would emit NaN from softmax. The self-loop above
        # makes that impossible, but zeroing defensively keeps a NaN from
        # silently poisoning the whole batch if that invariant ever changes.
        attention = torch.nan_to_num(attention, nan=0.0)

        gate = None
        gated = attention
        if self.edge_gate:
            if edge_features is None:
                raise ValueError(
                    "edge_gate is enabled but edge_features is None."
                )
            if edge_features.dim() == 3:
                edge_features = edge_features.unsqueeze(0)
            if edge_features.shape[:3] != (b, n, n) or \
                    edge_features.shape[3] != self.edge_dim:
                raise ValueError(
                    f"edge_features shape {tuple(edge_features.shape)} does not "
                    f"match (B={b}, N={n}, N={n}, edge_dim={self.edge_dim})"
                )
            # g_ij = sigmoid(w_g^T E_ij + b_g)
            g = torch.sigmoid(self.gate(edge_features))       # (B,N,N,gate_out)
            gate = g.permute(0, 3, 1, 2)                       # (B,gate_out,N,N)
            if not self.gate_per_head:
                gate = gate.expand(b, self.heads, n, n)
            # alpha_tilde_ij = g_ij * alpha_ij
            gated = attention * gate
            if self.renormalize_after_gate:
                denom = gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                gated = gated / denom

        weights = self.dropout(gated)
        values = self._projected_values(x)                      # (B,H,N,D)
        out = torch.einsum("bhij,bhjd->bhid", weights, values)  # (B,H,N,D)

        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(b, n, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)

        if self.res_proj is not None:
            out = out + self.res_proj(x)
        if self.bias is not None:
            out = out + self.bias

        trace = None
        if return_trace:
            trace = AttentionTrace(
                attention=attention.detach(),
                edge_gate=gate.detach() if gate is not None else None,
                gated_attention=gated.detach(),
                output=out.detach(),
                support_mask=support.detach(),
                meta={
                    "version": self.version,
                    "heads": self.heads,
                    "head_dim": self.head_dim,
                    "in_dim": self.in_dim,
                    "out_dim": self.out_dim if self.concat else self.head_dim,
                    "edge_gate": self.edge_gate,
                    "gate_per_head": self.gate_per_head,
                    "renormalized_after_gate": self.renormalize_after_gate,
                    "attention_type": (
                        "dynamic (GATv2)" if self.version == "v2"
                        else "static (GAT)"
                    ),
                },
            )
        return out, trace

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, heads={self.heads}, "
            f"version={self.version}, edge_gate={self.edge_gate}, "
            f"concat={self.concat}"
        )


class GraphReadout(nn.Module):
    """Pool node representations into one graph embedding.

    Args:
        in_dim: Node feature width.
        mode: ``"mean_max"`` concatenates mean and max pooling (output width
            ``2 * in_dim``); ``"attention"`` uses a learned attention-pooling
            vector (output width ``in_dim``).

    Shape:
        ``(B, N, in_dim)`` -> ``(B, out_dim)``.
    """

    def __init__(self, in_dim: int, mode: str = "mean_max") -> None:
        super().__init__()
        if mode not in ("mean_max", "attention"):
            raise ValueError(
                f"mode must be 'mean_max' or 'attention', got {mode!r}"
            )
        self.in_dim = in_dim
        self.mode = mode
        if mode == "attention":
            self.score = nn.Linear(in_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.score.weight)
            self.out_dim = in_dim
        else:
            self.score = None
            self.out_dim = 2 * in_dim

    def forward(self, x: torch.Tensor, return_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pool node features into a graph embedding.

        Returns:
            ``(graph_embedding, pooling_weights)``. Pooling weights are
            ``(B, N)`` for attention pooling and ``None`` for mean/max.
        """
        if self.mode == "attention":
            w = torch.softmax(self.score(x).squeeze(-1), dim=-1)  # (B, N)
            pooled = torch.einsum("bn,bnd->bd", w, x)
            return pooled, (w if return_weights else None)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return pooled, None

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"in_dim={self.in_dim}, mode={self.mode}, out_dim={self.out_dim}"


__all__ = [
    "ADJ_EPS",
    "AttentionTrace",
    "DenseGraphAttentionLayer",
    "GraphReadout",
]
