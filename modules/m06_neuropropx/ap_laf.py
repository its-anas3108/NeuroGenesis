"""
M11.2 — AP-LAF: Anatomical-Prior and Learned-Attention Fusion (Section 8.2).
===========================================================================

Produces the adaptive adjacency

.. math::

    A^* = \\alpha\\, A_{\\text{prior}} + (1 - \\alpha)\\, A_{\\text{att}},
    \\qquad \\alpha = \\sigma(a) \\in [0, 1]

where :math:`A_{\\text{prior}}` is the documented anatomical prior
(:mod:`modules.m05_graph_construction.anatomical_prior`) and
:math:`A_{\\text{att}}` is a learned, subject-specific attention adjacency.

Two design decisions that materially affect the result
------------------------------------------------------

**1. The prior is row-normalised before it is mixed.** ``A_prior`` carries raw
ordinal weights in ``[0.50, 0.95]`` with a zero diagonal; ``A_att`` is a
row-softmax and so sums to 1 per row. Convex-combining tensors on different
scales would make ``alpha`` a scale correction rather than a meaningful
prior-versus-data trade-off, and the learned ``alpha`` would be
uninterpretable. Both operands are therefore row-stochastic before mixing, which
makes ``A_star`` row-stochastic too and ``alpha`` directly readable as "how much
of the adjacency mass comes from anatomy". The un-normalised prior is preserved
in :attr:`APLAFOutput.raw_prior` for display, so the dashboard can still show
the anatomical weights as authored.

**2. Learned attention is *not* restricted to the prior's support by default.**
Masking ``A_att`` to the 13 anatomically-connected pairs would make the learned
branch incapable of discovering an informative pathway the prior omits, which
defeats the purpose of learning it. Full attention over all 25 ordered pairs
(self-loops included) is therefore the default, and ``mask_to_prior=True``
exists only as an ablation lever.

Attention mechanism
-------------------

The pairwise score uses the GATv2 (Brody et al., 2022) *dynamic* form — the
nonlinearity is applied **before** the scoring vector::

    e_ij = a^T LeakyReLU( W [h_i || h_j] )

rather than the GAT-v1 static form ``LeakyReLU(a^T W[h_i || h_j])``. This
matters here: with only five nodes, static attention would collapse to a global
node ranking that is identical for every subject, and the "learned attention"
branch would contribute nothing subject-specific.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI
from modules.m05_graph_construction.anatomical_prior import (
    build_prior_matrix,
    prior_mask,
)
from modules.m06_neuropropx.types import APLAFOutput

logger = get_logger(__name__)


def row_normalize(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalise a non-negative matrix to sum 1 per row.

    A row that sums to zero is left at zero rather than being replaced by a
    uniform distribution: an ROI with no outgoing prior edge genuinely has no
    anatomical outflow, and manufacturing uniform connectivity there would
    invent structure the prior does not assert.

    Args:
        matrix: ``(..., N, N)`` non-negative tensor.
        eps: Numerical floor.

    Returns:
        Row-normalised tensor of the same shape.
    """
    row_sum = matrix.sum(dim=-1, keepdim=True)
    safe = torch.where(row_sum > eps, row_sum, torch.ones_like(row_sum))
    normed = matrix / safe
    return torch.where(row_sum > eps, normed, torch.zeros_like(normed))


class APLAF(nn.Module):
    """Anatomical-Prior and Learned-Attention Fusion.

    Args:
        node_dim: Width of the node representation ``H`` used to compute
            attention.
        hidden_dim: Width of the attention projection ``W``.
        alpha_logit_init: Initial value of the mixing logit ``a``. ``0.0``
            yields ``alpha = 0.5``, an unbiased start.
        learn_alpha: Whether ``a`` is trainable. Fixed by ablations A1/A3.
        learn_attention: Whether the attention branch is active at all. When
            ``False``, ``A_att`` is returned as the row-normalised prior and
            ``alpha`` is forced to 1, which is exactly ablation A1
            (prior only) with unchanged tensor shapes downstream.
        mask_to_prior: Restrict attention to the prior's support. Ablation
            lever only; see the module docstring.
        negative_slope: LeakyReLU slope inside the GATv2 scorer.
        n_roi: Number of nodes.

    Shape:
        input ``H`` of shape ``(B, N, node_dim)`` -> ``A_star`` of ``(B, N, N)``.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        alpha_logit_init: float = 0.0,
        learn_alpha: bool = True,
        learn_attention: bool = True,
        mask_to_prior: bool = False,
        negative_slope: float = 0.2,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.learn_alpha = learn_alpha
        self.learn_attention = learn_attention
        self.mask_to_prior = mask_to_prior
        self.negative_slope = negative_slope
        self.n_roi = n_roi

        # Anatomical prior: registered as a buffer so it moves with the module
        # across devices and is saved in the checkpoint, but is never optimised.
        raw = torch.from_numpy(build_prior_matrix(include_self_loops=False))
        self.register_buffer("raw_prior", raw, persistent=True)
        prior_with_loops = torch.from_numpy(
            build_prior_matrix(include_self_loops=True, self_loop_weight=1.0)
        )
        self.register_buffer(
            "prior_normalized", row_normalize(prior_with_loops), persistent=True
        )
        self.register_buffer(
            "support_mask",
            torch.from_numpy(prior_mask(include_self_loops=True)),
            persistent=True,
        )

        alpha_param = torch.tensor(float(alpha_logit_init))
        if learn_alpha and learn_attention:
            self.alpha_logit = nn.Parameter(alpha_param)
        else:
            # With attention disabled, alpha is pinned to 1 (prior only), so a
            # trainable logit would be a dead parameter that still shows up in
            # the parameter count and confuses ablation comparisons.
            self.register_buffer("alpha_logit", alpha_param, persistent=True)

        if learn_attention:
            self.W = nn.Linear(2 * node_dim, hidden_dim, bias=True)
            self.score = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
            nn.init.xavier_uniform_(self.score.weight)
        else:
            self.W = None
            self.score = None

    # ── Alpha ─────────────────────────────────────────────────────────────

    def alpha(self) -> torch.Tensor:
        """Return the mixing weight ``alpha = sigmoid(a)``, clamped to [0, 1].

        When the attention branch is disabled, ``alpha`` is exactly 1 so that
        ``A_star`` reduces to the prior.
        """
        if not self.learn_attention:
            return torch.ones((), device=self.prior_normalized.device,
                              dtype=self.prior_normalized.dtype)
        return torch.sigmoid(self.alpha_logit)

    # ── Attention ─────────────────────────────────────────────────────────

    def attention_adjacency(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the learned attention adjacency ``A_att``.

        Args:
            h: ``(B, N, node_dim)`` node representations.

        Returns:
            ``(B, N, N)`` row-stochastic attention matrix.
        """
        b, n, _ = h.shape

        # Build all ordered pairs [h_i || h_j] -> (B, N, N, 2*node_dim).
        h_i = h.unsqueeze(2).expand(b, n, n, self.node_dim)
        h_j = h.unsqueeze(1).expand(b, n, n, self.node_dim)
        pairs = torch.cat([h_i, h_j], dim=-1)

        # GATv2 dynamic attention: nonlinearity before the scoring vector.
        hidden = F.leaky_relu(self.W(pairs), negative_slope=self.negative_slope)
        scores = self.score(hidden).squeeze(-1)  # (B, N, N)

        if self.mask_to_prior:
            mask = self.support_mask.unsqueeze(0).expand(b, n, n)
            scores = scores.masked_fill(~mask, float("-inf"))

        return torch.softmax(scores, dim=-1)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, h: torch.Tensor) -> APLAFOutput:
        """Fuse the anatomical prior with learned attention.

        Args:
            h: ``(B, N, node_dim)`` node representations, or an unbatched
                ``(N, node_dim)`` tensor.

        Returns:
            An :class:`~modules.m06_neuropropx.types.APLAFOutput`.

        Raises:
            ValueError: On a node-count or feature-width mismatch.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if h.dim() != 3:
            raise ValueError(
                f"Expected (B, N, node_dim) or (N, node_dim); got {tuple(h.shape)}"
            )
        b, n, d = h.shape
        if n != self.n_roi:
            raise ValueError(
                f"Node axis is {n} but AP-LAF was built for {self.n_roi} nodes."
            )
        if d != self.node_dim:
            raise ValueError(
                f"Node feature width is {d} but AP-LAF was built for "
                f"{self.node_dim}."
            )

        prior = self.prior_normalized.unsqueeze(0).expand(b, n, n)

        if self.learn_attention:
            a_att = self.attention_adjacency(h)
        else:
            # Ablation A1: no learned branch. Returning the prior here (rather
            # than zeros) keeps A_star row-stochastic and keeps the edge-feature
            # tensor consumed by SAEG-GATv2 the same width.
            a_att = prior

        alpha = self.alpha()
        a_star = alpha * prior + (1.0 - alpha) * a_att

        return APLAFOutput(
            adaptive_adjacency=a_star,
            prior_adjacency=prior,
            attention_adjacency=a_att,
            alpha=alpha,
            raw_prior=self.raw_prior,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a description of the fusion configuration for the dashboard."""
        return {
            "formula": "A_star = alpha * A_prior + (1 - alpha) * A_att",
            "alpha": float(self.alpha().detach().cpu()),
            "alpha_is_learned": bool(self.learn_alpha and self.learn_attention),
            "attention_active": self.learn_attention,
            "attention_mechanism": (
                "GATv2 dynamic attention: e_ij = a^T LeakyReLU(W[h_i || h_j])"
                if self.learn_attention else "disabled (ablation)"
            ),
            "masked_to_prior_support": self.mask_to_prior,
            "prior_normalisation": (
                "A_prior is row-normalised (with self-loops) before mixing so "
                "that both operands are row-stochastic and alpha is "
                "interpretable as the share of adjacency mass from anatomy."
            ),
            "n_nodes": self.n_roi,
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"node_dim={self.node_dim}, hidden_dim={self.hidden_dim}, "
            f"learn_attention={self.learn_attention}, "
            f"learn_alpha={self.learn_alpha}, mask_to_prior={self.mask_to_prior}"
        )


__all__ = ["APLAF", "row_normalize"]
