"""
M11.4 — SAGR: Stage-Aware Graph Representation (Section 8.4).
============================================================

Assembles the enriched node representation

.. math::

    X_i^* = \\left[\\, X_i^{\\text{morph}} \\;\\|\\; E_i^{3D} \\;\\|\\; RV_i \\,\\right]

optionally extended with graph centrality features, and packages it with the
adaptive adjacency and the propagation matrix into the NeuroProp-X output

.. math::

    G^* = (V, X^*, A^*, P)

Component slices
----------------

The concatenation order is recorded in
:attr:`~modules.m06_neuropropx.types.SAGROutput.component_slices` as
``name -> (start, stop)`` column ranges. This is what lets the XAI layer
attribute a SHAP value back to *which kind* of evidence it came from —
morphometry, learned 3D spatial structure, vulnerability, or topology — rather
than reporting an anonymous column index.

Centrality features
-------------------

Four robust, cheap quantities are derived from the adaptive adjacency
:math:`A^*`, all computed with dense tensor operations so they stay
differentiable and add no dependency:

* **out-strength** — row sum of :math:`A^*`
* **in-strength** — column sum of :math:`A^*`
* **strength balance** — out-strength minus in-strength, i.e. whether the region
  is a net source or net sink under the adaptive graph
* **eigenvector-style centrality** — a few power-iteration steps on the
  symmetrised :math:`A^*`, normalised per subject

Betweenness and closeness were deliberately not included: on a 5-node graph
they take very few distinct values and are dominated by the discretisation, so
they would add columns without adding information.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI
from modules.m06_neuropropx.types import SAGROutput

logger = get_logger(__name__)

#: Names of the centrality features, in the order they are concatenated.
CENTRALITY_FEATURES: List[str] = [
    "out_strength",
    "in_strength",
    "strength_balance",
    "eigenvector_centrality",
]


def graph_centrality(a_star: torch.Tensor, n_power_iters: int = 8,
                     eps: float = 1e-12) -> torch.Tensor:
    """Compute dense, differentiable centrality features from ``A_star``.

    Args:
        a_star: ``(B, N, N)`` adaptive adjacency, assumed non-negative.
        n_power_iters: Power-iteration steps for the eigenvector-style score.
            Eight is ample for a 5-node graph.
        eps: Numerical floor.

    Returns:
        ``(B, N, 4)`` tensor whose last axis follows :data:`CENTRALITY_FEATURES`.
    """
    b, n, _ = a_star.shape

    out_strength = a_star.sum(dim=2)
    in_strength = a_star.sum(dim=1)
    balance = out_strength - in_strength

    # Symmetrise before power iteration: an eigenvector centrality on a directed
    # graph with zero-out-degree nodes is not guaranteed to converge, and the
    # symmetrised version answers the question we actually want ("how centrally
    # embedded is this region") without that failure mode.
    sym = 0.5 * (a_star + a_star.transpose(1, 2))
    v = torch.full((b, n, 1), 1.0 / n, dtype=a_star.dtype, device=a_star.device)
    for _ in range(n_power_iters):
        v = torch.bmm(sym, v)
        norm = v.norm(dim=1, keepdim=True).clamp_min(eps)
        v = v / norm
    eig = v.squeeze(-1).abs()

    # Normalise each feature to [0, 1] per subject so that a subject with high
    # overall connectivity mass does not dominate the scale.
    def unit(x: torch.Tensor) -> torch.Tensor:
        lo = x.amin(dim=1, keepdim=True)
        hi = x.amax(dim=1, keepdim=True)
        return (x - lo) / (hi - lo).clamp_min(eps)

    return torch.stack(
        [unit(out_strength), unit(in_strength), unit(balance), unit(eig)],
        dim=-1,
    )


class SAGR(nn.Module):
    """Stage-Aware Graph Representation assembly.

    Args:
        morph_dim: Width of the morphometric feature block.
        cnn_dim: Width of the 3D CNN embedding block. Pass ``0`` for the
            morphometry-only ablations.
        use_centrality: Append the four centrality features.
        n_roi: Number of nodes.
    """

    def __init__(
        self,
        morph_dim: int,
        cnn_dim: int,
        use_centrality: bool = True,
        n_roi: int = N_ROI,
    ) -> None:
        super().__init__()
        if morph_dim < 0 or cnn_dim < 0:
            raise ValueError(
                f"Block widths must be non-negative; got morph_dim={morph_dim}, "
                f"cnn_dim={cnn_dim}"
            )
        if morph_dim == 0 and cnn_dim == 0:
            raise ValueError(
                "SAGR needs at least one of morph_dim or cnn_dim to be positive; "
                "a node representation consisting only of RV carries no "
                "regional evidence."
            )

        self.morph_dim = morph_dim
        self.cnn_dim = cnn_dim
        self.use_centrality = use_centrality
        self.n_roi = n_roi

        self.component_slices: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        if morph_dim:
            self.component_slices["morphometry"] = (cursor, cursor + morph_dim)
            cursor += morph_dim
        if cnn_dim:
            self.component_slices["cnn_embedding"] = (cursor, cursor + cnn_dim)
            cursor += cnn_dim
        self.component_slices["regional_vulnerability"] = (cursor, cursor + 1)
        cursor += 1
        if use_centrality:
            self.component_slices["centrality"] = (
                cursor, cursor + len(CENTRALITY_FEATURES)
            )
            cursor += len(CENTRALITY_FEATURES)
        self.out_dim = cursor

    @property
    def feature_names(self) -> List[str]:
        """Return per-column names of the enriched node representation."""
        names: List[str] = []
        names += [f"morph_{i}" for i in range(self.morph_dim)]
        names += [f"cnn_{i}" for i in range(self.cnn_dim)]
        names.append("regional_vulnerability")
        if self.use_centrality:
            names += list(CENTRALITY_FEATURES)
        return names

    def forward(
        self,
        morph_features: Optional[torch.Tensor],
        cnn_embeddings: Optional[torch.Tensor],
        regional_vulnerability: torch.Tensor,
        a_star: torch.Tensor,
    ) -> SAGROutput:
        """Assemble ``X_star``.

        Args:
            morph_features: ``(B, N, morph_dim)`` or ``None`` when
                ``morph_dim == 0``.
            cnn_embeddings: ``(B, N, cnn_dim)`` or ``None`` when
                ``cnn_dim == 0``.
            regional_vulnerability: ``(B, N)`` ``RV`` from SRVE.
            a_star: ``(B, N, N)`` adaptive adjacency, used for centrality.

        Returns:
            A :class:`~modules.m06_neuropropx.types.SAGROutput`.

        Raises:
            ValueError: If a required block is missing or mis-shaped.
        """
        blocks: List[torch.Tensor] = []

        if self.morph_dim:
            if morph_features is None:
                raise ValueError(
                    f"SAGR was built with morph_dim={self.morph_dim} but "
                    "morph_features is None."
                )
            if morph_features.dim() == 2:
                morph_features = morph_features.unsqueeze(0)
            if morph_features.shape[-1] != self.morph_dim:
                raise ValueError(
                    f"morph_features width is {morph_features.shape[-1]}, "
                    f"expected {self.morph_dim}"
                )
            blocks.append(morph_features)

        if self.cnn_dim:
            if cnn_embeddings is None:
                raise ValueError(
                    f"SAGR was built with cnn_dim={self.cnn_dim} but "
                    "cnn_embeddings is None."
                )
            if cnn_embeddings.dim() == 2:
                cnn_embeddings = cnn_embeddings.unsqueeze(0)
            if cnn_embeddings.shape[-1] != self.cnn_dim:
                raise ValueError(
                    f"cnn_embeddings width is {cnn_embeddings.shape[-1]}, "
                    f"expected {self.cnn_dim}"
                )
            blocks.append(cnn_embeddings)

        rv = regional_vulnerability
        if rv.dim() == 1:
            rv = rv.unsqueeze(0)
        blocks.append(rv.unsqueeze(-1))

        centrality: Optional[torch.Tensor] = None
        if self.use_centrality:
            if a_star.dim() == 2:
                a_star = a_star.unsqueeze(0)
            centrality = graph_centrality(a_star)
            blocks.append(centrality)

        x_star = torch.cat(blocks, dim=-1)

        if x_star.shape[-1] != self.out_dim:
            raise ValueError(
                f"Assembled X_star width {x_star.shape[-1]} does not match the "
                f"declared out_dim {self.out_dim}; component_slices would be "
                "wrong and XAI attribution would be misassigned."
            )

        return SAGROutput(
            node_features=x_star,
            component_slices=dict(self.component_slices),
            centrality=centrality,
        )

    def summary(self) -> Dict[str, object]:
        """Return a description of the assembled representation."""
        return {
            "formula": "X_i_star = [X_i_morph || E_i_3D || RV_i"
                       + (" || centrality]" if self.use_centrality else "]"),
            "out_dim": self.out_dim,
            "component_slices": {
                k: list(v) for k, v in self.component_slices.items()
            },
            "centrality_features": (
                list(CENTRALITY_FEATURES) if self.use_centrality else []
            ),
            "n_nodes": self.n_roi,
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (
            f"morph_dim={self.morph_dim}, cnn_dim={self.cnn_dim}, "
            f"use_centrality={self.use_centrality}, out_dim={self.out_dim}"
        )


__all__ = ["SAGR", "CENTRALITY_FEATURES", "graph_centrality"]
