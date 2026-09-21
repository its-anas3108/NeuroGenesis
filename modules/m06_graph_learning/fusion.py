"""
M13 — Multimodal fusion (Section 10).
=====================================

Combines the graph branch into the shared representation used by **both** the
current-stage classifier and the Stage-TGT:

.. code-block:: text

    graph branch   ->  Z_G   ->  MLP  ->  Z_H

The ``spatial_dim``/``z_3d`` plumbing is retained so the fusion head can still
run graph-only (``spatial_dim=0``, the only mode the model uses now) without
further changes.

``Z_H`` is deliberately shared. If the classifier and the Stage-TGT each had
their own trunk, the stage prototypes would live in a different space from the
one the classifier decides in, and "distance to the AD prototype" would carry no
relationship to "probability of AD". Sharing the trunk is what makes the
propensity score commensurate with the classification.

Sizing
------

The fusion head is kept small on purpose. With 154 training subjects, a wide
fusion MLP is the easiest place in the whole architecture to overfit: it sees a
fully-connected view of every branch at once. Defaults are
``hidden_dim=128 -> out_dim=64``, with dropout before each linear layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from modules.common.config import FusionConfig
from modules.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FusionOutput:
    """Result of the fusion head."""

    #: ``(B, out_dim)`` shared representation ``Z_H``.
    z_h: torch.Tensor
    #: ``(B, spatial_dim)`` projected 3D CNN branch ``Z_3D``.
    z_3d: Optional[torch.Tensor] = None
    #: ``(B, graph_dim)`` graph branch ``Z_G``.
    z_g: Optional[torch.Tensor] = None
    #: ``(B, spatial_dim + graph_dim)`` concatenated pre-MLP vector ``Z_F``.
    z_f: Optional[torch.Tensor] = None

    def statistics(self, batch_index: int = 0) -> Dict[str, Any]:
        """Return descriptive statistics of the latent vectors for display."""
        def stats(t: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
            if t is None:
                return None
            v = t[batch_index].detach().cpu()
            return {
                "dim": int(v.numel()),
                "mean": float(v.mean()),
                "std": float(v.std()) if v.numel() > 1 else 0.0,
                "min": float(v.min()),
                "max": float(v.max()),
                "l2_norm": float(v.norm()),
            }
        return {
            "Z_3D": stats(self.z_3d),
            "Z_G": stats(self.z_g),
            "Z_F": stats(self.z_f),
            "Z_H": stats(self.z_h),
        }


class MultimodalFusion(nn.Module):
    """Fuse the spatial and graph branches into the shared representation ``Z_H``.

    Args:
        spatial_dim: Width of ``Z_3D``. Pass ``0`` to run graph-only (used by
            ablations A0-A5, which have no CNN branch).
        graph_dim: Width of ``Z_G``. Pass ``0`` to run spatial-only.
        cfg: Fusion hyper-parameters.

    Shape:
        ``(B, spatial_dim)``, ``(B, graph_dim)`` -> ``(B, out_dim)``.
    """

    def __init__(
        self,
        spatial_dim: int,
        graph_dim: int,
        cfg: Optional[FusionConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.spatial_dim = spatial_dim
        self.graph_dim = graph_dim
        fused_in = spatial_dim + graph_dim
        if fused_in == 0:
            raise ValueError(
                "MultimodalFusion needs at least one branch: both spatial_dim "
                "and graph_dim are zero."
            )
        self.fused_in = fused_in
        self.out_dim = self.cfg.out_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(fused_in, self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z_3d: Optional[torch.Tensor] = None,
        z_g: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """Fuse the branches.

        Args:
            z_3d: ``(B, spatial_dim)`` spatial branch, or ``None`` when
                ``spatial_dim == 0``.
            z_g: ``(B, graph_dim)`` graph branch, or ``None`` when
                ``graph_dim == 0``.

        Returns:
            A :class:`FusionOutput`.

        Raises:
            ValueError: If a configured branch is missing or mis-shaped.
        """
        parts = []
        if self.spatial_dim:
            if z_3d is None:
                raise ValueError(
                    f"spatial_dim={self.spatial_dim} but z_3d is None."
                )
            if z_3d.shape[-1] != self.spatial_dim:
                raise ValueError(
                    f"z_3d width {z_3d.shape[-1]} != spatial_dim "
                    f"{self.spatial_dim}"
                )
            parts.append(z_3d)
        if self.graph_dim:
            if z_g is None:
                raise ValueError(f"graph_dim={self.graph_dim} but z_g is None.")
            if z_g.shape[-1] != self.graph_dim:
                raise ValueError(
                    f"z_g width {z_g.shape[-1]} != graph_dim {self.graph_dim}"
                )
            parts.append(z_g)

        z_f = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        z_h = self.mlp(z_f)

        return FusionOutput(z_h=z_h, z_3d=z_3d, z_g=z_g, z_f=z_f)

    def summary(self) -> Dict[str, Any]:
        """Return a description of the fusion head."""
        return {
            "method": "concatenation followed by a two-layer MLP with LayerNorm "
                      "and dropout",
            "formula": "Z_F = [Z_3D || Z_G];  Z_H = MLP(Z_F)",
            "spatial_dim": self.spatial_dim,
            "graph_dim": self.graph_dim,
            "fused_dim": self.fused_in,
            "hidden_dim": self.cfg.hidden_dim,
            "out_dim": self.out_dim,
            "dropout": self.cfg.dropout,
            "shared_with": "current-stage classifier and Stage-TGT",
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (f"spatial_dim={self.spatial_dim}, graph_dim={self.graph_dim}, "
                f"out_dim={self.out_dim}")


__all__ = ["FusionOutput", "MultimodalFusion"]
