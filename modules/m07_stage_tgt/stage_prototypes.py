"""
Stage prototypes (Section 12).
==============================

Learns one prototype vector per disease stage in the shared representation space
``Z_H``:

.. math::

    c_{\\text{CN}},\\quad c_{\\text{MCI}},\\quad c_{\\text{AD}}

and, for a subject representation ``Z_H``, the squared distances and their
softmax similarity:

.. math::

    d_s = \\lVert Z_H - c_s \\rVert^2, \\qquad
    q_s = \\mathrm{softmax}(-d_s)

Why prototypes live in ``Z_H``
------------------------------

They occupy the *same* space the classifier decides in. That is what makes "this
subject's representation sits close to the AD prototype" and "this subject is
classified AD" two views of one geometry rather than two unrelated scores. If
the prototypes had their own projection, a high AD-associated propensity could
coexist with a confident CN classification and neither number would constrain
the other.

Temperature
-----------

``-d_s`` is divided by a learnable temperature before the softmax. Squared
distances in a 64-dimensional space are numerically large, and an unscaled
softmax over them saturates to a one-hot vector almost immediately — every
subject would report a propensity of 1.0 for its nearest stage and 0.0 for the
others, which is both uninformative and misleading. The temperature is stored as
a log-parameter so it stays strictly positive.

Ordering
--------

The prototypes are **not** constrained to be collinear or evenly spaced. The
ordered structure CN -> MCI -> AD is imposed instead by the ordinal
stage-order loss in :mod:`modules.training.losses`, which is a soft constraint
the data can push back on. Hard-wiring the geometry would build the desired
conclusion into the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER

logger = get_logger(__name__)


@dataclass
class PrototypeOutput:
    """Distances and similarities of a batch to the stage prototypes."""

    #: ``(B, N_STAGE)`` squared Euclidean distances ``d_s``.
    distances: torch.Tensor
    #: ``(B, N_STAGE)`` softmax similarities ``q_s``.
    similarities: torch.Tensor
    #: ``(N_STAGE, d_model)`` the prototype vectors themselves.
    prototypes: torch.Tensor
    #: Scalar temperature actually applied.
    temperature: torch.Tensor

    def distance_dict(self, batch_index: int = 0) -> Dict[str, float]:
        """Return ``{stage: squared distance}`` for one subject."""
        d = self.distances[batch_index].detach().cpu().numpy()
        return {s: float(d[i]) for i, s in enumerate(STAGE_ORDER)}

    def similarity_dict(self, batch_index: int = 0) -> Dict[str, float]:
        """Return ``{stage: similarity}`` for one subject."""
        q = self.similarities[batch_index].detach().cpu().numpy()
        return {s: float(q[i]) for i, s in enumerate(STAGE_ORDER)}

    def nearest_stage(self, batch_index: int = 0) -> str:
        """Return the stage whose prototype is closest."""
        return STAGE_ORDER[int(self.distances[batch_index].argmin())]


class StagePrototypes(nn.Module):
    """Learnable per-stage prototype vectors with distance-based similarity.

    Args:
        d_model: Width of the representation space (that of ``Z_H``).
        n_stages: Number of ordered stages.
        init_scale: Standard deviation of the prototype initialisation.
        learn_temperature: Learn the softmax temperature.
        init_temperature: Initial temperature. Defaults to ``sqrt(d_model)``
            when ``None``, matching the scale at which squared distances grow
            with dimensionality.
    """

    def __init__(
        self,
        d_model: int,
        n_stages: int = N_STAGE,
        init_scale: float = 0.5,
        learn_temperature: bool = True,
        init_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages

        self.prototypes = nn.Parameter(torch.empty(n_stages, d_model))
        nn.init.normal_(self.prototypes, mean=0.0, std=init_scale)

        temp = float(init_temperature) if init_temperature is not None \
            else float(d_model) ** 0.5
        log_temp = torch.log(torch.tensor(max(temp, 1e-3)))
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp, persistent=True)

    def temperature(self) -> torch.Tensor:
        """Return the strictly positive softmax temperature."""
        return self.log_temperature.exp()

    def forward(self, z_h: torch.Tensor) -> PrototypeOutput:
        """Compute distances and similarities to every stage prototype.

        Args:
            z_h: ``(B, d_model)`` shared representation.

        Returns:
            A :class:`PrototypeOutput`.

        Raises:
            ValueError: On a width mismatch.
        """
        if z_h.dim() == 1:
            z_h = z_h.unsqueeze(0)
        if z_h.shape[-1] != self.d_model:
            raise ValueError(
                f"Z_H width {z_h.shape[-1]} != prototype d_model {self.d_model}"
            )

        # (B, 1, D) - (1, S, D) -> (B, S, D) -> (B, S)
        diff = z_h.unsqueeze(1) - self.prototypes.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)

        temp = self.temperature()
        similarities = torch.softmax(-distances / temp.clamp_min(1e-6), dim=-1)

        return PrototypeOutput(
            distances=distances,
            similarities=similarities,
            prototypes=self.prototypes,
            temperature=temp,
        )

    # ── Geometry introspection ────────────────────────────────────────────

    def prototype_geometry(self) -> Dict[str, Any]:
        """Describe the learned prototype arrangement.

        The ``ordering_respected`` flag reports whether the learned geometry
        actually places MCI between CN and AD, i.e. whether
        ``||c_CN - c_MCI||`` and ``||c_MCI - c_AD||`` are each shorter than
        ``||c_CN - c_AD||``. This is a *measured property of the trained model*,
        not an architectural guarantee — reporting it is how the ordinal
        assumption gets checked rather than assumed.
        """
        with torch.no_grad():
            protos = self.prototypes.detach().cpu()
            pairwise: Dict[str, float] = {}
            for i, a in enumerate(STAGE_ORDER):
                for j, b in enumerate(STAGE_ORDER):
                    if i < j:
                        pairwise[f"{a}-{b}"] = float(
                            torch.norm(protos[i] - protos[j])
                        )
            d_cn_mci = pairwise.get("CN-MCI", 0.0)
            d_mci_ad = pairwise.get("MCI-AD", 0.0)
            d_cn_ad = pairwise.get("CN-AD", 0.0)
            ordered = (d_cn_mci < d_cn_ad) and (d_mci_ad < d_cn_ad)
            return {
                "pairwise_distances": pairwise,
                "norms": {
                    s: float(torch.norm(protos[i]))
                    for i, s in enumerate(STAGE_ORDER)
                },
                "ordering_respected": bool(ordered),
                "ordering_check": (
                    "MCI lies between CN and AD when both adjacent distances "
                    "are shorter than the CN-AD distance."
                ),
                "temperature": float(self.temperature().detach().cpu()),
                "d_model": self.d_model,
            }

    def summary(self) -> Dict[str, Any]:
        """Return a description of the module."""
        return {
            "formula": "d_s = ||Z_H - c_s||^2 ;  q_s = softmax(-d_s / T)",
            "stages": list(STAGE_ORDER),
            "d_model": self.d_model,
            "temperature_is_learned": isinstance(self.log_temperature, nn.Parameter),
            "space": "shared representation Z_H (same space the classifier "
                     "decides in)",
            "ordering_constraint": "imposed softly by the ordinal stage-order "
                                   "loss, not hard-wired into the geometry",
            "geometry": self.prototype_geometry(),
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"n_stages={self.n_stages}, d_model={self.d_model}"


__all__ = ["PrototypeOutput", "StagePrototypes"]
