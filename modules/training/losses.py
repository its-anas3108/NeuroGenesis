"""
Loss functions (Section 13).
============================

.. math::

    L_{\\text{total}} = L_{\\text{cls}}
        + \\lambda_{\\text{proto}} L_{\\text{proto}}
        + \\lambda_{\\text{order}} L_{\\text{order}}
        + \\lambda_{\\text{align}} L_{\\text{align}}

Every component is documented below, and :class:`LossBreakdown` reports each
term separately at every epoch so that a total which stops improving can be
attributed to the term responsible.

``L_cls`` — classification
--------------------------

Class-weighted cross-entropy on the current-stage logits. Weights are
inverse-frequency, computed from the **training split only**. With CN 88 / MCI 46
/ AD 20 training sessions, unweighted cross-entropy reaches 57% accuracy by
predicting CN for everything, and would select exactly that model.

``L_proto`` — prototype consistency
-----------------------------------

.. math:: L_{\\text{proto}} = \\frac{1}{B}\\sum_b
          \\lVert Z_H^{(b)} - c_{y_b} \\rVert^2

Pulls each subject's representation toward its own stage prototype. This is what
gives the prototypes their meaning; without it they would drift to arbitrary
positions and the propensity geometry would be noise.

The distance is normalised by ``d_model`` so that the term's scale does not
depend on the representation width, which keeps ``lambda_proto`` transferable if
``fusion.out_dim`` is retuned.

``L_order`` — ordinal stage order
---------------------------------

Two margin hinges, both encoding CN < MCI < AD:

*Prototype geometry* — MCI must sit between CN and AD:

.. math::

    \\mathrm{relu}\\!\\left(\\lVert c_{\\text{CN}} - c_{\\text{MCI}}\\rVert^2 + m
    - \\lVert c_{\\text{CN}} - c_{\\text{AD}}\\rVert^2\\right)
    + \\mathrm{relu}\\!\\left(\\lVert c_{\\text{MCI}} - c_{\\text{AD}}\\rVert^2 + m
    - \\lVert c_{\\text{CN}} - c_{\\text{AD}}\\rVert^2\\right)

*Per-sample ranking* — for a CN subject, ``d_CN < d_MCI < d_AD``; for an AD
subject, ``d_AD < d_MCI < d_CN``. **No ranking constraint is applied to MCI
subjects**, because MCI lies between the extremes and there is no defensible
ordering of ``d_CN`` versus ``d_AD`` for them — imposing one would assert
something the stage scale does not say.

``L_align`` — Stage-TGT alignment
---------------------------------

Cross-entropy on the Stage-TGT alignment head. This term is an addition to the
design's three-term loss, and it is not optional: the alignment head is the only
consumer of the transformer output ``Z_T``, so without it the transformer gets no
gradient and every reported stage-alignment and stage-transition-propensity value
would come from randomly initialised weights. Class weights are shared with
``L_cls`` for the same imbalance reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common.config import LossConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_INDEX, STAGE_ORDER
from modules.model import ModelOutput

logger = get_logger(__name__)


@dataclass
class LossBreakdown:
    """Per-term loss values for one batch or epoch."""

    total: torch.Tensor
    cls: torch.Tensor
    proto: Optional[torch.Tensor] = None
    order: Optional[torch.Tensor] = None
    align: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        """Return the terms as plain floats for logging."""
        out = {"total": float(self.total.detach()), "cls": float(self.cls.detach())}
        for name in ("proto", "order", "align"):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value.detach())
        return out


class NeuroGenesisLoss(nn.Module):
    """The composite training objective.

    Args:
        cfg: Loss configuration (weights, margin, imbalance strategy).
        class_weights: Length-``N_STAGE`` inverse-frequency weights computed on
            the **training split only**. ``None`` means unweighted, which is only
            appropriate if ``cfg.imbalance == "none"``.
        d_model: Width of ``Z_H``, used to normalise the prototype distances.
    """

    def __init__(
        self,
        cfg: Optional[LossConfig] = None,
        class_weights: Optional[np.ndarray] = None,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.d_model = max(int(d_model), 1)

        if self.cfg.imbalance == "class_weighted":
            if class_weights is None:
                logger.warning(
                    "imbalance='class_weighted' but no class_weights were "
                    "supplied; falling back to unweighted cross-entropy. On the "
                    "OASIS-1 cohort this biases the model toward CN."
                )
                weights = None
            else:
                weights = torch.as_tensor(
                    np.asarray(class_weights, dtype=np.float32)
                )
                if weights.numel() != N_STAGE:
                    raise ValueError(
                        f"class_weights must have {N_STAGE} entries "
                        f"(one per stage), got {weights.numel()}"
                    )
        else:
            weights = None

        if weights is None:
            self.register_buffer("class_weights", torch.ones(N_STAGE))
            self.use_weights = False
        else:
            self.register_buffer("class_weights", weights)
            self.use_weights = True

    # ── Terms ─────────────────────────────────────────────────────────────

    def classification_loss(self, logits: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
        """Class-weighted cross-entropy on the current-stage logits."""
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights if self.use_weights else None,
            label_smoothing=self.cfg.label_smoothing,
        )

    def prototype_loss(self, distances: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        """Mean squared distance from each subject to its own stage prototype.

        Args:
            distances: ``(B, N_STAGE)`` squared distances to every prototype.
            targets: ``(B,)`` true stage indices.
        """
        own = distances.gather(1, targets.view(-1, 1)).squeeze(1)
        return (own / self.d_model).mean()

    def order_loss(self, prototypes: torch.Tensor, distances: torch.Tensor,
                   targets: torch.Tensor) -> torch.Tensor:
        """Margin-based ordinal constraint encoding CN < MCI < AD.

        Args:
            prototypes: ``(N_STAGE, D)`` learned prototypes.
            distances: ``(B, N_STAGE)`` squared distances.
            targets: ``(B,)`` true stage indices.
        """
        m = self.cfg.order_margin
        i_cn, i_mci, i_ad = (
            STAGE_INDEX["CN"], STAGE_INDEX["MCI"], STAGE_INDEX["AD"]
        )
        scale = float(self.d_model)

        # Prototype geometry: MCI must lie between CN and AD.
        d_cn_mci = ((prototypes[i_cn] - prototypes[i_mci]) ** 2).sum() / scale
        d_mci_ad = ((prototypes[i_mci] - prototypes[i_ad]) ** 2).sum() / scale
        d_cn_ad = ((prototypes[i_cn] - prototypes[i_ad]) ** 2).sum() / scale
        geometry = F.relu(d_cn_mci + m - d_cn_ad) + F.relu(d_mci_ad + m - d_cn_ad)

        # Per-sample ranking, extremes only.
        d = distances / scale
        terms: List[torch.Tensor] = []

        cn_mask = targets == i_cn
        if cn_mask.any():
            dc = d[cn_mask]
            terms.append(
                (F.relu(dc[:, i_cn] - dc[:, i_mci] + m)
                 + F.relu(dc[:, i_mci] - dc[:, i_ad] + m)).mean()
            )
        ad_mask = targets == i_ad
        if ad_mask.any():
            da = d[ad_mask]
            terms.append(
                (F.relu(da[:, i_ad] - da[:, i_mci] + m)
                 + F.relu(da[:, i_mci] - da[:, i_cn] + m)).mean()
            )
        # MCI subjects are deliberately unconstrained: the stage scale does not
        # order d_CN against d_AD for a subject that lies between them.

        ranking = (
            torch.stack(terms).mean() if terms
            else torch.zeros((), device=prototypes.device,
                             dtype=prototypes.dtype)
        )
        return geometry + ranking

    def alignment_loss(self, alignment_logits: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on the Stage-TGT alignment head."""
        return F.cross_entropy(
            alignment_logits,
            targets,
            weight=self.class_weights if self.use_weights else None,
        )

    # ── Composite ─────────────────────────────────────────────────────────

    def forward(self, output: ModelOutput,
                targets: torch.Tensor) -> LossBreakdown:
        """Compute the composite loss.

        Terms whose component is absent from the model variant are skipped
        rather than contributed as zero, so the reported breakdown distinguishes
        "this term is not part of this variant" from "this term evaluated to
        zero".

        Args:
            output: The model's forward output.
            targets: ``(B,)`` true stage indices.

        Returns:
            A :class:`LossBreakdown`.
        """
        targets = targets.to(dtype=torch.long, device=output.logits.device)
        cls = self.classification_loss(output.logits, targets)
        total = cls

        proto = order = align = None

        if output.stage_tgt is not None:
            proto_out = output.stage_tgt.prototype_output
            if self.cfg.lambda_proto:
                proto = self.prototype_loss(proto_out.distances, targets)
                total = total + self.cfg.lambda_proto * proto
            if self.cfg.lambda_order:
                order = self.order_loss(
                    proto_out.prototypes, proto_out.distances, targets
                )
                total = total + self.cfg.lambda_order * order

        if output.propensity is not None and self.cfg.lambda_align:
            align = self.alignment_loss(
                output.propensity.alignment_logits, targets
            )
            total = total + self.cfg.lambda_align * align

        return LossBreakdown(
            total=total, cls=cls, proto=proto, order=order, align=align
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the objective for the report."""
        return {
            "formula": "L_total = L_cls + lambda_proto*L_proto "
                       "+ lambda_order*L_order + lambda_align*L_align",
            "terms": {
                "L_cls": {
                    "definition": "class-weighted cross-entropy on the "
                                  "current-stage logits",
                    "weight": 1.0,
                },
                "L_proto": {
                    "definition": "mean squared distance from each subject to "
                                  "its own stage prototype, normalised by "
                                  "d_model",
                    "weight": self.cfg.lambda_proto,
                },
                "L_order": {
                    "definition": "margin hinges enforcing CN < MCI < AD in the "
                                  "prototype geometry and in the per-sample "
                                  "distance ranking for the extreme classes",
                    "weight": self.cfg.lambda_order,
                    "margin": self.cfg.order_margin,
                    "note": "MCI subjects are intentionally unconstrained in "
                            "the ranking term",
                },
                "L_align": {
                    "definition": "cross-entropy on the Stage-TGT alignment "
                                  "head; the transformer's only gradient path",
                    "weight": self.cfg.lambda_align,
                    "note": "an addition to the design's three-term loss, "
                            "required so the Stage-TGT branch is trained rather "
                            "than decorative",
                },
            },
            "imbalance_strategy": self.cfg.imbalance,
            "class_weights": (
                self.class_weights.detach().cpu().tolist()
                if self.use_weights else None
            ),
            "class_weight_source": "inverse frequency on the training split only",
            "label_smoothing": self.cfg.label_smoothing,
            "stages": list(STAGE_ORDER),
        }


__all__ = ["LossBreakdown", "NeuroGenesisLoss"]
