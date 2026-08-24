"""
Stage-transition propensity head (Section 12).
==============================================

Turns the Stage-TGT output into the three quantities the design asks for. All
three are bounded in ``[0, 1]`` and each has an explicit definition, because a
propensity score with no stated definition cannot be reviewed.

1. Stage alignment ``q~``
-------------------------

A learned linear head on ``Z_T`` produces a softmax distribution over the three
ordered stages. This is the head's own view of which *stage representation* the
subject resembles, trained as an auxiliary objective on the same labels as the
classifier. It is deliberately a separate head: if it simply reused the
classifier's probabilities it would add no information.

2. Advanced-stage alignment
---------------------------

The expected ordinal position under ``q~``:

.. math::

    \\text{advanced} = \\sum_s \\frac{\\mathrm{idx}(s)}{S - 1}\\, \\tilde{q}_s
    \\;\\in\\; [0, 1]

0 means fully CN-aligned, 1 fully AD-aligned, 0.5 squarely MCI-aligned.

3. AD-associated propensity
---------------------------

A **geometric** quantity: how close the subject's representation sits to the AD
prototype *relative to* the CN prototype.

.. math::

    t = \\frac{\\lVert Z_H - c_{\\text{CN}} \\rVert}
             {\\lVert Z_H - c_{\\text{CN}} \\rVert
              + \\lVert Z_H - c_{\\text{AD}} \\rVert}

``t = 0`` on the CN prototype, ``1`` on the AD prototype, ``0.5`` equidistant.

**Why this form rather than a projection onto the CN-to-AD axis.** The obvious
alternative is the scalar projection below. It was implemented first and
measured, and it fails in a way worth recording: it is sensitive to any
common-mode offset of the representation cloud away from the prototype simplex.
On a trained model whose class centroids sat ~8.5 away from their own prototypes
while the prototypes were only ~5.1 apart, the projection returned 0.87 for CN
subjects and 0.69 for AD subjects -- inverted -- because the shared offset
dominated the class-dependent component. The ratio-of-distances form is
invariant to that shared offset by construction, since the offset enters both
the numerator and the denominator. The projection is still reported alongside,
as ``ad_axis_projection``, because it remains a useful diagnostic of how well the
prototype geometry has actually trained.

The superseded axis form, for reference:

.. math::

    t = \\mathrm{clip}\\!\\left(
        \\frac{\\langle Z_H - c_{\\text{CN}},\\; c_{\\text{AD}} - c_{\\text{CN}}
        \\rangle}{\\lVert c_{\\text{AD}} - c_{\\text{CN}} \\rVert^2},\\; 0,\\; 1
    \\right)

This is also why AD-associated propensity can differ substantially from
``P(AD)`` for the same subject, which is the intended behaviour. ``P(AD)`` asks
"which of three classes does this subject belong to"; ``t`` asks "how far toward
the AD representation does this subject already sit". An MCI subject can be
confidently *not* AD (low ``P(AD)``) while sitting well past the midpoint between
the CN and AD prototypes (high ``t``). The two answer different questions and are
reported separately for exactly that reason.

Neither quantity has parameters of its own — both are read off the learned
prototype geometry — so neither can be tuned to produce a desired-looking
number.

4. Stage-transition propensity
------------------------------

Given the predicted stage ``k``, the alignment mass at strictly more advanced
stages, normalised against the predicted stage itself:

.. math::

    \\text{transition} = \\frac{\\sum_{s > k} \\tilde{q}_s}
                              {\\tilde{q}_k + \\sum_{s > k} \\tilde{q}_s}

For a predicted stage of AD there is no more advanced stage, so the quantity is
**undefined** and is reported as ``None`` with the advanced-stage alignment used
in its place. Emitting a number there would be inventing one.

Language (Sections 12, 32)
--------------------------

None of these is a conversion probability, a guaranteed future diagnosis, or a
biological estimate of onset. Every accessor carries the disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_INDEX, STAGE_ORDER
from modules.m07_stage_tgt.stage_transformer import StageTransformer

logger = get_logger(__name__)


@dataclass
class PropensityOutput:
    """Stage-propensity quantities for a batch."""

    #: ``(B, N_STAGE)`` raw alignment logits.
    alignment_logits: torch.Tensor
    #: ``(B, N_STAGE)`` alignment distribution ``q~``.
    alignment: torch.Tensor
    #: ``(B,)`` expected ordinal position in ``[0, 1]``.
    advanced_stage_alignment: torch.Tensor
    #: ``(B,)`` relative proximity to the AD prototype versus the CN
    #: prototype, in ``[0, 1]``. This is the reported AD-associated propensity.
    ad_associated_propensity: torch.Tensor
    #: ``(B,)`` transition propensity relative to the predicted stage. Entries
    #: are ``NaN`` where the predicted stage is the most advanced one.
    stage_transition_propensity: torch.Tensor
    #: ``(B,)`` predicted stage indices these quantities are relative to.
    reference_stage: torch.Tensor
    #: ``(B,)`` the alternative CN-to-AD axis projection, retained as a
    #: diagnostic of prototype-geometry quality (see the module docstring).
    ad_axis_projection: Optional[torch.Tensor] = None

    def alignment_dict(self, batch_index: int = 0) -> Dict[str, float]:
        """Return ``{stage: alignment}`` for one subject."""
        q = self.alignment[batch_index].detach().cpu().numpy()
        return {s: float(q[i]) for i, s in enumerate(STAGE_ORDER)}

    def report(self, batch_index: int = 0) -> Dict[str, Any]:
        """Return a complete, self-describing propensity record for one subject.

        ``stage_transition_propensity`` is ``None`` — never a number — when the
        reference stage is the most advanced one.
        """
        idx = int(self.reference_stage[batch_index])
        ref_stage = STAGE_ORDER[idx]
        transition = float(self.stage_transition_propensity[batch_index].detach())
        has_next = idx < N_STAGE - 1
        next_stage = STAGE_ORDER[idx + 1] if has_next else None

        return {
            "reference_stage": ref_stage,
            "stage_alignment": self.alignment_dict(batch_index),
            "advanced_stage_alignment": float(
                self.advanced_stage_alignment[batch_index].detach()
            ),
            "ad_associated_propensity": float(
                self.ad_associated_propensity[batch_index].detach()
            ),
            "ad_axis_projection": (
                float(self.ad_axis_projection[batch_index].detach())
                if self.ad_axis_projection is not None else None
            ),
            "stage_transition_propensity": (
                transition if has_next and transition == transition else None
            ),
            "transition_target_stage": next_stage,
            "transition_note": (
                f"Propensity of the current representation toward the "
                f"{next_stage}-associated representation."
                if has_next else
                "The reference stage is the most advanced stage in the "
                "vocabulary, so stage-transition propensity is undefined. "
                "Advanced-stage alignment is reported in its place."
            ),
            "definitions": {
                "advanced_stage_alignment":
                    "Expected ordinal position under the stage-alignment "
                    "distribution: 0 = fully CN-aligned, 1 = fully AD-aligned.",
                "ad_associated_propensity":
                    "d_CN / (d_CN + d_AD): distance to the CN prototype "
                    "relative to distance to the AD prototype. 0 = on the CN "
                    "prototype, 1 = on the AD prototype. Answers 'how far "
                    "toward the AD representation', which is a different "
                    "question from the classifier's P(AD).",
                "ad_axis_projection":
                    "Diagnostic only: the scalar projection onto the CN-to-AD "
                    "axis. Sensitive to a common-mode offset of the "
                    "representation cloud, so it is not the reported "
                    "propensity.",
                "stage_transition_propensity":
                    "Alignment mass at stages strictly more advanced than the "
                    "reference stage, normalised against the reference stage.",
            },
            "disclaimer": StageTransformer.disclaimer(),
        }


class PropensityHead(nn.Module):
    """Derive stage-propensity quantities from the Stage-TGT output.

    Args:
        d_model: Width of ``Z_T``.
        n_stages: Number of ordered stages.
        dropout: Dropout before the alignment head.
    """

    def __init__(self, d_model: int, n_stages: int = N_STAGE,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages
        self.dropout = nn.Dropout(dropout)
        self.alignment = nn.Linear(d_model, n_stages)
        nn.init.xavier_uniform_(self.alignment.weight)
        nn.init.zeros_(self.alignment.bias)

        # Ordinal position weights: 0, 0.5, 1 for three stages. A buffer rather
        # than a literal so the head stays correct if the stage vocabulary grows.
        positions = torch.linspace(0.0, 1.0, steps=n_stages)
        self.register_buffer("ordinal_positions", positions, persistent=True)

    # ── Geometry ──────────────────────────────────────────────────────────

    @staticmethod
    def relative_ad_proximity(z_h: torch.Tensor, prototypes: torch.Tensor,
                              eps: float = 1e-12) -> torch.Tensor:
        """Relative proximity to the AD prototype versus the CN prototype.

        ``d_CN / (d_CN + d_AD)`` using Euclidean distances. Bounded in
        ``[0, 1]`` by construction, and invariant to a common-mode offset of the
        representation cloud because the offset affects both distances.

        Args:
            z_h: ``(B, D)`` shared representations.
            prototypes: ``(N_STAGE, D)`` learned stage prototypes.
            eps: Floor on the denominator, for the degenerate case where both
                prototypes coincide with the subject.

        Returns:
            ``(B,)`` values in ``[0, 1]``.
        """
        c_cn = prototypes[STAGE_INDEX["CN"]]
        c_ad = prototypes[STAGE_INDEX["AD"]]
        d_cn = (z_h - c_cn).norm(dim=-1)
        d_ad = (z_h - c_ad).norm(dim=-1)
        return (d_cn / (d_cn + d_ad).clamp_min(eps)).clamp(0.0, 1.0)

    @staticmethod
    def ad_axis_projection(z_h: torch.Tensor, prototypes: torch.Tensor,
                           eps: float = 1e-12) -> torch.Tensor:
        """Project representations onto the CN-to-AD prototype axis.

        Reported as a diagnostic rather than as the propensity score: it is
        sensitive to a common-mode offset between the representation cloud and
        the prototype simplex, and can invert when the prototypes sit further
        from their own class centroids than from each other.

        Args:
            z_h: ``(B, D)`` shared representations.
            prototypes: ``(N_STAGE, D)`` learned stage prototypes.
            eps: Floor on the squared axis length.

        Returns:
            ``(B,)`` scalar positions clipped to ``[0, 1]``.
        """
        c_cn = prototypes[STAGE_INDEX["CN"]]
        c_ad = prototypes[STAGE_INDEX["AD"]]
        axis = c_ad - c_cn
        denom = (axis * axis).sum().clamp_min(eps)
        t = ((z_h - c_cn) * axis).sum(dim=-1) / denom
        return t.clamp(0.0, 1.0)

    def _transition(self, alignment: torch.Tensor,
                    reference: torch.Tensor) -> torch.Tensor:
        """Compute transition propensity relative to a reference stage.

        Returns ``NaN`` for rows whose reference stage is the most advanced one,
        so that :meth:`PropensityOutput.report` can surface ``None`` instead of
        fabricating a value.
        """
        b, s = alignment.shape
        idx = torch.arange(s, device=alignment.device).unsqueeze(0).expand(b, s)
        ref = reference.view(b, 1)

        ahead = (idx > ref).float() * alignment
        at_ref = (idx == ref).float() * alignment
        numer = ahead.sum(dim=1)
        denom = numer + at_ref.sum(dim=1)

        out = torch.where(
            denom > 0, numer / denom.clamp_min(1e-12), torch.zeros_like(numer)
        )
        is_last = reference >= (s - 1)
        return out.masked_fill(is_last, float("nan"))

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        z_t: torch.Tensor,
        z_h: torch.Tensor,
        prototypes: torch.Tensor,
        reference_stage: Optional[torch.Tensor] = None,
    ) -> PropensityOutput:
        """Compute all stage-propensity quantities.

        Args:
            z_t: ``(B, d_model)`` Stage-TGT output.
            z_h: ``(B, D)`` shared representation, used for the geometric
                projection.
            prototypes: ``(N_STAGE, D)`` learned stage prototypes.
            reference_stage: ``(B,)`` predicted stage indices. When ``None``,
                the alignment head's own argmax is used — but the classifier's
                prediction should be passed in practice, so that the reported
                transition is relative to the stage the system actually
                predicts.

        Returns:
            A :class:`PropensityOutput`.

        Raises:
            ValueError: On a width mismatch.
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if z_h.dim() == 1:
            z_h = z_h.unsqueeze(0)
        if z_t.shape[-1] != self.d_model:
            raise ValueError(
                f"Z_T width {z_t.shape[-1]} != head d_model {self.d_model}"
            )
        if prototypes.shape[-1] != z_h.shape[-1]:
            raise ValueError(
                f"Prototype width {prototypes.shape[-1]} != Z_H width "
                f"{z_h.shape[-1]}; the CN-to-AD projection requires both to "
                "live in the same space."
            )

        logits = self.alignment(self.dropout(z_t))
        q = torch.softmax(logits, dim=-1)

        advanced = (q * self.ordinal_positions.unsqueeze(0)).sum(dim=-1)
        ad_prop = self.relative_ad_proximity(z_h, prototypes)
        ad_axis = self.ad_axis_projection(z_h, prototypes)

        ref = reference_stage if reference_stage is not None else q.argmax(dim=-1)
        ref = ref.to(dtype=torch.long, device=q.device)
        transition = self._transition(q, ref)

        return PropensityOutput(
            alignment_logits=logits,
            alignment=q,
            advanced_stage_alignment=advanced,
            ad_associated_propensity=ad_prop,
            ad_axis_projection=ad_axis,
            stage_transition_propensity=transition,
            reference_stage=ref,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the head."""
        return {
            "outputs": [
                "stage_alignment (learned softmax over ordered stages)",
                "advanced_stage_alignment (expected ordinal position)",
                "ad_associated_propensity (relative proximity to the AD "
                "prototype versus the CN prototype)",
                "ad_axis_projection (diagnostic)",
                "stage_transition_propensity (alignment mass beyond the "
                "reference stage)",
            ],
            "learned_parameters": "the alignment head only; both geometric "
                                  "quantities are read off the prototype "
                                  "geometry and have no parameters of their own",
            "undefined_case": "stage_transition_propensity is reported as None "
                              "when the reference stage is AD, the most "
                              "advanced stage in the vocabulary",
            "d_model": self.d_model,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "disclaimer": StageTransformer.disclaimer(),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"d_model={self.d_model}, n_stages={self.n_stages}"


__all__ = ["PropensityOutput", "PropensityHead"]
