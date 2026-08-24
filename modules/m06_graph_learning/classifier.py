"""
Current-stage CN/MCI/AD classification head (Section 11).
========================================================

.. math:: p = \\mathrm{softmax}(W_c Z_H + b_c)

A single linear layer on the shared representation ``Z_H``. Kept linear
deliberately: the representation learning happens upstream, and a deeper head
would add capacity exactly where the label signal is thinnest (30 AD sessions).
A linear head also makes the classifier weights directly interpretable as a
per-class direction in ``Z_H``.

Uncertainty
-----------

:class:`StageClassifier` reports predictive entropy and margin alongside the
class probabilities, because Section 20 requires a confidence indicator in the
report. Both are computed from the softmax output:

* **normalised entropy** — ``H(p) / log(3)``, in ``[0, 1]``; 0 means a
  fully confident prediction, 1 means uniform over the three stages.
* **margin** — top probability minus runner-up; small margins mark subjects
  sitting on a decision boundary between adjacent stages.

Neither is a calibrated probability of being correct. Softmax outputs from a
model trained on 154 subjects with a class-weighted loss are not calibrated, and
:meth:`StageClassifier.uncertainty_note` states so wherever they are rendered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER

logger = get_logger(__name__)


@dataclass
class ClassificationOutput:
    """Result of the stage classifier."""

    #: ``(B, N_STAGE)`` raw logits.
    logits: torch.Tensor
    #: ``(B, N_STAGE)`` softmax probabilities.
    probabilities: torch.Tensor
    #: ``(B,)`` predicted class indices.
    predictions: torch.Tensor

    def probability_dict(self, batch_index: int = 0) -> Dict[str, float]:
        """Return ``{stage: probability}`` for one subject."""
        p = self.probabilities[batch_index].detach().cpu().numpy()
        return {stage: float(p[i]) for i, stage in enumerate(STAGE_ORDER)}

    def predicted_stage(self, batch_index: int = 0) -> str:
        """Return the argmax stage name for one subject."""
        return STAGE_ORDER[int(self.predictions[batch_index])]

    def confidence(self, batch_index: int = 0) -> Dict[str, Any]:
        """Return uncertainty indicators for one subject."""
        p = self.probabilities[batch_index].detach().cpu()
        sorted_p, _ = torch.sort(p, descending=True)
        entropy = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        return {
            "top_probability": float(sorted_p[0]),
            "margin": float(sorted_p[0] - sorted_p[1]),
            "entropy": entropy,
            "normalized_entropy": entropy / math.log(N_STAGE),
            "note": StageClassifier.uncertainty_note(),
        }


class StageClassifier(nn.Module):
    """Linear CN/MCI/AD classification head.

    Args:
        in_dim: Width of the shared representation ``Z_H``.
        n_classes: Number of stages.
        dropout: Dropout applied to ``Z_H`` before the linear layer.
    """

    def __init__(self, in_dim: int, n_classes: int = N_STAGE,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z_h: torch.Tensor) -> ClassificationOutput:
        """Classify the shared representation.

        Args:
            z_h: ``(B, in_dim)`` shared representation.

        Returns:
            A :class:`ClassificationOutput`.

        Raises:
            ValueError: On a width mismatch.
        """
        if z_h.dim() == 1:
            z_h = z_h.unsqueeze(0)
        if z_h.shape[-1] != self.in_dim:
            raise ValueError(
                f"Z_H width {z_h.shape[-1]} != classifier in_dim {self.in_dim}"
            )
        logits = self.fc(self.dropout(z_h))
        probs = F.softmax(logits, dim=-1)
        return ClassificationOutput(
            logits=logits,
            probabilities=probs,
            predictions=logits.argmax(dim=-1),
        )

    def class_directions(self) -> Dict[str, List[float]]:
        """Return the learned weight vector for each stage.

        Because the head is linear, ``W_c[s]`` is literally the direction in
        ``Z_H`` that increases the score for stage ``s``.
        """
        w = self.fc.weight.detach().cpu().numpy()
        return {stage: w[i].tolist() for i, stage in enumerate(STAGE_ORDER)}

    @staticmethod
    def uncertainty_note() -> str:
        """Return the mandatory caveat for the confidence indicators."""
        return (
            "Confidence indicators are derived from softmax outputs and are not "
            "calibrated probabilities of correctness. A low-entropy prediction "
            "indicates that the model separated the classes confidently in its "
            "own representation, not that the classification is verified."
        )

    def summary(self) -> Dict[str, Any]:
        """Return a description of the head."""
        return {
            "formula": "p = softmax(W_c Z_H + b_c)",
            "in_dim": self.in_dim,
            "classes": list(STAGE_ORDER),
            "depth": "single linear layer",
            "rationale": "Representation learning happens upstream; a deeper "
                         "head would add capacity where the label signal is "
                         "thinnest and would obscure interpretability.",
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"in_dim={self.in_dim}, n_classes={self.n_classes}"


__all__ = ["ClassificationOutput", "StageClassifier"]
