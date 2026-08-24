"""
M11.1 — SRVE: Stage-Aware Regional Vulnerability Estimation (Section 8.1).
=========================================================================

Estimates, for each speech-related ROI, a scalar in ``[0, 1]`` describing how
informative that region is for discriminating the current CN/MCI/AD stage:

.. math::

    RV_i = \\sigma\\!\\left(w_v^\\top \\hat{x}_i + b_v\\right)

where :math:`\\hat{x}_i` is the normalized node feature vector of ROI ``i``.

Design notes
------------

**One shared ``w_v`` across ROIs, plus a per-ROI bias.** A separate weight
vector per region would multiply the parameter count by five and let each ROI
fit its own idiosyncratic direction on 154 training subjects. Sharing ``w_v``
forces a single, interpretable "what does structural degradation look like"
direction in feature space, which is what makes the resulting scores comparable
*between* regions — the comparison the ROI ranking depends on. A per-ROI bias
``b_v[i]`` still lets each region carry its own baseline level.

**These are learned, not hand-computed.** The pre-refactor implementation
derived vulnerability from hard-coded normative volumes and magic constants and
had no trainable parameters at all. Here ``w_v`` and ``b_v`` are optimised by
the classification objective, so vulnerability means "the model found this
region discriminative", which is a claim the data can support.

**Wording.** :math:`RV_i` is a model-derived discriminative vulnerability
score. It is not a biological probability of future degeneration. See
:meth:`~modules.m06_neuropropx.types.NeuroPropXOutput.disclaimer`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER
from modules.m06_neuropropx.types import SRVEOutput

logger = get_logger(__name__)


class SRVE(nn.Module):
    """Stage-Aware Regional Vulnerability Estimation.

    Args:
        in_dim: Width of the normalized node feature vector ``xhat_i``.
        n_roi: Number of ROIs.
        per_roi_bias: Give each ROI its own bias term. Recommended: it lets a
            region carry a baseline vulnerability level without needing its own
            weight vector.
        feature_names: Optional names of the input features, stored so the
            dashboard can show the learned weight attached to each feature.

    Shape:
        input ``(B, n_roi, in_dim)`` -> ``RV`` of shape ``(B, n_roi)``.
    """

    def __init__(
        self,
        in_dim: int,
        n_roi: int = N_ROI,
        per_roi_bias: bool = True,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")

        self.in_dim = in_dim
        self.n_roi = n_roi
        self.per_roi_bias = per_roi_bias
        self.feature_names = list(feature_names) if feature_names else None

        if self.feature_names and len(self.feature_names) != in_dim:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but "
                f"in_dim is {in_dim}; the weight-to-feature mapping shown in "
                "the dashboard would be wrong."
            )

        #: Shared vulnerability direction ``w_v``, shape ``(in_dim,)``.
        self.w_v = nn.Parameter(torch.zeros(in_dim))
        if per_roi_bias:
            self.b_v = nn.Parameter(torch.zeros(n_roi))
        else:
            self.b_v = nn.Parameter(torch.zeros(1))

        # Small random init: a zero w_v would make every RV exactly 0.5 with an
        # identical gradient for all features, and the symmetry never breaks.
        nn.init.normal_(self.w_v, mean=0.0, std=1.0 / max(1.0, in_dim ** 0.5))

    def forward(self, x: torch.Tensor,
                keep_features: bool = False) -> SRVEOutput:
        """Compute regional vulnerability.

        Args:
            x: ``(B, n_roi, in_dim)`` normalized node features. A single
                unbatched ``(n_roi, in_dim)`` tensor is promoted to batch 1.
            keep_features: Retain the input in the output for dashboard
                traceability. Off during training to avoid holding a reference
                to every batch's features.

        Returns:
            An :class:`~modules.m06_neuropropx.types.SRVEOutput`.

        Raises:
            ValueError: On an ROI-axis or feature-width mismatch.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(
                f"Expected (B, n_roi, in_dim) or (n_roi, in_dim); got "
                f"{tuple(x.shape)}"
            )
        b, n, f = x.shape
        if n != self.n_roi:
            raise ValueError(
                f"ROI axis is {n} but SRVE was built for {self.n_roi} ROIs "
                f"({ROI_ORDER})."
            )
        if f != self.in_dim:
            raise ValueError(
                f"Feature width is {f} but SRVE was built for {self.in_dim}. "
                "A width change means the scaler or feature set was rebuilt; "
                "re-instantiate SRVE rather than reusing a stale one."
            )

        # (B, N, F) . (F,) -> (B, N)
        logits = torch.einsum("bnf,f->bn", x, self.w_v)
        logits = logits + (self.b_v if self.per_roi_bias else self.b_v.expand(n))
        rv = torch.sigmoid(logits)

        return SRVEOutput(
            regional_vulnerability=rv,
            logits=logits,
            normalized_features=x if keep_features else None,
        )

    # ── Introspection for the dashboard (Section 52) ───────────────────────

    def learned_weights(self) -> Dict[str, float]:
        """Return ``w_v`` keyed by feature name, sorted by descending magnitude.

        Section 52 asks for the actual learned weights to be displayed so that a
        reviewer can trace feature -> calculation -> vulnerability.
        """
        w = self.w_v.detach().cpu().numpy()
        names = self.feature_names or [f"f{i}" for i in range(self.in_dim)]
        pairs = {names[i]: float(w[i]) for i in range(self.in_dim)}
        return dict(sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def learned_bias(self) -> Dict[str, float]:
        """Return the per-ROI bias term ``b_v`` keyed by ROI name."""
        b = self.b_v.detach().cpu().numpy().reshape(-1)
        if self.per_roi_bias:
            return {roi: float(b[i]) for i, roi in enumerate(ROI_ORDER)}
        return {roi: float(b[0]) for roi in ROI_ORDER}

    def explain(self, x: torch.Tensor, batch_index: int = 0,
                roi_index: int = 0) -> Dict[str, object]:
        """Decompose one ROI's vulnerability into per-feature contributions.

        The contribution of feature ``k`` is ``w_v[k] * xhat_i[k]``; those terms
        plus the bias sum exactly to the pre-sigmoid logit, so the decomposition
        is exact rather than an approximation.

        Args:
            x: ``(B, n_roi, in_dim)`` normalized features.
            batch_index: Subject within the batch.
            roi_index: ROI within :data:`ROI_ORDER`.

        Returns:
            A dict with the feature vector, per-feature contributions, the bias,
            the logit and the resulting ``RV``.
        """
        with torch.no_grad():
            out = self.forward(x, keep_features=True)
            feats = out.normalized_features[batch_index, roi_index]
            w = self.w_v
            contrib = (w * feats).cpu().numpy()
            names = self.feature_names or [f"f{i}" for i in range(self.in_dim)]
            bias = self.learned_bias()[ROI_ORDER[roi_index]]
            logit = float(out.logits[batch_index, roi_index])
            return {
                "roi": ROI_ORDER[roi_index],
                "features": {names[i]: float(feats[i]) for i in range(self.in_dim)},
                "weights": {names[i]: float(w[i]) for i in range(self.in_dim)},
                "contributions": {
                    names[i]: float(contrib[i]) for i in range(self.in_dim)
                },
                "bias": bias,
                "logit": logit,
                "regional_vulnerability": float(
                    out.regional_vulnerability[batch_index, roi_index]
                ),
                "formula": "RV_i = sigmoid(w_v^T xhat_i + b_v)",
                "reconstruction_check": float(contrib.sum() + bias),
            }

    def extra_repr(self) -> str:
        """Torch module repr."""
        return (f"in_dim={self.in_dim}, n_roi={self.n_roi}, "
                f"per_roi_bias={self.per_roi_bias}")


class ConstantVulnerability(nn.Module):
    """SRVE replacement that emits a constant 0.5 for every ROI.

    Used by ablations A0-A3, which must run the rest of the pipeline with SRVE
    disabled. Returning a constant rather than skipping the module keeps every
    downstream tensor shape identical, so the ablation measures the removal of
    SRVE's *information* and not an incidental architectural change.

    The constant is 0.5 because that is ``sigmoid(0)``: the neutral point of the
    SRVE output range, carrying no vulnerability signal either way.
    """

    def __init__(self, n_roi: int = N_ROI, value: float = 0.5) -> None:
        super().__init__()
        self.n_roi = n_roi
        self.value = value

    def forward(self, x: torch.Tensor,
                keep_features: bool = False) -> SRVEOutput:
        """Return a constant vulnerability vector matching ``x``'s batch size."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        rv = torch.full((b, self.n_roi), self.value,
                        dtype=x.dtype, device=x.device)
        logits = torch.zeros_like(rv)
        return SRVEOutput(
            regional_vulnerability=rv,
            logits=logits,
            normalized_features=x if keep_features else None,
        )

    def learned_weights(self) -> Dict[str, float]:
        """No learned weights exist in the ablated variant."""
        return {}

    def learned_bias(self) -> Dict[str, float]:
        """Constant bias, reported for interface parity."""
        return {roi: 0.0 for roi in ROI_ORDER}

    def extra_repr(self) -> str:
        """Torch module repr."""
        return f"n_roi={self.n_roi}, value={self.value} (SRVE ablated)"


__all__ = ["SRVE", "ConstantVulnerability"]
