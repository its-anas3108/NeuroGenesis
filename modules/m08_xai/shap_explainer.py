"""
Feature attribution for the morphometric branch (Section 15A).
=============================================================

Provides SHAP values over the per-ROI morphometric features, with a documented
fallback when the ``shap`` package is unavailable.

Two backends, and the difference is stated rather than hidden
-------------------------------------------------------------

``shap`` installed
    ``shap.KernelExplainer`` over the model's predicted class probability as a
    function of the flattened ``(N_ROI x N_FEATURES)`` morphometric input, with a
    background set drawn from the **training split only**. This yields true
    Shapley values under the usual feature-independence approximation.

``shap`` absent
    An **exact permutation-importance** attribution is computed instead, and
    :attr:`AttributionResult.method` reports ``"permutation"`` so that no figure
    or table can label it "SHAP". Permutation importance is a genuine
    attribution method with genuine limitations — it measures marginal effect
    under feature shuffling, is unsigned by construction unless a signed variant
    is requested, and does not decompose additively the way Shapley values do.
    Both are reported honestly; neither is presented as the other.

This is deliberate. The pre-refactor code contained a function that generated
numbers from hard-coded arithmetic and labelled them "SHAP feature importance".
That is the specific failure this module exists to avoid.

Background sets and leakage
---------------------------

The background/reference distribution is drawn from the training split.
Attribution against a background that includes test subjects would let test-set
structure into the explanation of a test-set prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER, STAGE_ORDER, roi_short
from modules.m04_feature_extraction.feature_spec import FEATURE_ORDER

logger = get_logger(__name__)


def shap_available() -> bool:
    """Return whether the ``shap`` package can be imported."""
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class AttributionResult:
    """Per-feature attribution for one target class.

    Attributes:
        method: ``"shap_kernel"`` or ``"permutation"``. Consumers must display
            this; a permutation attribution may not be labelled SHAP.
        target_stage: Class the attribution explains.
        values: ``(N_ROI, N_FEATURES)`` attribution per ROI x feature.
        feature_names: Column names matching the second axis.
        roi_names: Row names matching the first axis.
        signed: Whether the values carry a direction (positive pushes the model
            toward the target class).
        n_background: Size of the background/reference set used.
        notes: Caveats to display alongside the values.
    """

    method: str
    target_stage: str
    values: np.ndarray
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_ORDER))
    roi_names: List[str] = field(default_factory=lambda: list(ROI_ORDER))
    signed: bool = True
    n_background: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def is_shap(self) -> bool:
        """True only for a genuine Shapley-value attribution."""
        return self.method == "shap_kernel"

    def per_roi(self) -> Dict[str, float]:
        """Aggregate attribution magnitude per ROI.

        Absolute values are summed across features because a region matters if it
        moves the prediction at all; signed cancellation across its own features
        would understate an influential region whose features push in opposite
        directions.
        """
        totals = np.abs(self.values).sum(axis=1)
        return {roi: float(totals[i]) for i, roi in enumerate(self.roi_names)}

    def per_feature(self) -> Dict[str, float]:
        """Aggregate attribution magnitude per feature, summed over ROIs."""
        totals = np.abs(self.values).sum(axis=0)
        return {f: float(totals[i]) for i, f in enumerate(self.feature_names)}

    def top_features(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the ``k`` largest individual ROI x feature attributions."""
        entries = [
            {
                "roi": self.roi_names[i],
                "roi_short": roi_short(self.roi_names[i]),
                "feature": self.feature_names[j],
                "value": float(self.values[i, j]),
                "magnitude": float(abs(self.values[i, j])),
                "direction": (
                    "increases" if self.values[i, j] > 0 else "decreases"
                ) if self.signed else "unsigned",
            }
            for i in range(self.values.shape[0])
            for j in range(self.values.shape[1])
        ]
        entries.sort(key=lambda e: e["magnitude"], reverse=True)
        return entries[:k]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "method": self.method,
            "is_shap": self.is_shap,
            "target_stage": self.target_stage,
            "signed": self.signed,
            "n_background": self.n_background,
            "roi_names": list(self.roi_names),
            "feature_names": list(self.feature_names),
            "values": self.values.tolist(),
            "per_roi": self.per_roi(),
            "per_feature": self.per_feature(),
            "top_features": self.top_features(10),
            "notes": list(self.notes),
        }


class FeatureAttributor:
    """Compute morphometric feature attributions for a trained model.

    Args:
        predict_fn: Maps ``(n_samples, N_ROI * N_FEATURES)`` flattened
            morphometric inputs to ``(n_samples, N_STAGE)`` class probabilities.
            The caller supplies this so the attributor never needs to know how
            the model handles patches, embeddings or devices.
        background: ``(n_background, N_ROI, N_FEATURES)`` training-split
            morphometric features used as the reference distribution.
        feature_names: Feature column names.
        roi_names: ROI row names.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background: np.ndarray,
        feature_names: Optional[List[str]] = None,
        roi_names: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> None:
        self._raw_predict_fn = predict_fn
        self.batch_size = max(int(batch_size), 1)
        self.predict_fn = self._chunked_predict
        self.feature_names = list(feature_names or FEATURE_ORDER)
        self.roi_names = list(roi_names or ROI_ORDER)

        background = np.asarray(background, dtype=np.float64)
        if background.ndim != 3:
            raise ValueError(
                "background must have shape (n, N_ROI, N_FEATURES); got "
                f"{background.shape}"
            )
        if background.shape[1] != len(self.roi_names):
            raise ValueError(
                f"background ROI axis is {background.shape[1]}, expected "
                f"{len(self.roi_names)}"
            )
        self.background = background
        self.n_roi = background.shape[1]
        self.n_feat = background.shape[2]
        self.flat_background = background.reshape(background.shape[0], -1)

    def _chunked_predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``predict_fn`` in fixed-size chunks.

        Attribution builds one row per (input, background-value) pair, which is
        ``N_ROI * N_FEATURES * n_background`` rows -- 700 at the defaults. Passing
        those to the model in one call allocated tens of gigabytes when the
        forward pass touched the 3D CNN. Chunking bounds peak memory
        independently of the attribution's fan-out.
        """
        rows = np.asarray(x, dtype=np.float32)
        if rows.shape[0] <= self.batch_size:
            return np.asarray(self._raw_predict_fn(rows), dtype=np.float64)
        parts = [
            np.asarray(self._raw_predict_fn(rows[i:i + self.batch_size]),
                       dtype=np.float64)
            for i in range(0, rows.shape[0], self.batch_size)
        ]
        return np.concatenate(parts, axis=0)

    # ── SHAP backend ──────────────────────────────────────────────────────

    def _shap_values(self, sample: np.ndarray, target_index: int,
                     n_background: int, nsamples: int) -> np.ndarray:
        """Compute Kernel SHAP values for one sample and target class."""
        import shap

        n = min(n_background, self.flat_background.shape[0])
        # k-means summarisation keeps KernelExplainer tractable; with 154
        # training subjects the full background would need 154 coalition
        # evaluations per perturbation.
        summary = shap.kmeans(self.flat_background, min(n, 10)) if n > 10 \
            else self.flat_background[:n]

        def f(x: np.ndarray) -> np.ndarray:
            return self.predict_fn(np.asarray(x, dtype=np.float32))[:, target_index]

        explainer = shap.KernelExplainer(f, summary)
        values = explainer.shap_values(
            sample.reshape(1, -1), nsamples=nsamples, silent=True
        )
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return array.reshape(self.n_roi, self.n_feat)

    # ── Permutation backend ───────────────────────────────────────────────

    def _permutation_values(self, sample: np.ndarray, target_index: int,
                            n_background: int) -> np.ndarray:
        """Compute exact single-feature replacement attributions.

        For each of the ``N_ROI x N_FEATURES`` inputs, the feature is replaced by
        each background value in turn and the mean change in the target-class
        probability is recorded. The result is signed — positive means the
        subject's actual value pushes the prediction toward the target class
        relative to the reference distribution — and needs no sampling, because
        the enumeration over a summarised background is exhaustive.
        """
        n = min(n_background, self.flat_background.shape[0])
        reference = self.flat_background[:n]
        flat = sample.reshape(1, -1).astype(np.float32)

        baseline = float(self.predict_fn(flat)[0, target_index])
        n_inputs = flat.shape[1]

        # Build every single-feature substitution in one batch: n_inputs * n
        # rows, which is 70 * 10 = 700 forward passes on the default settings.
        perturbed = np.repeat(flat, n_inputs * n, axis=0)
        for idx in range(n_inputs):
            start = idx * n
            perturbed[start:start + n, idx] = reference[:, idx]

        probs = self.predict_fn(perturbed)[:, target_index]
        deltas = np.array([
            baseline - float(probs[idx * n:(idx + 1) * n].mean())
            for idx in range(n_inputs)
        ])
        return deltas.reshape(self.n_roi, self.n_feat)

    # ── Public ────────────────────────────────────────────────────────────

    def explain(
        self,
        sample: np.ndarray,
        target_stage: str,
        n_background: int = 20,
        nsamples: int = 200,
        force_permutation: bool = False,
    ) -> AttributionResult:
        """Attribute one subject's prediction to its morphometric features.

        Args:
            sample: ``(N_ROI, N_FEATURES)`` standardised features for one subject.
            target_stage: Stage whose probability is being explained.
            n_background: Background samples to use.
            nsamples: Kernel SHAP coalition samples. Ignored by the permutation
                backend, which is exhaustive.
            force_permutation: Use the permutation backend even if ``shap`` is
                available. Useful for a like-for-like comparison of the two.

        Returns:
            An :class:`AttributionResult` whose ``method`` names the backend
            actually used.

        Raises:
            ValueError: If ``target_stage`` is unknown or the sample is
                mis-shaped.
        """
        if target_stage not in STAGE_ORDER:
            raise ValueError(
                f"Unknown stage {target_stage!r}; expected one of {STAGE_ORDER}"
            )
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape != (self.n_roi, self.n_feat):
            raise ValueError(
                f"sample must have shape ({self.n_roi}, {self.n_feat}); got "
                f"{sample.shape}"
            )
        target_index = STAGE_ORDER.index(target_stage)
        notes: List[str] = []

        use_shap = shap_available() and not force_permutation
        if use_shap:
            try:
                values = self._shap_values(
                    sample, target_index, n_background, nsamples
                )
                method = "shap_kernel"
                notes.append(
                    "Kernel SHAP with a k-means-summarised background drawn "
                    "from the training split. Shapley values assume feature "
                    "independence, which correlated morphometric features "
                    "violate to some degree."
                )
            except Exception as exc:  # noqa: BLE001 - backend may fail variously
                logger.warning(
                    "Kernel SHAP failed (%s); falling back to permutation "
                    "attribution.", exc
                )
                values = self._permutation_values(
                    sample, target_index, n_background
                )
                method = "permutation"
                notes.append(f"Kernel SHAP failed and was not used: {exc}")
        else:
            values = self._permutation_values(sample, target_index, n_background)
            method = "permutation"
            if force_permutation:
                notes.append("Permutation attribution requested explicitly.")
            else:
                notes.append(
                    "The `shap` package is not installed, so exact "
                    "single-feature replacement (permutation) attribution was "
                    "used instead. These are NOT Shapley values: they measure "
                    "the marginal effect of replacing one feature with its "
                    "training-split reference values and do not decompose "
                    "additively. Install `shap` for Shapley attribution."
                )

        return AttributionResult(
            method=method,
            target_stage=target_stage,
            values=values,
            feature_names=list(self.feature_names),
            roi_names=list(self.roi_names),
            signed=True,
            n_background=min(n_background, self.flat_background.shape[0]),
            notes=notes,
        )

    def explain_all_stages(
        self, sample: np.ndarray, **kwargs: Any
    ) -> Dict[str, AttributionResult]:
        """Attribute the prediction for every stage (class-specific explanation)."""
        return {
            stage: self.explain(sample, stage, **kwargs)
            for stage in STAGE_ORDER
        }


__all__ = [
    "shap_available",
    "AttributionResult",
    "FeatureAttributor",
]
