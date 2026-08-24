"""
Leakage-safe morphometric feature scaler (Sections 6, 23, 24).
=============================================================

Fits on the **training split only**, then transforms every split with the frozen
statistics. Two things are fitted:

1. **Per-ROI, per-feature standardisation.** Statistics are computed separately
   for each ROI rather than pooled. Broca's raw volume and STG's raw volume have
   genuinely different scales, and pooling them would leave the standardised
   values encoding "which ROI is this" instead of "how unusual is this region for
   its own kind".

2. **The atrophy-index reference median.** ``atrophy_index`` is defined relative
   to a cohort median, which makes it a *fitted* quantity. Computing it over the
   whole dataset would leak the test set's volume distribution into the training
   features, so the median comes from the training split alone.

Robust statistics
-----------------

Median and IQR are used instead of mean and standard deviation. With 154 training
subjects, a single failed skull-strip or an empty ROI mask produces an extreme
outlier, and a mean/SD scaler would let that one subject compress everyone else
into a narrow band. IQR is divided by 1.349 so that the resulting scale matches
a standard deviation for normally distributed data, keeping the values on a
familiar range.

Persistence
-----------

:meth:`MorphometricScaler.save` writes the fitted statistics to JSON so that
inference on a new subject months later uses exactly the values the model was
trained with. A scaler is never re-fitted at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER
from modules.m04_feature_extraction.feature_spec import (
    FEATURE_ORDER,
    N_FEATURES,
)

logger = get_logger(__name__)

#: IQR-to-SD conversion factor for a normal distribution.
_IQR_TO_SD = 1.349


@dataclass
class ScalerStats:
    """Fitted statistics for one ROI x feature cell."""

    center: float
    scale: float
    n: int
    #: True when the fitted scale was degenerate and had to be replaced by 1.0.
    degenerate: bool = False


class MorphometricScaler:
    """Per-ROI robust scaler with a fitted atrophy-index reference.

    Args:
        feature_order: Feature column order. Defaults to
            :data:`~modules.m04_feature_extraction.feature_spec.FEATURE_ORDER`.
        roi_order: ROI order. Defaults to :data:`ROI_ORDER`.
        clip: Clip standardised values to ``[-clip, clip]``. Bounds the influence
            of an outlier subject at inference time, when there is no
            opportunity to inspect it. ``None`` disables clipping.
    """

    def __init__(
        self,
        feature_order: Optional[List[str]] = None,
        roi_order: Optional[List[str]] = None,
        clip: Optional[float] = 5.0,
    ) -> None:
        self.feature_order = list(feature_order or FEATURE_ORDER)
        self.roi_order = list(roi_order or ROI_ORDER)
        self.clip = clip
        self.fitted = False
        #: ``(roi, feature) -> ScalerStats``
        self.stats: Dict[Tuple[str, str], ScalerStats] = {}
        #: ``roi -> median normalized_volume`` on the training split.
        self.atrophy_reference: Dict[str, float] = {}
        self.fit_report: Dict[str, Any] = {}

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame,
            train_session_ids: Optional[List[str]] = None) -> "MorphometricScaler":
        """Fit statistics on the training split.

        Args:
            features: Long-format table with columns ``session_id``,
                ``roi_name`` and one column per feature.
            train_session_ids: Sessions belonging to the training split. When
                ``None``, *every* row is used — acceptable only when the caller
                has already filtered to the training split, and logged loudly
                because it is the shape a leak takes.

        Returns:
            ``self``, fitted.

        Raises:
            KeyError: If required columns are absent.
            ValueError: If the training subset is empty.
        """
        for col in ("session_id", "roi_name"):
            if col not in features.columns:
                raise KeyError(
                    f"Feature table is missing required column {col!r}. "
                    f"Present: {list(features.columns)}"
                )

        if train_session_ids is None:
            logger.warning(
                "MorphometricScaler.fit was called without train_session_ids; "
                "fitting on ALL supplied rows. This is only safe if the caller "
                "has already restricted the table to the training split."
            )
            train = features
        else:
            train = features[features["session_id"].isin(set(train_session_ids))]

        if train.empty:
            raise ValueError(
                "The training subset is empty — no rows matched "
                "train_session_ids. Nothing can be fitted."
            )

        available = [f for f in self.feature_order if f in train.columns]
        missing = [f for f in self.feature_order if f not in train.columns]

        self.stats = {}
        degenerate: List[str] = []
        for roi in self.roi_order:
            sub = train[train["roi_name"] == roi]
            for feat in available:
                values = pd.to_numeric(sub[feat], errors="coerce").dropna().values
                if values.size == 0:
                    self.stats[(roi, feat)] = ScalerStats(0.0, 1.0, 0, True)
                    degenerate.append(f"{roi}/{feat} (no data)")
                    continue
                center = float(np.median(values))
                q75, q25 = np.percentile(values, [75, 25])
                scale = float((q75 - q25) / _IQR_TO_SD)
                is_degenerate = not np.isfinite(scale) or scale < 1e-8
                if is_degenerate:
                    # A zero IQR means the feature is constant across the
                    # training split for this ROI. Standardising by ~0 would
                    # produce infinities; a unit scale makes the column a
                    # constant zero, which the model can simply ignore.
                    scale = 1.0
                    degenerate.append(f"{roi}/{feat} (zero IQR)")
                self.stats[(roi, feat)] = ScalerStats(
                    center=center, scale=scale, n=int(values.size),
                    degenerate=is_degenerate,
                )

        # Atrophy-index reference: per-ROI median eTIV-normalized volume.
        self.atrophy_reference = {}
        if "normalized_volume" in train.columns:
            for roi in self.roi_order:
                sub = train[train["roi_name"] == roi]
                vals = pd.to_numeric(
                    sub["normalized_volume"], errors="coerce"
                ).dropna().values
                if vals.size:
                    self.atrophy_reference[roi] = float(np.median(vals))

        self.fitted = True
        self.fit_report = {
            "n_train_sessions": int(train["session_id"].nunique()),
            "n_train_rows": int(len(train)),
            "features_fitted": available,
            "features_missing": missing,
            "degenerate_cells": degenerate,
            "atrophy_reference": dict(self.atrophy_reference),
            "clip": self.clip,
            "center_statistic": "median",
            "scale_statistic": f"IQR / {_IQR_TO_SD}",
        }

        logger.info(
            "Scaler fitted on %d training session(s), %d row(s); %d degenerate "
            "cell(s)%s",
            self.fit_report["n_train_sessions"],
            self.fit_report["n_train_rows"],
            len(degenerate),
            f"; missing features {missing}" if missing else "",
        )
        if degenerate:
            logger.warning(
                "Degenerate scaler cells (constant or absent in training data): "
                "%s", degenerate[:10]
            )
        return self

    # ── Derived features ──────────────────────────────────────────────────

    def add_atrophy_index(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add the ``atrophy_index`` column using the fitted reference.

        Args:
            features: Long-format feature table containing
                ``normalized_volume``.

        Returns:
            A copy with ``atrophy_index`` added. Rows whose ROI has no fitted
            reference, or whose ``normalized_volume`` is missing, receive
            ``NaN`` — never a fabricated 0.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError(
                "add_atrophy_index requires a fitted scaler: the index is "
                "defined relative to a training-split reference median."
            )
        out = features.copy()
        if "normalized_volume" not in out.columns or not self.atrophy_reference:
            out["atrophy_index"] = np.nan
            return out

        def compute(row: pd.Series) -> float:
            ref = self.atrophy_reference.get(row["roi_name"])
            val = row.get("normalized_volume")
            if ref is None or ref <= 0 or val is None or not np.isfinite(val):
                return np.nan
            return float(np.clip(1.0 - (val / ref), -1.0, 1.0))

        out["atrophy_index"] = out.apply(compute, axis=1)
        return out

    # ── Transform ─────────────────────────────────────────────────────────

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardise features in place-style, returning a new table.

        Args:
            features: Long-format feature table.

        Returns:
            A copy in which every fitted feature column is standardised.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("transform called before fit.")

        out = features.copy()
        for feat in self.feature_order:
            if feat not in out.columns:
                continue
            values = pd.to_numeric(out[feat], errors="coerce").astype(float)
            centers = out["roi_name"].map(
                lambda r: self.stats.get((r, feat), ScalerStats(0.0, 1.0, 0)).center
            )
            scales = out["roi_name"].map(
                lambda r: self.stats.get((r, feat), ScalerStats(0.0, 1.0, 0)).scale
            )
            scaled = (values - centers) / scales
            if self.clip is not None:
                scaled = scaled.clip(-self.clip, self.clip)
            out[feat] = scaled
        return out

    def to_tensor_array(
        self, features: pd.DataFrame, session_ids: List[str],
        fill_missing: float = 0.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """Assemble a ``(n_sessions, N_ROI, N_FEATURES)`` array.

        Args:
            features: Standardised long-format feature table.
            session_ids: Sessions to include, in the desired output order.
            fill_missing: Value for absent or non-finite cells. After
                standardisation, 0.0 is the fitted per-ROI median, so a missing
                cell becomes "typical for this region" rather than an extreme
                value. This is imputation and is reported, not hidden: the
                returned warning list names every affected session.

        Returns:
            ``(array, warnings)``.

        Raises:
            ValueError: If no requested session is present in the table.
        """
        index = {
            (str(r["session_id"]), str(r["roi_name"])): r
            for _, r in features.iterrows()
        }
        n = len(session_ids)
        arr = np.full((n, len(self.roi_order), len(self.feature_order)),
                      fill_missing, dtype=np.float32)
        warnings: List[str] = []
        found_any = False

        for si, sid in enumerate(session_ids):
            for ri, roi in enumerate(self.roi_order):
                row = index.get((str(sid), roi))
                if row is None:
                    warnings.append(f"{sid}/{roi}: no feature row")
                    continue
                found_any = True
                for fi, feat in enumerate(self.feature_order):
                    val = row.get(feat, np.nan)
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        fval = np.nan
                    if not np.isfinite(fval):
                        warnings.append(f"{sid}/{roi}/{feat}: non-finite")
                        continue
                    arr[si, ri, fi] = fval

        if not found_any:
            raise ValueError(
                f"None of the {n} requested session(s) were found in the "
                "feature table."
            )
        if warnings:
            logger.warning(
                "Feature assembly imputed %d cell(s) with %.1f; first few: %s",
                len(warnings), fill_missing, warnings[:5],
            )
        return arr, warnings

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        """Write fitted statistics to JSON."""
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_order": self.feature_order,
            "roi_order": self.roi_order,
            "clip": self.clip,
            "atrophy_reference": self.atrophy_reference,
            "fit_report": self.fit_report,
            "stats": [
                {
                    "roi": roi, "feature": feat,
                    "center": s.center, "scale": s.scale, "n": s.n,
                    "degenerate": s.degenerate,
                }
                for (roi, feat), s in self.stats.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Scaler saved: %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MorphometricScaler":
        """Reload a scaler written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(
            feature_order=payload["feature_order"],
            roi_order=payload["roi_order"],
            clip=payload.get("clip"),
        )
        scaler.stats = {
            (row["roi"], row["feature"]): ScalerStats(
                center=row["center"], scale=row["scale"], n=row["n"],
                degenerate=row.get("degenerate", False),
            )
            for row in payload["stats"]
        }
        scaler.atrophy_reference = payload.get("atrophy_reference", {})
        scaler.fit_report = payload.get("fit_report", {})
        scaler.fitted = True
        logger.info("Scaler loaded: %s", path)
        return scaler

    def summary(self) -> Dict[str, Any]:
        """Return a description of the fitted scaler for the dashboard."""
        return {
            "fitted": self.fitted,
            "n_features": len(self.feature_order),
            "n_roi": len(self.roi_order),
            "center_statistic": "median (per ROI, per feature)",
            "scale_statistic": f"IQR / {_IQR_TO_SD} (per ROI, per feature)",
            "robustness_rationale": (
                "Median and IQR are used instead of mean and SD so that a "
                "single failed skull-strip or empty ROI mask cannot compress "
                "the rest of the cohort."
            ),
            "clip": self.clip,
            "leakage_control": "fitted on the training split only",
            "atrophy_reference": dict(self.atrophy_reference),
            "fit_report": dict(self.fit_report),
        }


__all__ = ["ScalerStats", "MorphometricScaler"]
