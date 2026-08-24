"""
M8 — Morphometric feature extraction (Section 6).
=================================================

Thin, well-typed wrapper over the **preserved**
``features.feature_extractor.FeatureExtractor``. That class is kept unchanged —
it computes the 13 real per-ROI volumetric, intensity and textural features that
the pre-refactor pipeline produced, and there was no reason to rewrite working
code.

What this wrapper adds:

* a stable long-format table keyed by ``session_id`` and ``roi_name``, in
  canonical :data:`ROI_ORDER`;
* the four derived features of
  :mod:`modules.m04_feature_extraction.feature_spec` (``gm_fraction``,
  ``normalized_volume``, ``surface_to_volume_ratio``, and the placeholder for
  the cohort-referenced ``atrophy_index``);
* a data-quality audit that reports degraded features instead of letting them
  pass as real numbers.

The data-quality audit matters
------------------------------

The preserved extractor computes surface area with ``skimage.marching_cubes``
inside a ``try/except`` that returns ``0.0`` on any failure, and
``cortical_thickness_mm`` then falls back to a hard-coded ``2.5``. If
scikit-image is not installed, both columns silently become constants — a
constant feature is not an error the model can detect, it simply contributes
nothing while appearing in every table and figure as though it were measured.
:class:`FeatureQualityReport` therefore checks for exactly that and says so.
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
    DERIVED_FEATURES,
    FEATURE_ORDER,
    RAW_FEATURES,
)

logger = get_logger(__name__)


@dataclass
class FeatureQualityReport:
    """Audit of which features are genuinely measured versus degraded."""

    n_sessions: int = 0
    n_rows: int = 0
    #: Feature -> reason it is unusable or degraded.
    degraded: Dict[str, str] = field(default_factory=dict)
    #: Features that are constant across the whole table.
    constant: List[str] = field(default_factory=list)
    #: Features with any non-finite value, and the count.
    non_finite: Dict[str, int] = field(default_factory=dict)
    #: Sessions for which fewer than N_ROI rows were produced.
    incomplete_sessions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def usable_features(self) -> List[str]:
        """Features that are neither degraded nor constant."""
        bad = set(self.degraded) | set(self.constant)
        return [f for f in FEATURE_ORDER if f not in bad]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot."""
        return {
            "n_sessions": self.n_sessions,
            "n_rows": self.n_rows,
            "degraded": dict(self.degraded),
            "constant": list(self.constant),
            "non_finite": dict(self.non_finite),
            "incomplete_sessions": list(self.incomplete_sessions),
            "usable_features": self.usable_features,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Sessions: {self.n_sessions}  Rows: {self.n_rows} "
            f"(expected {self.n_sessions * N_ROI})",
            f"Usable features: {len(self.usable_features)}/{len(FEATURE_ORDER)}",
        ]
        for feat, reason in self.degraded.items():
            lines.append(f"  DEGRADED {feat}: {reason}")
        if self.constant:
            lines.append(f"  CONSTANT: {self.constant}")
        if self.non_finite:
            lines.append(f"  NON-FINITE: {self.non_finite}")
        if self.incomplete_sessions:
            lines.append(
                f"  INCOMPLETE ({len(self.incomplete_sessions)}): "
                f"{self.incomplete_sessions[:5]}"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class MorphometricFeatureExtractor:
    """Extract per-ROI morphometric features for one or many subjects.

    Args:
        output_dir: Directory for the underlying extractor's own artifacts.
        voxel_volume_mm3: Volume of one voxel in mm^3.
        gm_threshold: Grey-matter intensity threshold as a fraction of the patch
            maximum.
    """

    def __init__(
        self,
        output_dir: Path,
        voxel_volume_mm3: float = 1.0,
        gm_threshold: float = 0.30,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.voxel_volume_mm3 = voxel_volume_mm3
        self.gm_threshold = gm_threshold
        self._extractor = None  # created lazily

    def _backend(self):
        """Instantiate the preserved extractor on first use."""
        if self._extractor is None:
            from features.feature_extractor import FeatureExtractor

            self._extractor = FeatureExtractor(
                output_dir=self.output_dir,
                voxel_volume_mm3=self.voxel_volume_mm3,
                gm_threshold=self.gm_threshold,
            )
        return self._extractor

    # ── Extraction ────────────────────────────────────────────────────────

    def extract_subject(
        self,
        patches: Dict[str, np.ndarray],
        session_id: str,
        etiv: Optional[float] = None,
    ) -> pd.DataFrame:
        """Extract features for one subject's five ROI patches.

        Args:
            patches: ``{roi_name: (D, H, W) array}``. Missing ROIs are reported
                as rows of ``NaN`` rather than being dropped, so the table
                always has ``N_ROI`` rows per session and a downstream shape
                error cannot hide a missing region.
            session_id: Session identifier.
            etiv: Estimated total intracranial volume from the OASIS-1 metadata
                table, used for ``normalized_volume``. ``None`` leaves the
                eTIV-dependent features as ``NaN``.

        Returns:
            Long-format table with ``N_ROI`` rows.
        """
        backend = self._backend()
        records: List[Dict[str, Any]] = []

        for roi in ROI_ORDER:
            patch = patches.get(roi)
            if patch is None:
                logger.warning("Session %s is missing ROI patch %s", session_id, roi)
                record: Dict[str, Any] = {f: np.nan for f in RAW_FEATURES}
            else:
                raw = backend.extract_single(
                    patch=np.asarray(patch, dtype=np.float32),
                    patient_id=session_id,
                    roi_name=roi,
                )
                record = {f: raw.get(f, np.nan) for f in RAW_FEATURES}
            record["session_id"] = session_id
            record["roi_name"] = roi
            record["etiv"] = etiv if etiv is not None else np.nan
            records.append(record)

        df = pd.DataFrame(records)
        return self.add_derived(df)

    @staticmethod
    def add_derived(features: pd.DataFrame) -> pd.DataFrame:
        """Add the derived features that need no cohort reference.

        ``atrophy_index`` is *not* computed here — it depends on a training-split
        median and is added by
        :meth:`~modules.m04_feature_extraction.scaler.MorphometricScaler.add_atrophy_index`.
        It is created as ``NaN`` so the column order stays stable.

        Args:
            features: Long-format table containing the raw features.

        Returns:
            A copy with the derived columns added.
        """
        out = features.copy()

        def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
            n = pd.to_numeric(num, errors="coerce").astype(float)
            d = pd.to_numeric(den, errors="coerce").astype(float)
            # A zero or non-finite denominator yields NaN rather than inf: an
            # empty ROI has no defined fraction, and inf would propagate through
            # the scaler as a clipped extreme.
            return pd.Series(
                np.where((d > 0) & np.isfinite(d) & np.isfinite(n), n / d, np.nan),
                index=out.index,
            )

        out["gm_fraction"] = safe_ratio(
            out.get("gm_volume_mm3", pd.Series(dtype=float)),
            out.get("brain_volume_mm3", pd.Series(dtype=float)),
        )
        out["normalized_volume"] = safe_ratio(
            out.get("brain_volume_mm3", pd.Series(dtype=float)),
            out.get("etiv", pd.Series(dtype=float)),
        )
        out["surface_to_volume_ratio"] = safe_ratio(
            out.get("surface_area_vox", pd.Series(dtype=float)),
            out.get("brain_volume_mm3", pd.Series(dtype=float)),
        )
        if "atrophy_index" not in out.columns:
            out["atrophy_index"] = np.nan

        ordered = ["session_id", "roi_name", "etiv"] + FEATURE_ORDER
        present = [c for c in ordered if c in out.columns]
        rest = [c for c in out.columns if c not in present]
        return out[present + rest]

    # ── Quality audit ─────────────────────────────────────────────────────

    @staticmethod
    def audit(features: pd.DataFrame) -> FeatureQualityReport:
        """Audit a feature table for degraded, constant or missing values.

        Args:
            features: Long-format feature table.

        Returns:
            A :class:`FeatureQualityReport`.
        """
        report = FeatureQualityReport(
            n_sessions=int(features["session_id"].nunique())
            if "session_id" in features.columns else 0,
            n_rows=len(features),
        )

        # Surface area collapsing to zero everywhere means marching cubes never
        # ran — almost always a missing scikit-image install.
        if "surface_area_vox" in features.columns:
            sa = pd.to_numeric(features["surface_area_vox"], errors="coerce")
            if sa.notna().any() and float(np.nanmax(sa.values)) <= 0.0:
                report.degraded["surface_area_vox"] = (
                    "All values are zero. The preserved extractor returns 0.0 "
                    "when skimage.marching_cubes is unavailable or fails; "
                    "install scikit-image to obtain real surface areas."
                )
                report.degraded["cortical_thickness_mm"] = (
                    "Derived from surface area, which is unavailable, so the "
                    "extractor's hard-coded 2.5 mm fallback was used. Not a "
                    "measurement."
                )
                report.degraded["surface_to_volume_ratio"] = (
                    "Derived from surface area, which is unavailable."
                )

        for feat in FEATURE_ORDER:
            if feat not in features.columns:
                report.degraded.setdefault(feat, "column absent from the table")
                continue
            values = pd.to_numeric(features[feat], errors="coerce")
            n_bad = int(values.isna().sum())
            if n_bad:
                report.non_finite[feat] = n_bad
            finite = values.dropna()
            if len(finite) > 1 and float(finite.std()) < 1e-12:
                report.constant.append(feat)

        if "session_id" in features.columns:
            counts = features.groupby("session_id").size()
            report.incomplete_sessions = sorted(
                counts[counts < N_ROI].index.astype(str).tolist()
            )

        all_nan = [f for f, n in report.non_finite.items()
                   if n == report.n_rows and report.n_rows > 0]
        if all_nan:
            report.warnings.append(
                f"Features that are entirely missing: {all_nan}. These carry no "
                "information and will be standardised to a constant zero."
            )
        if not report.usable_features:
            report.warnings.append(
                "No usable features remain. Training cannot proceed."
            )

        logger.info("Feature quality audit:\n%s", report.summary())
        return report


def save_features(
    features: pd.DataFrame,
    report: FeatureQualityReport,
    out_dir: Path,
    name: str = "morphometric_features",
) -> Dict[str, Path]:
    """Persist a feature table and its quality report.

    Args:
        features: Long-format feature table.
        report: The accompanying quality report.
        out_dir: Destination directory.
        name: Filename stem.

    Returns:
        Mapping of logical name -> written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}_quality.json"
    features.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Features saved: %s", csv_path)
    return {"features_csv": csv_path, "quality_json": json_path}


__all__ = [
    "FeatureQualityReport",
    "MorphometricFeatureExtractor",
    "save_features",
]
