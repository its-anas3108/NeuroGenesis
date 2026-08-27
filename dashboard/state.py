"""
Dashboard data access layer.
============================

Reads persisted artifacts and reports **what is actually present**. Nothing here
computes a result or invents a value: if an artifact is absent, the accessor
returns ``None`` and the page renders an explicit "not available" panel.

Two invariants the dashboard depends on:

**No fabricated status.** Module status comes from
:class:`~modules.common.run_state.RunStateTracker`, which only records what
executed. A module that never ran cannot show as completed.

**Provenance is loud.** :attr:`DashboardState.smoke_marker` is checked on every
page that displays a metric. When the outputs tree is synthetic, the page says so
before showing any number.

The loader is read-only: :func:`~modules.common.paths.create_output_dirs` is
called with ``create=False`` so that opening the dashboard never creates empty
directories that would then look like completed pipeline stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.config import NeuroGenesisConfig
from modules.common.paths import create_output_dirs
from modules.common.roi_constants import ROI_ORDER, STAGE_ORDER
from modules.common.run_state import PIPELINE, RunStateTracker, StageStatus


def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON file, returning ``None`` if absent or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file, returning ``None`` if absent or malformed."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError, ValueError):
        return None


def _read_npy(path: Path) -> Optional[np.ndarray]:
    """Read a ``.npy`` array, returning ``None`` if absent or malformed."""
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None


@dataclass
class DashboardState:
    """Read-only view of an outputs tree."""

    outputs: Path
    cfg: NeuroGenesisConfig = field(default_factory=NeuroGenesisConfig)

    def __post_init__(self) -> None:
        self.outputs = Path(self.outputs)
        self.dirs = create_output_dirs(self.outputs, create=False)
        self.tracker = RunStateTracker(self.outputs, autosave=False)

    # ── Provenance ────────────────────────────────────────────────────────

    @property
    def smoke_marker(self) -> Optional[Dict[str, Any]]:
        """Return synthetic-data provenance, or ``None`` for a real tree.

        Delegates to :func:`modules.common.provenance.detect_provenance`
        so the dashboard cannot disagree with the report generator about
        whether a tree is synthetic.
        """
        from modules.common.provenance import detect_provenance

        provenance = detect_provenance(self.outputs, self.cfg.paths.mri_dir)
        return provenance.to_dict() if provenance.is_synthetic else None

    @property
    def is_synthetic(self) -> bool:
        """True when any number in this tree derives from generated data."""
        return self.smoke_marker is not None

    def exists(self) -> bool:
        """True if the outputs root exists at all."""
        return self.outputs.exists()

    # ── Cohort and features ───────────────────────────────────────────────

    def cohort(self) -> Optional[pd.DataFrame]:
        """Return the cohort table."""
        return _read_csv(self.outputs / "patient" / "cohort.csv")

    def cohort_report(self) -> Optional[Dict[str, Any]]:
        """Return the cohort assembly report."""
        return _read_json(self.outputs / "patient" / "cohort_report.json")

    def features(self) -> Optional[pd.DataFrame]:
        """Return the unstandardised feature table."""
        return _read_csv(
            self.outputs / "features" / "morphometric_features.csv"
        )

    def feature_quality(self) -> Optional[Dict[str, Any]]:
        """Return the feature quality audit."""
        return _read_json(
            self.outputs / "features" / "morphometric_features_quality.json"
        )

    def subjects(self) -> List[str]:
        """Return every subject the tree knows about, from any source."""
        found: set = set(self.tracker.list_subjects())
        cohort = self.cohort()
        if cohort is not None and "session_id" in cohort.columns:
            found.update(cohort["session_id"].astype(str))
        patches = self.outputs / "roi" / "patches"
        if patches.exists():
            found.update(p.name for p in patches.iterdir() if p.is_dir())
        return sorted(found)

    def subjects_with_patches(self) -> List[str]:
        """Return subjects that have an ROI patch tensor on disk."""
        root = self.outputs / "roi" / "patches"
        if not root.exists():
            return []
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / f"{p.name}_roi_tensor.npy").exists()
        )

    # ── Pipeline state ────────────────────────────────────────────────────

    def pipeline_status(self, subject_id: Optional[str] = None
                        ) -> pd.DataFrame:
        """Return the ordered M1..M19 status table for display."""
        self.tracker.reset(None)
        rows = self.tracker.summary(subject_id)
        return pd.DataFrame(rows)

    def stage_record(self, code: str, subject_id: Optional[str] = None
                     ) -> Optional[Dict[str, Any]]:
        """Return one module's full record, including artifacts and metrics."""
        from modules.common.run_state import COHORT_CODES
        from dataclasses import asdict

        scope = None if code in COHORT_CODES else subject_id
        try:
            return asdict(self.tracker.get(code, scope))
        except KeyError:
            return None

    def completion(self, subject_id: Optional[str] = None) -> float:
        """Return the fraction of modules completed."""
        return self.tracker.completion_fraction(subject_id)

    # ── Per-subject imaging artifacts ─────────────────────────────────────

    def preprocessing(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Return the M1-M5 preprocessing manifest."""
        return _read_json(
            self.outputs / "preprocessing" / subject_id
            / f"{subject_id}_preprocessing.json"
        )

    def segmentation(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Return the M6-M7 segmentation manifest."""
        return _read_json(
            self.outputs / "roi" / subject_id / f"{subject_id}_segmentation.json"
        )

    def patch_tensor(self, subject_id: str) -> Optional[np.ndarray]:
        """Return the ``(5, 48, 48, 48)`` ROI patch tensor."""
        return _read_npy(
            self.outputs / "roi" / "patches" / subject_id
            / f"{subject_id}_roi_tensor.npy"
        )

    def cnn_embeddings(self, subject_id: str
                       ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Return cached CNN embeddings and their summary."""
        root = self.outputs / "cnn_embeddings" / subject_id
        return (
            _read_npy(root / f"{subject_id}_cnn_embeddings.npy"),
            _read_json(root / f"{subject_id}_cnn_embeddings.json"),
        )

    # ── NeuroProp-X artifacts ─────────────────────────────────────────────

    def neuropropx(self, subject_id: str) -> Dict[str, Any]:
        """Return every persisted NeuroProp-X artifact for one subject."""
        root = self.outputs / "neuropropx"
        return {
            "vulnerability": _read_csv(
                root / "vulnerability" / subject_id
                / f"{subject_id}_regional_vulnerability.csv"
            ),
            "A_prior": _read_npy(
                root / "adaptive_adjacency" / subject_id
                / f"{subject_id}_A_prior.npy"
            ),
            "A_att": _read_npy(
                root / "adaptive_adjacency" / subject_id
                / f"{subject_id}_A_att.npy"
            ),
            "A_star": _read_npy(
                root / "adaptive_adjacency" / subject_id
                / f"{subject_id}_A_star.npy"
            ),
            "propagation": _read_npy(
                root / "propagation" / subject_id
                / f"{subject_id}_propagation_score.npy"
            ),
            "X_star": _read_npy(
                root / "enriched_graph" / subject_id / f"{subject_id}_X_star.npy"
            ),
            "metadata": _read_json(
                root / "enriched_graph" / subject_id / f"{subject_id}_G_star.json"
            ),
        }

    # ── Model outputs ─────────────────────────────────────────────────────

    def subject_report(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Return the machine-readable subject report."""
        return _read_json(
            self.outputs / "reports" / subject_id / f"{subject_id}_report.json"
        )

    def report_markdown(self, subject_id: str) -> Optional[str]:
        """Return the Markdown subject report."""
        path = (self.outputs / "reports" / subject_id
                / f"{subject_id}_report.md")
        return path.read_text(encoding="utf-8") if path.exists() else None

    def predictions(self, variant: str = "A7") -> Optional[pd.DataFrame]:
        """Return the test-split per-subject predictions."""
        return _read_csv(
            self.outputs / "predictions" / f"test_predictions_{variant}.csv"
        )

    def test_metrics(self, variant: str = "A7") -> Optional[Dict[str, Any]]:
        """Return the test-split metrics."""
        return _read_json(
            self.outputs / "predictions" / f"test_metrics_{variant}.json"
        )

    def training_history(self, variant: str = "A7",
                         experiment: str = "default") -> Optional[Dict[str, Any]]:
        """Return the recorded training history."""
        return _read_json(
            self.outputs / "experiments" / experiment / f"training_{variant}.json"
        )

    def split_manifest(self) -> Optional[Dict[str, Any]]:
        """Return the split manifest."""
        return _read_json(self.outputs / "splits" / "split.json")

    def scaler(self) -> Optional[Dict[str, Any]]:
        """Return the fitted scaler statistics."""
        return _read_json(
            self.outputs / "scalers" / "morphometric_scaler.json"
        )

    # ── Analysis ──────────────────────────────────────────────────────────

    # ── OASIS-1 dataset integrity ─────────────────────────────────────────

    def dataset_validation(self) -> Optional[Dict[str, Any]]:
        """Return the OASIS-1 validation report."""
        return _read_json(
            self.outputs / "dataset_validation" / "oasis1_validation_report.json"
        )

    def dataset_summary(self) -> Optional[pd.DataFrame]:
        """Return the per-session OASIS-1 dataset summary."""
        return _read_csv(
            self.outputs / "dataset_validation" / "oasis1_dataset_summary.csv"
        )

    def dataset_provenance(self) -> Optional[Dict[str, Any]]:
        """Return the recorded OASIS-1 provenance block."""
        return _read_json(
            self.outputs / "dataset_validation" / "oasis1_provenance.json"
        )

    def integrity_report(self) -> Optional[Dict[str, Any]]:
        """Return the pre-training dataset integrity report."""
        return _read_json(
            self.outputs / "dataset_validation" / "dataset_integrity_report.json"
        )

    def split_csvs(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Return the three per-split CSVs required by Section 9."""
        return {
            split: _read_csv(self.outputs / "splits" / f"oasis1_{split}.csv")
            for split in ("train", "val", "test")
        }

    def statistics(self) -> Optional[Dict[str, Any]]:
        """Return the stage-wise statistics report."""
        return _read_json(
            self.outputs / "statistics" / "statistics_report.json"
        )

    def statistics_table(self, name: str) -> Optional[pd.DataFrame]:
        """Return one persisted statistics table by filename stem."""
        return _read_csv(self.outputs / "statistics" / f"{name}.csv")

    def ablation(self) -> Optional[Dict[str, Any]]:
        """Return the ablation results."""
        return _read_json(self.outputs / "ablation" / "ablation_results.json")

    def ablation_table(self, name: str) -> Optional[pd.DataFrame]:
        """Return one persisted ablation table by filename stem."""
        return _read_csv(self.outputs / "ablation" / f"{name}.csv")

    def baselines(self) -> Optional[Dict[str, Any]]:
        """Return the baseline results."""
        return _read_json(self.outputs / "baselines" / "baseline_results.json")

    def baseline_table(self) -> Optional[pd.DataFrame]:
        """Return the baseline comparison table."""
        return _read_csv(self.outputs / "baselines" / "baseline_comparison.csv")

    def roi_ranking(self) -> Optional[Dict[str, Any]]:
        """Return the ROI ranking stability report."""
        return _read_json(
            self.outputs / "roi_ranking" / "table8_ranking_stability.json"
        )

    def roi_ranking_table(self, name: str) -> Optional[pd.DataFrame]:
        """Return one persisted ROI-ranking table by filename stem."""
        return _read_csv(self.outputs / "roi_ranking" / f"{name}.csv")

    def figures(self) -> Dict[str, Path]:
        """Return every generated figure, keyed by filename stem."""
        root = self.outputs / "figures"
        if not root.exists():
            return {}
        return {p.stem: p for p in sorted(root.glob("*.png"))}

    def checkpoints(self) -> Dict[str, Path]:
        """Return every saved checkpoint, keyed by variant name."""
        root = self.outputs / "checkpoints"
        if not root.exists():
            return {}
        return {p.stem: p for p in sorted(root.glob("*.pt"))}

    # ── Environment ───────────────────────────────────────────────────────

    @staticmethod
    def environment() -> Dict[str, Any]:
        """Report which optional dependencies are available."""
        from modules.m02_preprocessing import check_imaging_dependencies
        from modules.m08_xai import shap_available
        from modules.training.baselines import available_baselines

        imaging = check_imaging_dependencies()
        return {
            "imaging": imaging["available"],
            "imaging_can_run": imaging["can_run"],
            "imaging_message": imaging["message"],
            "shap": shap_available(),
            "baselines": available_baselines(),
        }

    def readiness(self) -> List[Dict[str, Any]]:
        """Return an artifact-presence checklist for the overview page."""
        checks = [
            ("Cohort table", self.outputs / "patient" / "cohort.csv",
             "run.py --mode preprocess"),
            ("Feature table",
             self.outputs / "features" / "morphometric_features.csv",
             "run.py --mode preprocess"),
            ("Fitted scaler",
             self.outputs / "scalers" / "morphometric_scaler.json",
             "run.py --mode train_full"),
            ("Split manifest", self.outputs / "splits" / "split.json",
             "run.py --mode train_full"),
            ("Model checkpoint", self.outputs / "checkpoints" / "A7.pt",
             "run.py --mode train_full"),
            ("Test predictions",
             self.outputs / "predictions" / "test_predictions_A7.csv",
             "run.py --mode train_full"),
            ("Statistics",
             self.outputs / "statistics" / "statistics_report.json",
             "run.py --mode statistics"),
            ("Ablation",
             self.outputs / "ablation" / "ablation_results.json",
             "run.py --mode ablation"),
            ("ROI ranking",
             self.outputs / "roi_ranking" / "table8_ranking_stability.json",
             "run.py --mode xai"),
            ("Figures", self.outputs / "figures", "run.py --mode figures"),
        ]
        return [
            {
                "artifact": label,
                "present": path.exists(),
                "path": path.as_posix(),
                "produced_by": command,
            }
            for label, path, command in checks
        ]


__all__ = ["DashboardState"]
