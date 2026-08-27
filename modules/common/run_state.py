"""
Pipeline execution-state tracker (Sections 39, 40).
==================================================

The dashboard must show the real status of every pipeline module — NOT STARTED /
RUNNING / COMPLETED / FAILED — derived from actual execution, never hard-coded.
This module is the mechanism.

State is persisted as JSON so that a dashboard process can read what a separate
training process wrote:

    outputs/state/pipeline_global.json          cohort-level stages (M17, M18, ...)
    outputs/state/subjects/<subject_id>.json    per-subject stages (M1 ... M16)

Usage::

    tracker = RunStateTracker(outputs_root)

    with tracker.stage("M9", subject_id="OAS1_0001_MR1") as st:
        emb = encoder(patches)
        st.record_artifact("embeddings", path)
        st.record_metric("embed_dim", 128)

On a clean exit the stage is marked ``COMPLETED``; on an exception it is marked
``FAILED`` with the exception text preserved, and the exception is re-raised.
A stage that was never entered stays ``NOT_STARTED`` — the dashboard therefore
cannot display a green tick for work that did not happen.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modules.common.logging_utils import get_logger
from modules.common.serialization import json_safe

logger = get_logger(__name__)


class StageStatus(str, Enum):
    """Execution status of one pipeline module."""

    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline definition (Section 40)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModuleSpec:
    """Static description of one pipeline module."""

    code: str
    title: str
    #: ``"subject"`` stages run per subject; ``"cohort"`` stages run once over
    #: the whole dataset (statistics, ablation, training).
    scope: str
    description: str


#: The canonical M1..M19 pipeline. The dashboard renders this list in order.
PIPELINE: List[ModuleSpec] = [
    ModuleSpec("M1", "MRI Acquisition / Loading", "subject",
               "Load the T1 volume, header, affine and acquisition metadata."),
    ModuleSpec("M2", "Quality Control", "subject",
               "Artifact detection and QC scoring (SNR, motion, dropout, ringing)."),
    ModuleSpec("M3", "MRI Preprocessing", "subject",
               "N4 bias correction, WM-peak normalization, CLAHE, anisotropic "
               "diffusion, intensity normalization."),
    ModuleSpec("M4", "Skull Stripping", "subject",
               "Brain extraction and brain-mask generation."),
    ModuleSpec("M5", "Spatial Standardization", "subject",
               "Resample to a common 128x128x128 grid."),
    ModuleSpec("M6", "Harvard-Oxford ROI Extraction", "subject",
               "Localize the five speech-related ROIs from the atlas."),
    ModuleSpec("M7", "ROI Patch Extraction", "subject",
               "Crop context-padded 48x48x48 patches -> (5, 48, 48, 48) tensor."),
    ModuleSpec("M8", "Morphometric Feature Extraction", "subject",
               "Per-ROI volumetric, textural and intensity features."),
    ModuleSpec("M9", "3D CNN Spatial Encoding", "subject",
               "Lightweight 3D CNN -> one 128-d embedding per ROI patch."),
    ModuleSpec("M10", "Brain Graph Construction", "subject",
               "Subject-specific 5-node speech graph with the anatomical prior."),
    ModuleSpec("M11", "NeuroProp-X", "subject",
               "Disease-aware enhanced graph G* from the ordinary graph G."),
    ModuleSpec("M11.1", "SRVE", "subject",
               "Stage-Aware Regional Vulnerability Estimation."),
    ModuleSpec("M11.2", "AP-LAF", "subject",
               "Anatomical-Prior and Learned-Attention Fusion."),
    ModuleSpec("M11.3", "ANP", "subject",
               "Adaptive Neurodegeneration Propagation representation."),
    ModuleSpec("M11.4", "SAGR", "subject",
               "Stage-Aware Graph Representation; assembles G*."),
    ModuleSpec("M12", "SAEG-GATv2", "subject",
               "Stage-aware edge-gated GATv2 graph encoder."),
    ModuleSpec("M13", "Multimodal Fusion", "subject",
               "Fuse the 3D CNN and graph embeddings."),
    ModuleSpec("M14", "Stage-TGT", "subject",
               "Stage prototypes, transformer and stage-transition propensity."),
    ModuleSpec("M15", "ROI Ranking", "subject",
               "Unified ROI importance from vulnerability, attention and SHAP."),
    ModuleSpec("M16", "XAI", "subject",
               "SHAP, attention and vulnerability explanations."),
    ModuleSpec("M17", "Statistics", "cohort",
               "Stage-wise ROI/feature comparison with FDR correction."),
    ModuleSpec("M18", "Ablation", "cohort",
               "A0..A7 component ablation over repeated stratified splits."),
    ModuleSpec("M19", "Final Report", "subject",
               "Assemble the subject report."),
]

#: Fast lookup by module code.
PIPELINE_BY_CODE: Dict[str, ModuleSpec] = {m.code: m for m in PIPELINE}

#: Codes that run once over the cohort rather than per subject.
COHORT_CODES = {m.code for m in PIPELINE if m.scope == "cohort"}


# ──────────────────────────────────────────────────────────────────────────────
# State records
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    """Mutable execution record for one module, for one scope."""

    code: str
    status: str = StageStatus.NOT_STARTED.value
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    #: Logical name -> path of artifacts this stage produced. The dashboard
    #: offers these as downloads (Section 39 requirement 6).
    artifacts: Dict[str, str] = field(default_factory=dict)
    #: Numeric values worth surfacing (tensor shapes, timings, counts).
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: Free-text notes, e.g. why a stage was skipped.
    notes: List[str] = field(default_factory=list)

    def record_artifact(self, name: str, path: Path) -> None:
        """Attach a produced artifact path under a logical name."""
        self.artifacts[name] = Path(path).as_posix()

    def record_metric(self, name: str, value: Any) -> None:
        """Attach a numeric or short scalar value for display."""
        self.metrics[name] = _jsonable(value)

    def note(self, text: str) -> None:
        """Append a human-readable note."""
        self.notes.append(text)


def _jsonable(value: Any) -> Any:
    """Coerce common numeric/array types into JSON-safe values."""
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


class RunStateTracker:
    """Read/write the persisted execution state of the pipeline.

    Args:
        outputs_root: Root ``outputs/`` directory.
        autosave: Persist after every status transition. Keep this ``True`` in
            production so a dashboard in another process sees live progress; set
            ``False`` in tight unit tests to avoid filesystem churn.
    """

    def __init__(self, outputs_root: Path, autosave: bool = True) -> None:
        self.outputs_root = Path(outputs_root)
        self.state_dir = self.outputs_root / "state"
        self.subjects_dir = self.state_dir / "subjects"
        self.autosave = autosave
        self._global: Dict[str, StageRecord] = {}
        self._subjects: Dict[str, Dict[str, StageRecord]] = {}

    # ── Paths ─────────────────────────────────────────────────────────────

    @property
    def global_path(self) -> Path:
        """Path of the cohort-level state file."""
        return self.state_dir / "pipeline_global.json"

    def subject_path(self, subject_id: str) -> Path:
        """Path of one subject's state file."""
        return self.subjects_dir / f"{subject_id}.json"

    # ── Load / save ───────────────────────────────────────────────────────

    def load(self, subject_id: Optional[str] = None) -> Dict[str, StageRecord]:
        """Load state from disk for the cohort or for one subject.

        A missing or corrupt state file yields an all-``NOT_STARTED`` map rather
        than raising: an absent file legitimately means "nothing has run yet".
        """
        path = self.global_path if subject_id is None else self.subject_path(subject_id)
        records: Dict[str, StageRecord] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for code, raw in payload.get("stages", {}).items():
                    records[code] = StageRecord(**raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Corrupt state file %s (%s) — treating as empty.",
                               path, exc)
        if subject_id is None:
            self._global = records
        else:
            self._subjects[subject_id] = records
        return records

    def save(self, subject_id: Optional[str] = None) -> Path:
        """Persist state for the cohort or for one subject."""
        if subject_id is None:
            path, records = self.global_path, self._global
            scope: Dict[str, Any] = {"scope": "cohort"}
        else:
            path, records = self.subject_path(subject_id), self._subjects.get(subject_id, {})
            scope = {"scope": "subject", "subject_id": subject_id}
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **scope,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {code: asdict(rec) for code, rec in records.items()},
        }
        path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return path

    # ── Access ────────────────────────────────────────────────────────────

    def _bucket(self, subject_id: Optional[str]) -> Dict[str, StageRecord]:
        """Return the in-memory record map, loading from disk on first use."""
        if subject_id is None:
            if not self._global:
                self.load(None)
            return self._global
        if subject_id not in self._subjects:
            self.load(subject_id)
        return self._subjects.setdefault(subject_id, {})

    def get(self, code: str, subject_id: Optional[str] = None) -> StageRecord:
        """Return the record for ``code``, creating a NOT_STARTED one if absent.

        Raises:
            KeyError: If ``code`` is not a declared pipeline module, so that a
                typo cannot create a phantom stage the dashboard then renders.
        """
        if code not in PIPELINE_BY_CODE:
            raise KeyError(
                f"Unknown pipeline module {code!r}. "
                f"Declared modules: {list(PIPELINE_BY_CODE)}"
            )
        bucket = self._bucket(subject_id)
        return bucket.setdefault(code, StageRecord(code=code))

    def status(self, code: str, subject_id: Optional[str] = None) -> StageStatus:
        """Return the current :class:`StageStatus` of one module."""
        return StageStatus(self.get(code, subject_id).status)

    def mark_skipped(self, code: str, reason: str,
                     subject_id: Optional[str] = None) -> StageRecord:
        """Mark a module as deliberately skipped, with a stated reason."""
        rec = self.get(code, subject_id)
        rec.status = StageStatus.SKIPPED.value
        rec.note(reason)
        if self.autosave:
            self.save(subject_id)
        return rec

    def reset(self, subject_id: Optional[str] = None) -> None:
        """Clear all recorded state for the cohort or for one subject."""
        if subject_id is None:
            self._global = {}
        else:
            self._subjects[subject_id] = {}
        if self.autosave:
            self.save(subject_id)

    # ── Stage context manager ─────────────────────────────────────────────

    @contextmanager
    def stage(self, code: str, subject_id: Optional[str] = None
              ) -> Iterator[StageRecord]:
        """Run a pipeline module, recording RUNNING -> COMPLETED / FAILED.

        Args:
            code: Module code, e.g. ``"M11.2"``.
            subject_id: ``None`` for cohort-scoped modules.

        Yields:
            The mutable :class:`StageRecord`, so the body can attach artifacts
            and metrics that the dashboard will display.

        Raises:
            Whatever the body raises — the exception is recorded and re-raised,
            never swallowed. A silently swallowed failure would leave a stage
            looking complete with no output.
        """
        rec = self.get(code, subject_id)
        rec.status = StageStatus.RUNNING.value
        rec.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        rec.finished_at = None
        rec.duration_s = None
        rec.error = None
        rec.traceback = None
        if self.autosave:
            self.save(subject_id)

        t0 = time.perf_counter()
        try:
            yield rec
        except BaseException as exc:
            rec.status = StageStatus.FAILED.value
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.traceback = traceback.format_exc()
            rec.duration_s = round(time.perf_counter() - t0, 4)
            rec.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if self.autosave:
                self.save(subject_id)
            logger.error("[%s] FAILED%s: %s", code,
                         f" ({subject_id})" if subject_id else "", rec.error)
            raise
        else:
            rec.status = StageStatus.COMPLETED.value
            rec.duration_s = round(time.perf_counter() - t0, 4)
            rec.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if self.autosave:
                self.save(subject_id)
            logger.info("[%s] COMPLETED%s in %.3fs", code,
                        f" ({subject_id})" if subject_id else "", rec.duration_s)

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self, subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return an ordered M1..M19 status table for display.

        Cohort-scoped modules are always read from the global state file even
        when a ``subject_id`` is given, since statistics and ablation are
        properties of the dataset, not of one subject.
        """
        rows: List[Dict[str, Any]] = []
        for spec in PIPELINE:
            scope_id = None if spec.code in COHORT_CODES else subject_id
            rec = self.get(spec.code, scope_id)
            rows.append({
                "code": spec.code,
                "title": spec.title,
                "scope": spec.scope,
                "status": rec.status,
                "duration_s": rec.duration_s,
                "started_at": rec.started_at,
                "finished_at": rec.finished_at,
                "error": rec.error,
                "n_artifacts": len(rec.artifacts),
                "description": spec.description,
            })
        return rows

    def completion_fraction(self, subject_id: Optional[str] = None) -> float:
        """Fraction of pipeline modules in the COMPLETED state."""
        rows = self.summary(subject_id)
        if not rows:
            return 0.0
        done = sum(1 for r in rows if r["status"] == StageStatus.COMPLETED.value)
        return done / len(rows)

    def list_subjects(self) -> List[str]:
        """Return subject IDs that have a persisted state file."""
        if not self.subjects_dir.exists():
            return []
        return sorted(p.stem for p in self.subjects_dir.glob("*.json"))


__all__ = [
    "StageStatus",
    "ModuleSpec",
    "PIPELINE",
    "PIPELINE_BY_CODE",
    "COHORT_CODES",
    "StageRecord",
    "RunStateTracker",
]
