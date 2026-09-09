#!/usr/bin/env python
"""
NeuroGenesis pipeline orchestrator (Sections 29, 30).
=====================================================

NeuroGenesis is a multimodal NeuroAI framework for stage-wise analysis of
Alzheimer's disease using structural MRI of speech-related brain networks. It
localizes five speech-related regions using the Harvard-Oxford atlas, learns
local three-dimensional representations with a lightweight 3D CNN, constructs a
subject-specific brain graph, and applies the proposed NeuroProp-X framework to
estimate stage-relevant regional vulnerability, fuse anatomical priors with
learned graph attention, and derive adaptive propagation representations. The
resulting enriched disease graph is processed by an edge-gated GATv2 model for
CN/MCI/AD classification. A Stage-Temporal Graph Transformer learns the ordered
CN->MCI->AD representation to estimate model-derived stage-transition propensity.

Modes
-----

::

    python run.py --mode validate_dataset  # OASIS-1 integrity + validation report
    python run.py --mode preprocess     # M1-M8: imaging -> patches -> features
    python run.py --mode train_cnn      # M9: cache 3D CNN embeddings
    python run.py --mode train_graph    # graph stage on cached embeddings
    python run.py --mode train_full     # end-to-end training of the full model
    python run.py --mode ablation       # A0-A7 over repeated splits + baselines
    python run.py --mode evaluate       # test-split evaluation of a checkpoint
    python run.py --mode xai            # explanations for one or all subjects
    python run.py --mode report         # subject reports
    python run.py --mode statistics     # stage-wise statistical analysis
    python run.py --mode figures        # research figures
    python run.py --mode dashboard      # launch the Streamlit inspection app
    python run.py --mode status         # print pipeline state and readiness

Single-subject inference::

    python run.py --mode report --patient_id OAS1_0028_MR1

Every mode writes a config snapshot and a seed report into the experiment
directory, so a run can be reproduced from its own outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.logging_utils import get_logger, log_banner, setup_logging  # noqa: E402
from modules.common.paths import create_output_dirs  # noqa: E402
from modules.common.roi_constants import STAGE_ORDER  # noqa: E402
from modules.common.run_state import RunStateTracker, StageStatus  # noqa: E402
from modules.common.seeds import set_all_seeds  # noqa: E402

logger = get_logger(__name__)

MODES = (
    "validate_dataset",
    "preprocess", "train_cnn", "train_graph", "train_full", "ablation",
    "evaluate", "xai", "report", "statistics", "figures", "dashboard", "status",
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared context
# ──────────────────────────────────────────────────────────────────────────────

class Context:
    """Everything a mode needs, assembled once.

    Loading is lazy and each accessor raises a message naming the mode that
    produces the missing artifact, so a wrong-order invocation gives an
    actionable error instead of a stack trace.
    """

    def __init__(self, cfg: NeuroGenesisConfig) -> None:
        self.cfg = cfg
        self.outputs = Path(cfg.paths.outputs_dir)
        self.dirs = create_output_dirs(self.outputs)
        self.tracker = RunStateTracker(self.outputs)
        self._cohort: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._smoke: Optional[Dict[str, Any]] = None
        self._manager: Optional[Any] = None
        self._validated_index: Optional[pd.DataFrame] = None

    @property
    def smoke_marker(self) -> Optional[Dict[str, Any]]:
        """Return synthetic-data provenance, or ``None`` for a real tree.

        Checks both generators: fabricated ROI patches stamped into the
        outputs tree, and phantom MRI stamped into the dataset directory.
        The second is the one that would otherwise slip through, because
        its artifacts come out of the real imaging pipeline.
        """
        if self._smoke is None:
            from modules.common.provenance import detect_provenance

            provenance = detect_provenance(
                self.outputs, self.cfg.paths.mri_dir
            )
            self._smoke = provenance.to_dict() if provenance.is_synthetic \
                else {}
        return self._smoke or None

    @property
    def cohort_path(self) -> Path:
        """Path of the persisted cohort table."""
        return self.outputs / "patient" / "cohort.csv"

    @property
    def features_path(self) -> Path:
        """Path of the persisted feature table."""
        return self.outputs / "features" / "morphometric_features.csv"

    @property
    def scaler_path(self) -> Path:
        """Path of the fitted scaler."""
        return self.outputs / "scalers" / "morphometric_scaler.json"

    @property
    def split_path(self) -> Path:
        """Path of the split manifest."""
        return self.outputs / "splits" / "split.json"

    def checkpoint_path(self, variant: str = "A7") -> Path:
        """Path of a variant's checkpoint."""
        return self.outputs / "checkpoints" / f"{variant}.pt"

    def cohort(self, build_if_missing: bool = True) -> pd.DataFrame:
        """Return the cohort table, building it from metadata if absent."""
        if self._cohort is not None:
            return self._cohort
        if self.cohort_path.exists():
            self._cohort = pd.read_csv(self.cohort_path)
            return self._cohort
        if not build_if_missing:
            raise FileNotFoundError(
                f"No cohort table at {self.cohort_path}. Run "
                "`python run.py --mode preprocess` first."
            )
        from modules.m01_dataset import build_cohort, save_cohort

        cohort, report = build_cohort(self.cfg.paths, self.cfg.data)
        save_cohort(cohort, report, self.outputs / "patient")
        self._cohort = cohort
        return cohort

    def features(self) -> pd.DataFrame:
        """Return the unstandardised feature table."""
        if self._features is not None:
            return self._features
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"No feature table at {self.features_path}. Run "
                "`python run.py --mode preprocess` first. If no MRI data is "
                "available, `python tools/make_smoke_artifacts.py` generates "
                "clearly-labelled synthetic artifacts for validating the code "
                "path."
            )
        self._features = pd.read_csv(self.features_path)
        return self._features

    def oasis1_manager(self):
        """Return the OASIS-1 data manager for the configured root.

        Raises:
            FileNotFoundError: If no OASIS-1 root is configured. The
                message is the Section 17 wording, because a research
                run without real data must stop rather than substitute.
        """
        from modules.m01_dataset import OASIS1DataManager

        if self._manager is not None:
            return self._manager
        root = self.cfg.paths.oasis1_root
        if not root:
            raise FileNotFoundError(
                "REAL OASIS-1 DATA REQUIRED\n\n"
                "No OASIS-1 root is configured. Pass --oasis1-root "
                "<path> or set paths.oasis1_root in the config.\n\n"
                "Obtain OASIS-1 from "\
                "https://sites.wustl.edu/oasisbrains/home/oasis-1/ , "
                "extract the discs with tools/extract_oasis1.py, and "
                "point the run at the extraction directory.\n\n"
                "Synthetic and substitute datasets are disabled for "
                "research execution."
            )
        self._manager = OASIS1DataManager(
            oasis1_root=Path(root),
            volume_kind=self.cfg.data.oasis1_volume_kind,
            metadata_csv=Path(self.cfg.paths.metadata_csv),
        )
        return self._manager

    def validated_index(self, refresh: bool = False) -> pd.DataFrame:
        """Return the validated OASIS-1 index, from cache if available."""
        if self._validated_index is not None and not refresh:
            return self._validated_index
        cached = self.outputs / "dataset_validation" \
            / "oasis1_dataset_summary.csv"
        if cached.exists() and not refresh:
            self._validated_index = pd.read_csv(cached)
            return self._validated_index
        self._validated_index = self.oasis1_manager().index(
            validated=True, deep=self.cfg.data.deep_validation
        )
        return self._validated_index

    def enforce_integrity(self, session_ids, split=None) -> None:
        """Run the Section 8 gate; raises before any model sees data."""
        from modules.m01_dataset import check_dataset_integrity

        cohort = self.cohort()
        subject_of = dict(zip(
            cohort["session_id"].astype(str),
            cohort["subject_id"].astype(str),
        ))
        check_dataset_integrity(
            training_session_ids=list(session_ids),
            validated_index=self.validated_index(),
            dataset_source=self.cfg.data.dataset_source,
            synthetic_data_enabled=self.cfg.data.allow_synthetic_data,
            split_assignment=split.assignment() if split else None,
            subject_of=subject_of,
            provenance=self.smoke_marker,
            out_dir=self.outputs / "dataset_validation",
        )

    def trainable_cohort(self, require_patches: bool = False) -> pd.DataFrame:
        """Return the cohort restricted to sessions that can actually be used.

        A session is trainable only if the feature table has rows for it. The
        full labelled cohort is larger than the processed subset whenever
        preprocessing has been run on a subset, or whenever some subjects failed
        preprocessing. Splitting the full cohort in that situation produces
        splits whose sessions have no features, and the failure surfaces much
        later as an opaque lookup error.

        Args:
            require_patches: Also require a cached ROI patch tensor, which the
                variants with a 3D CNN branch need.

        Returns:
            The filtered cohort, in the original order.

        Raises:
            ValueError: If fewer than two sessions per stage remain, since a
                stratified train/val/test split is then impossible.
        """
        from modules.m06_spatial_encoder.patch_dataset import patch_tensor_path

        cohort = self.cohort()
        features = self.features()
        available = set(features["session_id"].astype(str))

        usable = cohort[cohort["session_id"].astype(str).isin(available)]
        if require_patches:
            usable = usable[usable["session_id"].astype(str).map(
                lambda s: patch_tensor_path(self.outputs, s).exists()
            )]

        dropped = len(cohort) - len(usable)
        if dropped:
            logger.info(
                "Restricting the cohort to processed sessions: %d of %d "
                "labelled session(s) have extracted features%s.",
                len(usable), len(cohort),
                " and ROI patches" if require_patches else "",
            )
            print(
                f"\nNOTE: {len(usable)} of {len(cohort)} labelled session(s) "
                f"have been processed; the remaining {dropped} are excluded "
                "from the split. Run `--mode preprocess` on the full dataset "
                "to use all of them."
            )

        counts = {
            stage: int((usable["stage"] == stage).sum())
            for stage in STAGE_ORDER
        }
        if min(counts.values()) < 2:
            raise ValueError(
                f"Too few processed sessions for a stratified split: {counts}. "
                "Each stage needs at least 2. Process more subjects with "
                "`--mode preprocess`."
            )
        return usable.reset_index(drop=True)

    def stamp(self, mode: str, seed_report: Any) -> None:
        """Write the config snapshot and seed report for reproducibility."""
        experiment = self.outputs / "experiments" / self.cfg.paths.experiment_id
        experiment.mkdir(parents=True, exist_ok=True)
        self.cfg.to_file(experiment / "config.json")
        (experiment / "environment.json").write_text(
            json.dumps(
                {"mode": mode, "seed_report": seed_report.to_dict()}, indent=2
            ),
            encoding="utf-8",
        )

    def build_datasets(self, split: Any, use_cnn: bool) -> Tuple[Any, Any, Any]:
        """Build train/val/test datasets for one split.

        The scaler is fitted on **this split's** training sessions and nothing
        else, which is what keeps repeated-split evaluation leakage-free.
        """
        from modules.m04_feature_extraction import MorphometricScaler
        from modules.m06_spatial_encoder.patch_dataset import ROIPatchDataset

        raw = self.features()
        scaler = MorphometricScaler().fit(
            raw, train_session_ids=split.train_sessions
        )
        table = scaler.transform(scaler.add_atrophy_index(raw))
        cohort = self.trainable_cohort()

        def make(sessions: List[str]) -> ROIPatchDataset:
            array, _ = scaler.to_tensor_array(table, sessions)
            return ROIPatchDataset(
                sessions, cohort, array, self.outputs, load_patches=use_cnn
            )

        return (make(split.train_sessions), make(split.val_sessions),
                make(split.test_sessions))


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────

def mode_validate_dataset(ctx: Context, args: argparse.Namespace) -> int:
    """Validate the uploaded OASIS-1 dataset (Sections 6, 17, 18).

    Runs before anything else and is the gate that decides whether a research
    run is possible at all. Produces the validation report and dataset summary
    the dashboard displays.
    """
    from modules.m01_dataset import save_validation, validate_oasis1

    log_banner(logger, "OASIS-1 dataset validation")
    try:
        manager = ctx.oasis1_manager()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    if not manager.root_exists():
        print(f"\n{manager.missing_data_message()}")
        return 1

    report, table = validate_oasis1(
        manager,
        metadata_csv=Path(ctx.cfg.paths.metadata_csv),
        cdr_to_stage=ctx.cfg.data.cdr_to_stage,
        missing_cdr_policy=ctx.cfg.data.missing_cdr_policy,
        deep=ctx.cfg.data.deep_validation,
    )
    written = save_validation(report, table, ctx.outputs / "dataset_validation")

    # Record the dataset provenance alongside the validation artifacts so any
    # later consumer can trace a result back to the exact volumes used.
    (ctx.outputs / "dataset_validation" / "oasis1_provenance.json").write_text(
        json.dumps(manager.provenance(), indent=2), encoding="utf-8"
    )

    print()
    print(report.summary())
    print()
    for name, path in written.items():
        print(f"  {name}: {path}")

    if not report.passed:
        print("\nDATASET VALIDATION FAILED. Training is disabled until the "
              "errors above are resolved.")
        return 1
    if report.is_subset:
        print("\nOASIS-1 SUBSET MODE: results will describe the uploaded "
              "subset, not the complete dataset.")
    return 0


# ── Parallel preprocessing (Section 13) ──────────────────────────────────────
#
# The 235 OASIS-1 subjects are independent, so M1-M8 parallelises across
# processes. Three properties make that safe here rather than merely faster:
#
# * **Every write is per-subject.** ``preprocessing/<sid>/``, ``roi/<sid>/``,
#   ``features/workdir/<sid>*`` and ``state/subjects/<sid>.json`` are all keyed
#   by subject, so two workers never touch the same file. The cohort-level
#   feature table is assembled by the parent after every worker has returned.
# * **A failure is contained.** A worker returns a failure record instead of
#   raising, so one bad subject is logged and retried later rather than taking
#   the run down.
# * **Thread oversubscription is capped.** SimpleITK's N4 defaults to every
#   logical core, so N workers each spawning 20 threads would thrash. Each
#   worker caps itself at ``cores // workers``.
#
# Determinism is unaffected: each subject's imaging result depends only on its
# own volume, and the parent re-sorts results into cohort order before
# assembling the table, so the output does not depend on completion order.

def _worker_thread_cap(n_workers: int) -> int:
    """Threads each worker may use, so N workers do not oversubscribe cores."""
    import os

    cores = os.cpu_count() or 1
    return max(1, cores // max(1, n_workers))


def _preprocess_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run M1-M8 for one subject. Executes in a separate process.

    Must stay a module-level function: Windows spawns rather than forks, so the
    callable has to be importable by name in the child.

    Args:
        task: ``config_path``, ``subject_id``, ``mri_path``, ``etiv``,
            ``resume`` and ``threads``.

    Returns:
        A record with ``subject_id``, ``ok``, and either ``features`` (row
        dicts) or ``stage``/``error`` naming where it failed.
    """
    subject_id = str(task["subject_id"])
    out: Dict[str, Any] = {"subject_id": subject_id, "ok": False,
                           "stage": None, "error": None,
                           "features": [], "resumed": False, "warnings": []}
    try:
        threads = int(task.get("threads") or 0)
        if threads > 0:
            try:
                import SimpleITK as sitk

                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
            except Exception:  # pragma: no cover - SimpleITK optional
                pass
            try:
                import torch

                torch.set_num_threads(threads)
            except Exception:  # pragma: no cover
                pass

        import nibabel as nib

        from modules.m02_preprocessing.pipeline import PreprocessingPipeline
        from modules.m03_segmentation import ROIPipeline
        from modules.m04_feature_extraction import MorphometricFeatureExtractor

        cfg = NeuroGenesisConfig.from_file(Path(task["config_path"]))
        outputs = Path(cfg.paths.outputs_dir)
        tracker = RunStateTracker(outputs)
        pre_dir = outputs / "preprocessing"

        cached = (
            _cached_preprocessing(
                pre_dir, subject_id, cfg.preprocess.target_shape
            ) if task.get("resume") else None
        )
        if cached is not None:
            final_path = cached["final_path"]
            final_affine = cached.get("final_affine")
            out["resumed"] = True
        else:
            pre = PreprocessingPipeline(cfg.preprocess, outputs, tracker).run(
                subject_id, Path(task["mri_path"])
            )
            pre.save(pre_dir / subject_id)
            out["warnings"] = list(pre.warnings)
            if not pre.succeeded or pre.final_path is None:
                out["stage"] = "preprocessing"
                out["error"] = pre.error or "unknown"
                return out
            final_path = pre.final_path
            final_affine = pre.final_affine

        image = nib.load(final_path)
        volume = np.asarray(image.get_fdata(), dtype=np.float32)
        affine = (np.asarray(final_affine) if final_affine is not None
                  else image.affine)

        seg, patches = ROIPipeline(
            cfg.preprocess, outputs, tracker,
            atlas_dir=cfg.paths.atlas_dir,
        ).run(subject_id, volume, affine)
        seg.save(outputs / "roi" / subject_id)
        if not seg.succeeded:
            out["stage"] = "roi"
            out["error"] = seg.error or "unknown"
            return out

        extractor = MorphometricFeatureExtractor(
            outputs / "features" / "workdir",
            gm_threshold=cfg.preprocess.gm_threshold,
        )
        etiv = task.get("etiv")
        with tracker.stage("M8", subject_id) as record:
            frame = extractor.extract_subject(
                patches, subject_id,
                etiv=float(etiv) if etiv is not None else None,
            )
            if record is not None:
                record.record_metric("n_rois", len(patches))
        out["features"] = frame.to_dict(orient="records")
        out["ok"] = True
        return out
    except BaseException as exc:  # noqa: BLE001 - must not kill the pool
        import traceback

        out["stage"] = out["stage"] or "worker"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        return out


def _cached_preprocessing(
    out_dir: Path, subject_id: str, target_shape: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Return a completed M1-M5 manifest for ``subject_id``, or ``None``.

    M3 (N4 bias-field correction) dominates the imaging cost, so a full-cohort
    preprocessing pass runs for many hours. Without this, an interruption -- a
    machine sleeping, a terminated shell -- throws away every subject already
    finished, because the feature table is only assembled at the end.

    A cached result is only honoured when the manifest says the subject
    succeeded *and* the standardised volume it names is still on disk, so a
    half-written or manually deleted artifact is recomputed rather than trusted.
    Nothing about the science changes: M6-M8 re-run from the same M5 volume and
    produce the same values they would have on an uninterrupted pass.
    """
    path = Path(out_dir) / subject_id / f"{subject_id}_preprocessing.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not payload.get("succeeded") or not payload.get("final_path"):
        return None
    if not Path(payload["final_path"]).exists():
        return None
    # Reject a volume standardised under different settings. Without this, a
    # changed `preprocess.target_shape` would silently mix grids across the
    # cohort -- every subject processed before the change keeping the old one.
    final_stage = (payload.get("stages") or [{}])[-1]
    cached_shape = final_stage.get("shape")
    if cached_shape is not None and list(cached_shape) != list(target_shape):
        logger.warning(
            "%s was standardised to %s but the config asks for %s — "
            "recomputing rather than reusing it.",
            subject_id, cached_shape, list(target_shape),
        )
        return None
    return payload


def _resolve_workers(requested: Optional[int], n_tasks: int) -> int:
    """Decide how many preprocessing worker processes to run.

    ``None`` or ``0`` means auto: enough workers to keep the machine busy
    without starving each one of the threads N4 needs. N4 scales poorly past a
    handful of threads, so several subjects at four threads each beats one
    subject at twenty.

    Args:
        requested: The ``--workers`` value, or ``None``/``0`` for auto.
        n_tasks: How many subjects there are to process.

    Returns:
        A worker count of at least 1 and never more than ``n_tasks``.
    """
    import os

    cores = os.cpu_count() or 1
    if requested and int(requested) > 0:
        chosen = int(requested)
    else:
        chosen = max(1, min(cores // 4, 6))
    return max(1, min(chosen, max(1, n_tasks)))


def _save_preprocessing_manifest(
    ctx: "Context",
    tasks: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, str]],
) -> Path:
    """Write the Section 24 accounting of every subject the run touched.

    A subject is never silently dropped: each one appears here as processed,
    resumed or failed, with the stage and error that stopped it, so the gap
    between the cohort size and the analysable size is always explainable.

    Returns:
        Path of the written manifest.
    """
    from modules.common.serialization import dump_json

    by_id = {str(r["subject_id"]): r for r in results}
    rows = []
    for task in tasks:
        sid = str(task["subject_id"])
        record = by_id.get(sid)
        rows.append({
            "subject_id": sid,
            "mri_path": task["mri_path"],
            "status": ("missing" if record is None else
                       "resumed" if record.get("resumed") and record["ok"] else
                       "processed" if record["ok"] else "failed"),
            "failed_stage": None if record is None or record["ok"]
            else record.get("stage"),
            "error": None if record is None or record["ok"]
            else record.get("error"),
        })

    cohort = ctx.cohort()
    stage_of = dict(zip(cohort["session_id"].astype(str), cohort["stage"]))
    ok_ids = [r["subject_id"] for r in rows if r["status"] in
              ("processed", "resumed")]
    class_counts: Dict[str, int] = {}
    for sid in ok_ids:
        stage = str(stage_of.get(sid, "unknown"))
        class_counts[stage] = class_counts.get(stage, 0) + 1

    payload = {
        "dataset_source": ctx.cfg.data.dataset_source,
        "synthetic_data_enabled": bool(ctx.cfg.data.allow_synthetic_data),
        "n_requested": len(tasks),
        "n_succeeded": len(ok_ids),
        "n_failed": len(failures),
        "class_counts_processed": class_counts,
        "failures": list(failures),
        "subjects": rows,
    }
    path = dump_json(payload, ctx.outputs / "preprocessing" /
                     "preprocessing_manifest.json")
    print("")
    print(f"Subjects requested : {len(tasks)}")
    print(f"Succeeded          : {len(ok_ids)}")
    print(f"Failed             : {len(failures)}")
    for stage in STAGE_ORDER:
        print(f"  {stage}: {class_counts.get(stage, 0)}")
    if failures:
        print("")
        print("Failed subjects (retry with --resume; completed work is kept):")
        for failure in failures[:20]:
            print(f"  {failure['subject']} [{failure['stage']}]: "
                  f"{failure['error']}")
    print(f"Manifest           : {path}")
    return path


def mode_preprocess(ctx: Context, args: argparse.Namespace) -> int:
    """M1-M8: run the imaging pipeline and extract morphometric features."""
    from modules.m02_preprocessing import check_imaging_dependencies
    from modules.m02_preprocessing.pipeline import PreprocessingPipeline
    from modules.m03_segmentation import ROIPipeline
    from modules.m04_feature_extraction import (
        MorphometricFeatureExtractor,
        save_features,
    )

    log_banner(logger, "M1-M8  Preprocessing, ROI extraction, features")

    # Section 17: a research run requires real OASIS-1. If it is not present we
    # stop here rather than falling back to anything.
    if ctx.cfg.paths.oasis1_root:
        manager = ctx.oasis1_manager()
        if not manager.root_exists() or not manager.discover():
            print(f"\n{manager.missing_data_message()}")
            return 1
        print(f"\nDATASET   : {manager.provenance()['dataset_source']}")
        print(f"SOURCE    : {manager.provenance()['source_description']}")
        print("DATA MODE : REAL DATA")
        print("SYNTHETIC : DISABLED")

    # Stamp synthetic provenance into the outputs tree before anything is
    # written. A run over phantom MRI otherwise produces artifacts that are
    # indistinguishable from a real run. With a real OASIS-1 root this is a
    # no-op, and the tree carries no synthetic marker.
    from modules.common.provenance import stamp_outputs

    stamped = stamp_outputs(ctx.outputs, ctx.cfg.paths.mri_dir)
    if stamped is not None:
        print(f"\nProvenance stamped: {stamped}")
        print(f"  {ctx.smoke_marker.get('warning', '')}\n")

    cohort = ctx.cohort()

    imaging = check_imaging_dependencies()
    processable = cohort[cohort["has_mri"]] if "has_mri" in cohort.columns \
        else cohort
    if args.patient_id:
        processable = processable[
            processable["session_id"].astype(str) == args.patient_id
        ]
    if ctx.cfg.preprocess.max_subjects:
        processable = processable.head(ctx.cfg.preprocess.max_subjects)

    print(f"\nCohort: {len(cohort)} labelled session(s); "
          f"{len(processable)} with an MRI volume on disk.")
    for stage in STAGE_ORDER:
        print(f"  {stage}: {int((cohort['stage'] == stage).sum())}")

    if not imaging["can_run"]:
        print(f"\n{imaging['message']}")
        print(
            "\nThe cohort and labels are ready, but no imaging can be "
            "performed. To validate the model, training, XAI, statistics and "
            "dashboard code paths without MRI data, generate clearly-labelled "
            "synthetic artifacts:\n"
            "    python tools/make_smoke_artifacts.py --out outputs_smoke\n"
            "    python run.py --mode train_full --outputs outputs_smoke"
        )
        return 1
    if processable.empty:
        print(
            f"\nNo MRI volumes found under {ctx.cfg.paths.mri_dir}. Place "
            "OASIS-1 T1 volumes there and re-run."
        )
        return 1

    preprocessor = PreprocessingPipeline(
        ctx.cfg.preprocess, ctx.outputs, ctx.tracker
    )
    roi_pipeline = ROIPipeline(
        ctx.cfg.preprocess, ctx.outputs, ctx.tracker,
        atlas_dir=ctx.cfg.paths.atlas_dir,
    )
    extractor = MorphometricFeatureExtractor(
        ctx.outputs / "features" / "workdir",
        gm_threshold=ctx.cfg.preprocess.gm_threshold,
    )

    frames: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    pre_dir = ctx.outputs / "preprocessing"
    n_resumed = 0

    # Build one task per subject. Order is the cohort's, and results are
    # re-sorted into it below, so the assembled table does not depend on which
    # worker finished first.
    # Each worker rebuilds the configuration from a snapshot rather than
    # unpickling it, so a worker can never run under a different
    # configuration than the one this run recorded.
    snapshot = ctx.outputs / "tmp" / "preprocess_config.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    ctx.cfg.to_file(snapshot)

    tasks: List[Dict[str, Any]] = []
    for _, row in processable.iterrows():
        etiv = row.get("eTIV")
        try:
            etiv_value: Optional[float] = float(etiv)
        except (TypeError, ValueError):
            etiv_value = None
        tasks.append({
            "config_path": str(snapshot),
            "subject_id": str(row["session_id"]),
            "mri_path": str(row["mri_path"]),
            "etiv": etiv_value,
            "resume": bool(args.resume),
        })

    order = {t["subject_id"]: i for i, t in enumerate(tasks)}
    n_workers = _resolve_workers(args.workers, len(tasks))
    results: List[Dict[str, Any]] = []


    if n_workers > 1:
        threads = _worker_thread_cap(n_workers)
        for task in tasks:
            task["threads"] = threads
        print(
            f"\nPreprocessing {len(tasks)} subject(s) with {n_workers} "
            f"worker(s), {threads} thread(s) each."
        )
        # Warm the atlas cache in the parent. Workers only read it after this,
        # so concurrent first-use cannot race on a download.
        try:
            from nilearn.datasets import fetch_atlas_harvard_oxford

            fetch_atlas_harvard_oxford(
                ctx.cfg.preprocess.atlas_name,
                data_dir=str(ctx.cfg.paths.atlas_dir),
            )
        except Exception as exc:  # noqa: BLE001 - a worker will report it
            logger.warning("Could not pre-warm the atlas cache: %s", exc)

        import concurrent.futures as cf

        done = 0
        with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_preprocess_worker, t): t for t in tasks}
            for future in cf.as_completed(futures):
                record = future.result()
                results.append(record)
                done += 1
                mark = "ok" if record["ok"] else f"FAILED [{record['stage']}]"
                logger.info("(%d/%d) %s %s%s", done, len(tasks),
                            record["subject_id"], mark,
                            " (resumed)" if record.get("resumed") else "")
    else:
        print(f"\nPreprocessing {len(tasks)} subject(s) with 1 worker.")
        for i, task in enumerate(tasks, 1):
            record = _preprocess_worker(task)
            results.append(record)
            mark = "ok" if record["ok"] else f"FAILED [{record['stage']}]"
            logger.info("(%d/%d) %s %s%s", i, len(tasks),
                        record["subject_id"], mark,
                        " (resumed)" if record.get("resumed") else "")

    results.sort(key=lambda r: order.get(r["subject_id"], 0))
    for record in results:
        if record.get("resumed"):
            n_resumed += 1
        if record["ok"]:
            frames.append(pd.DataFrame(record["features"]))
        else:
            failures.append({"subject": record["subject_id"],
                             "stage": str(record["stage"]),
                             "error": str(record["error"])})
            if record.get("traceback"):
                logger.debug(
                    "%s traceback:\n%s",
                    record["subject_id"], record["traceback"],
                )

    # Section 24: every subject is accounted for, none silently discarded.
    _save_preprocessing_manifest(ctx, tasks, results, failures)

    if not frames:
        print("\nNo subject completed feature extraction. Failures:")
        for failure in failures[:10]:
            print(f"  {failure['subject']} [{failure['stage']}]: "
                  f"{failure['error']}")
        return 1

    features = pd.concat(frames, ignore_index=True)
    report = extractor.audit(features)
    written = save_features(features, report, ctx.outputs / "features")
    print(f"\nFeatures written: {written['features_csv']}")
    if n_resumed:
        print(f"Resumed {n_resumed} subject(s) from completed M1-M5 output.")
    print(report.summary())
    if failures:
        print(f"\n{len(failures)} subject(s) failed; see the log for details.")
    return 0


def mode_train_cnn(ctx: Context, args: argparse.Namespace) -> int:
    """M9: cache 3D CNN embeddings for every subject with a patch tensor.

    The encoder is untrained at this point, so the cached embeddings are only
    useful for inspecting the encoder's behaviour and for the dashboard M9 panel.
    They are **not** used by ``--mode train_full``, which trains the encoder
    jointly with the rest of the model. The caveat is written into the manifest
    so a cached embedding cannot be mistaken for a trained representation.
    """
    from modules.m06_spatial_encoder.cnn3d import SpatialEncoder3D, save_embeddings
    from modules.m06_spatial_encoder.patch_dataset import patch_tensor_path

    log_banner(logger, "M9  3D CNN spatial encoding")
    encoder = SpatialEncoder3D(ctx.cfg.spatial_encoder)
    print(encoder.describe())

    cohort = ctx.cohort()
    sessions = [
        str(s) for s in cohort["session_id"]
        if patch_tensor_path(ctx.outputs, str(s)).exists()
    ]
    if args.patient_id:
        sessions = [s for s in sessions if s == args.patient_id]
    if not sessions:
        print(
            "\nNo ROI patch tensors found. Run `--mode preprocess` first, or "
            "generate synthetic artifacts with tools/make_smoke_artifacts.py."
        )
        return 1

    checkpoint = ctx.outputs / "checkpoints" / "cnn3d_untrained.pt"
    encoder.save_checkpoint(checkpoint, extra={
        "status": "untrained",
        "caveat": "This encoder has not been trained. Embeddings cached from it "
                  "describe the random initialisation, not learned structure.",
    })

    for subject_id in sessions:
        volume = np.load(patch_tensor_path(ctx.outputs, subject_id))
        with ctx.tracker.stage("M9", subject_id) as record:
            output = encoder.encode(volume, trace=True)
            written = save_embeddings(
                output, ctx.outputs / "cnn_embeddings" / subject_id, subject_id
            )
            if record is not None:
                record.record_metric(
                    "embed_dim", int(output.embeddings.shape[-1])
                )
                record.record_metric("encoder_status", "untrained")
                for name, path in written.items():
                    record.record_artifact(name, path)

    print(f"\nCached embeddings for {len(sessions)} subject(s) under "
          f"{ctx.outputs / 'cnn_embeddings'}")
    print(
        "\nNOTE: this encoder is untrained. These embeddings exist for "
        "inspection only; `--mode train_full` trains the encoder jointly with "
        "the rest of the model and does not read them."
    )
    return 0


def _train(ctx: Context, variant: str, epochs: Optional[int],
           use_cnn_override: Optional[bool] = None) -> int:
    """Shared training routine for ``train_graph`` and ``train_full``."""
    from modules.m01_dataset import make_subject_split
    from modules.m04_feature_extraction import FEATURE_ORDER, MorphometricScaler
    from modules.m06_spatial_encoder.patch_dataset import make_loader
    from modules.model import build_model
    from modules.training import Trainer

    model_probe = build_model(
        len(FEATURE_ORDER), ctx.cfg, variant, list(FEATURE_ORDER)
    )
    cohort = ctx.trainable_cohort(require_patches=model_probe.spec.use_cnn)
    features = ctx.features()

    split = make_subject_split(
        cohort,
        val_fraction=ctx.cfg.data.val_fraction,
        test_fraction=ctx.cfg.data.test_fraction,
        seed=ctx.cfg.repro.seed,
    )
    leaks = split.verify_disjoint()
    if leaks:
        print("LEAKAGE DETECTED - aborting:")
        for leak in leaks:
            print(f"  {leak}")
        return 1
    split.save(ctx.split_path)
    split.save_split_csvs(cohort, ctx.outputs / "splits")

    # Section 8: nothing reaches a model until every session is proven
    # to come from the validated OASIS-1 index.
    ctx.enforce_integrity(
        list(split.train_sessions) + list(split.val_sessions)
        + list(split.test_sessions),
        split=split,
    )

    print("\nTABLE 1  Dataset and split distribution")
    print(split.table1().to_string(index=False))
    for warning in split.warnings:
        print(f"WARNING: {warning}")

    scaler = MorphometricScaler().fit(
        features, train_session_ids=split.train_sessions
    )
    scaler.save(ctx.scaler_path)
    model = model_probe
    use_cnn = model.spec.use_cnn if use_cnn_override is None else use_cnn_override
    train_ds, val_ds, test_ds = ctx.build_datasets(split, use_cnn)

    print(f"\nVariant {variant}: {model.n_parameters():,} trainable parameters")
    print(f"Datasets  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}")
    print(f"Class counts  train={train_ds.class_counts()}  "
          f"val={val_ds.class_counts()}  test={test_ds.class_counts()}")

    trainer = Trainer(model, ctx.cfg)
    provenance = None
    if ctx.cfg.paths.oasis1_root:
        provenance = ctx.oasis1_manager().provenance()
    result = trainer.fit(
        make_loader(train_ds, ctx.cfg.train.batch_size, shuffle=True,
                    seed=ctx.cfg.repro.seed,
                    drop_last=len(train_ds) > 2 * ctx.cfg.train.batch_size),
        make_loader(val_ds, ctx.cfg.train.batch_size),
        checkpoint_path=ctx.checkpoint_path(variant),
        epochs=epochs,
        dataset_provenance=provenance,
        split_file=ctx.split_path,
    )
    Trainer.save_result(
        result,
        ctx.outputs / "experiments" / ctx.cfg.paths.experiment_id
        / f"training_{variant}.json",
    )

    print(f"\nBest epoch {result.best_epoch} | {result.monitor} = "
          f"{result.best_value}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if result.stage_geometry:
        print(f"Stage prototype ordering respected: "
              f"{result.stage_geometry['ordering_respected']} "
              f"(geometry tie-breaks: {result.n_geometry_tie_breaks})")

    evaluation = trainer.evaluate(make_loader(test_ds, ctx.cfg.train.batch_size))
    banner = (ctx.smoke_marker.get("banner", "SYNTHETIC DATA")
              if ctx.smoke_marker else "TEST-SPLIT METRICS")
    print(f"\n=== {banner} ===")
    print(evaluation["metrics"].summary())

    predictions = pd.DataFrame(evaluation["predictions"])
    out_dir = ctx.outputs / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / f"test_predictions_{variant}.csv", index=False)
    (out_dir / f"test_metrics_{variant}.json").write_text(
        json.dumps({
            "variant": variant,
            "is_synthetic": bool(ctx.smoke_marker),
            "metrics": evaluation["metrics"].to_dict(),
            "loss": evaluation["loss"],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nPredictions written to {out_dir}")
    return 0


def mode_train_graph(ctx: Context, args: argparse.Namespace) -> int:
    """Train the graph stage without the 3D CNN branch (ablation A5)."""
    log_banner(logger, "Graph-stage training (A5: NeuroProp-X + SAEG-GATv2)")
    return _train(ctx, args.variant or "A5", args.epochs)


def mode_train_full(ctx: Context, args: argparse.Namespace) -> int:
    """Train the full model end to end (A7)."""
    log_banner(logger, "Full model training (A7)")
    return _train(ctx, args.variant or "A7", args.epochs)


def mode_ablation(ctx: Context, args: argparse.Namespace) -> int:
    """A0-A7 ablation plus classical baselines over repeated splits."""
    from modules.m04_feature_extraction import MorphometricScaler
    from modules.training.ablation import AblationStudy
    from modules.training.baselines import BaselineStudy

    log_banner(logger, "M18  Ablation and baseline study")
    # The ladder includes CNN variants, so every session needs a patch
    # tensor as well as features.
    cohort = ctx.trainable_cohort(require_patches=True)
    features = ctx.features()

    study = AblationStudy(
        cohort=cohort, outputs_root=ctx.outputs, cfg=ctx.cfg,
        dataset_factory=ctx.build_datasets,
    )
    with ctx.tracker.stage("M18") as record:
        summaries = study.run(
            variants=args.variants.split(",") if args.variants else None,
            n_repeats=args.repeats or ctx.cfg.data.n_repeats,
            epochs=args.epochs,
        )
        written = study.save(ctx.outputs / "ablation")
        if record is not None:
            record.record_metric("n_variants", len(summaries))
            record.record_metric("n_repeats", len(study.splits))
            for name, path in written.items():
                record.record_artifact(name, path)

    banner = (ctx.smoke_marker.get("banner", "SYNTHETIC DATA")
              if ctx.smoke_marker else "ABLATION RESULTS")
    print(f"\n=== {banner} ===")
    print("\nTABLE 4  NeuroProp-X ablation")
    print(study.table4().to_string(index=False))
    print("\nTABLE 2  Main classification performance")
    print(study.table2().to_string(index=False))
    print("\nTABLE 3  Class-wise performance (A7)")
    print(study.table3("A7").to_string(index=False))
    print("\nTABLE 9  Final model comparison")
    print(study.table9().to_string(index=False))
    for caveat in study.caveats():
        print(f"\n> {caveat}")

    # Classical baselines over the same splits.
    labels = dict(zip(cohort["session_id"].astype(str), cohort["label"]))
    splits_data = []
    for split in study.splits:
        scaler = MorphometricScaler().fit(
            features, train_session_ids=split.train_sessions
        )
        table = scaler.transform(scaler.add_atrophy_index(features))

        def matrix(sessions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
            array, _ = scaler.to_tensor_array(table, sessions)
            return (array.reshape(len(sessions), -1),
                    np.array([labels[s] for s in sessions]))

        x_train, y_train = matrix(split.train_sessions)
        x_test, y_test = matrix(split.test_sessions)
        splits_data.append((split.repeat or 0, x_train, y_train, x_test, y_test))

    baseline_study = BaselineStudy()
    baseline_study.run(splits_data)
    baseline_study.save(ctx.outputs / "baselines")
    print("\nBASELINES")
    print(baseline_study.table().to_string(index=False))
    return 0


def mode_evaluate(ctx: Context, args: argparse.Namespace) -> int:
    """Evaluate a saved checkpoint on the recorded test split."""
    from modules.m01_dataset.splits import SplitManifest
    from modules.m06_spatial_encoder.patch_dataset import make_loader
    from modules.model import NeuroGenesisModel
    from modules.training import Trainer

    log_banner(logger, "Evaluation")
    variant = args.variant or "A7"
    checkpoint = ctx.checkpoint_path(variant)
    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}. Run `--mode train_full` first.")
        return 1
    if not ctx.split_path.exists():
        print(
            f"No split manifest at {ctx.split_path}. Evaluation must use the "
            "split the model was trained on; re-running the split here would "
            "risk evaluating on training subjects."
        )
        return 1

    split = SplitManifest.load(ctx.split_path)
    model = NeuroGenesisModel.load_checkpoint(checkpoint)
    # Evaluation reuses the recorded split, so the cohort filter only needs
    # to make the feature lookup succeed for those sessions.
    _, _, test_ds = ctx.build_datasets(split, model.spec.use_cnn)
    trainer = Trainer(model, ctx.cfg)
    evaluation = trainer.evaluate(make_loader(test_ds, ctx.cfg.train.batch_size))

    banner = (ctx.smoke_marker.get("banner", "SYNTHETIC DATA")
              if ctx.smoke_marker else "TEST-SPLIT METRICS")
    print(f"\n=== {banner} ===")
    print(evaluation["metrics"].summary())
    print("\nConfusion matrix (rows = true CN/MCI/AD):")
    print(evaluation["metrics"].confusion_matrix)
    return 0


def mode_statistics(ctx: Context, args: argparse.Namespace) -> int:
    """M17: stage-wise statistical analysis with FDR correction."""
    from modules.m09_statistics import run_stagewise_analysis

    log_banner(logger, "M17  Stage-wise statistical analysis")
    with ctx.tracker.stage("M17") as record:
        report = run_stagewise_analysis(
            ctx.features(), ctx.cohort(), ctx.cfg.stats
        )
        out_dir = ctx.outputs / "statistics"
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_frame().to_csv(out_dir / "stagewise_tests.csv", index=False)
        report.table5().to_csv(out_dir / "table5_stagewise.csv", index=False)
        report.table7().to_csv(out_dir / "table7_morphometry.csv", index=False)
        (out_dir / "statistics_report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        if record is not None:
            record.record_metric("n_tests_run", report.n_tests_run)
            record.record_metric("n_significant", report.n_significant)

    print(report.summary())
    print("\nTABLE 5  Stage-wise ROI statistical analysis (first rows)")
    table = report.table5()
    columns = [c for c in ("ROI", "Feature", "CN vs MCI p", "MCI vs AD p",
                           "CN vs AD p", "FDR p", "Effect Size", "Trend")
               if c in table.columns]
    print(table[columns].head(12).to_string(index=False))
    print(f"\nWritten to {ctx.outputs / 'statistics'}")
    return 0


def mode_xai(ctx: Context, args: argparse.Namespace) -> int:
    """M15-M16: explanations and ROI ranking."""
    from modules.inference import load_inference_pipeline
    from modules.m08_xai import ranking_stability, stagewise_rankings
    from modules.m08_xai.roi_ranking import ROIRanking

    log_banner(logger, "M15-M16  ROI ranking and explainable AI")
    variant = args.variant or "A7"
    provenance = (
        ctx.oasis1_manager().provenance() if ctx.cfg.paths.oasis1_root else None
    )
    pipeline = load_inference_pipeline(
        ctx.outputs, ctx.checkpoint_path(variant), ctx.cfg,
        scaler_path=ctx.scaler_path, split_path=ctx.split_path,
        smoke_marker=ctx.smoke_marker, dataset_provenance=provenance,
    )
    cohort = ctx.trainable_cohort()
    sessions = [args.patient_id] if args.patient_id else [
        str(s) for s in cohort["session_id"]
    ]
    if args.limit:
        sessions = sessions[:args.limit]

    per_subject: Dict[str, Tuple[str, ROIRanking]] = {}
    rankings: List[ROIRanking] = []
    records: List[Dict[str, Any]] = []

    for subject_id in sessions:
        try:
            result = pipeline.run(
                subject_id, explain=True, occlusion=not args.no_occlusion,
                tracker=ctx.tracker,
            )
        except (KeyError, FileNotFoundError) as exc:
            logger.warning("Skipping %s: %s", subject_id, exc)
            continue
        records.append(result.record)
        entries = result.record.get("roi_ranking") or []
        if entries:
            ranking = ROIRanking(
                scores={e["roi"]: e["score"] for e in entries}, ranking=entries
            )
            rankings.append(ranking)
            per_subject[subject_id] = (
                result.record.get("reference_stage") or "unknown", ranking
            )

    out_dir = ctx.outputs / "roi_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_subject_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    if rankings:
        stability = ranking_stability(rankings, top_k=3)
        (out_dir / "table8_ranking_stability.json").write_text(
            json.dumps(stability.to_dict(), indent=2), encoding="utf-8"
        )
        pd.DataFrame(stability.rows).to_csv(
            out_dir / "table8_ranking_stability.csv", index=False
        )
        print("\nTABLE 8  ROI ranking stability")
        print(pd.DataFrame(stability.rows).to_string(index=False))

        stage_rankings = stagewise_rankings(per_subject)
        rows = []
        for rank_index in range(5):
            row: Dict[str, Any] = {"Rank": rank_index + 1}
            for stage in STAGE_ORDER:
                ranking = stage_rankings.get(stage)
                row[stage] = (
                    ranking.ranking[rank_index]["roi_short"]
                    if ranking and rank_index < len(ranking.ranking) else "n/a"
                )
            rows.append(row)
        table6 = pd.DataFrame(rows)
        table6.to_csv(out_dir / "table6_stagewise_ranking.csv", index=False)
        print("\nTABLE 6  Stage-wise ROI ranking")
        print(table6.to_string(index=False))
    else:
        print("\nNo ROI ranking could be produced for any subject.")
        return 1

    print(f"\nWritten to {out_dir}")
    return 0


def mode_report(ctx: Context, args: argparse.Namespace) -> int:
    """M19: generate subject reports."""
    from modules.inference import load_inference_pipeline

    log_banner(logger, "M19  Report generation")
    variant = args.variant or "A7"
    provenance = (
        ctx.oasis1_manager().provenance() if ctx.cfg.paths.oasis1_root else None
    )
    pipeline = load_inference_pipeline(
        ctx.outputs, ctx.checkpoint_path(variant), ctx.cfg,
        scaler_path=ctx.scaler_path, split_path=ctx.split_path,
        smoke_marker=ctx.smoke_marker, dataset_provenance=provenance,
    )

    extra: Dict[str, Any] = {}
    statistics_path = ctx.outputs / "statistics" / "statistics_report.json"
    if statistics_path.exists():
        payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        extra["statistics_summary"] = {
            k: v for k, v in payload.items() if k != "results"
        }
        extra["significant_findings"] = [
            r for r in payload.get("results", []) if r.get("significant_fdr")
        ][:10]
    ablation_path = ctx.outputs / "ablation" / "table4_ablation.csv"
    if ablation_path.exists():
        from modules.training.ablation import AblationStudy

        extra["ablation_table"] = pd.read_csv(ablation_path)
        extra["ablation_caveats"] = AblationStudy.caveats()
    extra["checkpoint_path"] = ctx.checkpoint_path(variant).as_posix()

    cohort = ctx.trainable_cohort()
    sessions = [args.patient_id] if args.patient_id else [
        str(s) for s in cohort["session_id"]
    ]
    if args.limit:
        sessions = sessions[:args.limit]

    generated = 0
    for subject_id in sessions:
        try:
            result = pipeline.run(subject_id, explain=True,
                                  occlusion=not args.no_occlusion,
                                  tracker=ctx.tracker)
            written = pipeline.generate_report(
                result, ctx.outputs / "reports", extra=extra,
                tracker=ctx.tracker,
            )
            generated += 1
            if args.patient_id:
                print(f"\n{written['markdown'].read_text(encoding='utf-8')}")
                print(f"\nMachine-readable record:\n{result.to_json()[:1500]}")
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.warning("Report failed for %s: %s", subject_id, exc)

    print(f"\nGenerated {generated} report(s) under {ctx.outputs / 'reports'}")
    return 0 if generated else 1


def mode_figures(ctx: Context, args: argparse.Namespace) -> int:
    """Generate the research figures from persisted results."""
    from modules.m10_results import figures as F

    log_banner(logger, "Figures")
    out_dir = ctx.outputs / "figures"
    marker = ctx.smoke_marker
    written: List[Path] = []

    experiment = ctx.outputs / "experiments" / ctx.cfg.paths.experiment_id
    variant = args.variant or "A7"
    training_path = experiment / f"training_{variant}.json"
    curves = None
    if training_path.exists():
        history = json.loads(training_path.read_text(encoding="utf-8"))
        epochs = history.get("history", [])
        curves = {
            "epoch": [e["epoch"] for e in epochs],
            "train_loss": [e["train_loss"] for e in epochs],
            "val_loss": [e.get("val_loss") for e in epochs],
            "train_accuracy": [e["train_accuracy"] for e in epochs],
            "val_accuracy": [e.get("val_accuracy") for e in epochs],
            "val_balanced_accuracy": [
                e.get("val_balanced_accuracy") for e in epochs
            ],
        }
    written.append(F.training_curves(curves, out_dir, marker))

    metrics_path = ctx.outputs / "predictions" / f"test_metrics_{variant}.json"
    predictions_path = ctx.outputs / "predictions" / \
        f"test_predictions_{variant}.csv"
    matrix = None
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw = payload.get("metrics", {}).get("confusion_matrix")
        matrix = np.array(raw) if raw else None
    written.append(F.confusion_matrix_figure(matrix, out_dir,
                                             smoke_marker=marker))

    predictions = pd.read_csv(predictions_path) if predictions_path.exists() \
        else None
    if predictions is not None:
        probability_columns = [f"p_{s}" for s in STAGE_ORDER]
        if all(c in predictions.columns for c in probability_columns):
            y_true = [STAGE_ORDER.index(s) for s in predictions["true_stage"]]
            written.append(F.roc_curves(
                y_true, predictions[probability_columns].values, out_dir, marker
            ))
    else:
        written.append(F.roc_curves(None, None, out_dir, marker))
    written.append(F.propensity_figure(predictions, out_dir, marker))

    try:
        features = ctx.features()
        cohort = ctx.cohort()
        for feature in ("gm_volume_mm3", "entropy"):
            written.append(
                F.stagewise_feature_map(features, cohort, feature, out_dir, marker)
            )
            written.append(
                F.feature_violin(features, cohort, feature, out_dir, marker)
            )
    except FileNotFoundError:
        written.append(F.stagewise_feature_map(None, None, "gm_volume_mm3",
                                               out_dir, marker))

    ablation_path = ctx.outputs / "ablation" / "ablation_results.json"
    summaries = None
    if ablation_path.exists():
        from modules.training.ablation import VariantSummary

        payload = json.loads(ablation_path.read_text(encoding="utf-8"))
        summaries = {
            key: VariantSummary(
                variant=value["variant"], description=value["description"],
                n_repeats=value["n_repeats"],
                n_parameters=value["n_parameters"],
                aggregate=value["aggregate"],
            )
            for key, value in payload.get("summaries", {}).items()
        }
    written.append(F.ablation_plot(summaries, out_dir, smoke_marker=marker))

    stability_path = ctx.outputs / "roi_ranking" / \
        "table8_ranking_stability.json"
    rows = None
    if stability_path.exists():
        rows = json.loads(
            stability_path.read_text(encoding="utf-8")
        ).get("rows")
    written.append(F.ranking_stability_plot(rows, out_dir, marker))

    print(f"\nGenerated {len(written)} figure(s) under {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


def mode_dashboard(ctx: Context, args: argparse.Namespace) -> int:
    """Launch the Streamlit pipeline inspection app."""
    import subprocess

    app = _ROOT / "dashboard" / "app.py"
    if not app.exists():
        print(f"Dashboard app not found at {app}")
        return 1
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        print(
            f"Streamlit cannot be imported ({exc}).\n"
            "Install or repair it with:\n"
            "    pip install --upgrade 'streamlit>=1.40' 'starlette>=0.40'\n"
            f"Then run:\n    streamlit run {app} -- --outputs {ctx.outputs}"
        )
        return 1

    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--", "--outputs", str(ctx.outputs),
    ]
    print(f"Launching: {' '.join(command)}")
    return subprocess.call(command)


def mode_status(ctx: Context, args: argparse.Namespace) -> int:
    """Print pipeline readiness and per-module execution state."""
    from modules.m02_preprocessing import check_imaging_dependencies
    from modules.training.baselines import available_baselines
    from modules.m08_xai import shap_available

    log_banner(logger, "Pipeline status")
    print(f"Outputs root: {ctx.outputs}")
    if ctx.smoke_marker:
        print("\n*** THIS OUTPUT TREE CONTAINS SYNTHETIC SMOKE-TEST DATA ***")
        print(f"    {ctx.smoke_marker.get('warning', '')}")

    print("\nEnvironment")
    imaging = check_imaging_dependencies()
    for package, ok in imaging["available"].items():
        print(f"  {'OK  ' if ok else 'MISS'} {package}")
    print(f"  {'OK  ' if shap_available() else 'MISS'} shap "
          "(SHAP attribution; permutation fallback otherwise)")
    for name, ok in available_baselines().items():
        print(f"  {'OK  ' if ok else 'MISS'} baseline: {name}")

    print("\nArtifacts")
    for label, path in (
        ("cohort table", ctx.cohort_path),
        ("feature table", ctx.features_path),
        ("fitted scaler", ctx.scaler_path),
        ("split manifest", ctx.split_path),
        ("A7 checkpoint", ctx.checkpoint_path("A7")),
        ("ablation results", ctx.outputs / "ablation" / "ablation_results.json"),
        ("statistics", ctx.outputs / "statistics" / "statistics_report.json"),
    ):
        print(f"  {'present' if path.exists() else 'absent '}  {label}: {path}")

    subjects = ctx.tracker.list_subjects()
    print(f"\nSubjects with recorded state: {len(subjects)}")
    target = args.patient_id or (subjects[0] if subjects else None)
    if target:
        print(f"\nPipeline state for {target}:")
        for row in ctx.tracker.summary(target):
            duration = (f"{row['duration_s']:.2f}s"
                        if row["duration_s"] is not None else "-")
            print(f"  {row['code']:<6s} {row['status']:<12s} {duration:>8s}  "
                  f"{row['title']}")
        print(f"\nCompletion: {ctx.tracker.completion_fraction(target):.1%}")
    else:
        print("\nNo subject has recorded pipeline state yet.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

MODE_TABLE = {
    "validate_dataset": mode_validate_dataset,
    "preprocess": mode_preprocess,
    "train_cnn": mode_train_cnn,
    "train_graph": mode_train_graph,
    "train_full": mode_train_full,
    "ablation": mode_ablation,
    "evaluate": mode_evaluate,
    "xai": mode_xai,
    "report": mode_report,
    "statistics": mode_statistics,
    "figures": mode_figures,
    "dashboard": mode_dashboard,
    "status": mode_status,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="NeuroGenesis: stage-aware speech-network NeuroAI framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=MODES,
                        help="Pipeline stage to run.")
    parser.add_argument("--config", type=Path,
                        help="Config YAML/JSON. Defaults are used when omitted.")
    parser.add_argument("--outputs", type=Path,
                        help="Override the outputs root directory.")
    parser.add_argument("--oasis1-root", type=Path,
                        help="Root of the extracted real OASIS-1 "
                             "dataset. Required for research runs.")
    parser.add_argument("--experiment", type=str,
                        help="Experiment ID for the config and checkpoint stamp.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Preprocessing worker processes. 0 (default) "
                             "picks a count from the core count. Subjects are "
                             "independent and every write is per-subject, so "
                             "this changes only throughput, never the values "
                             "produced.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed M1-M5 output for subjects that "
                             "already have a standardised volume on disk, "
                             "instead of recomputing it. Only affects which "
                             "work is repeated, never the values produced.")
    parser.add_argument("--patient_id", type=str,
                        help="Restrict the mode to a single session ID.")
    parser.add_argument("--variant", type=str,
                        help="Model variant (A0..A7, A7_no_tgt). Default A7.")
    parser.add_argument("--variants", type=str,
                        help="Comma-separated variants for --mode ablation.")
    parser.add_argument("--epochs", type=int,
                        help="Override the configured epoch budget.")
    parser.add_argument("--repeats", type=int,
                        help="Repeated splits for --mode ablation.")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    parser.add_argument("--limit", type=int,
                        help="Process at most N subjects (xai, report).")
    parser.add_argument("--no-occlusion", action="store_true",
                        help="Skip the 3D CNN occlusion attribution.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    cfg = NeuroGenesisConfig.from_file(args.config) if args.config \
        else NeuroGenesisConfig()
    if args.outputs:
        cfg.paths.outputs_dir = args.outputs
    if args.oasis1_root:
        cfg.paths.oasis1_root = args.oasis1_root
    if args.experiment:
        cfg.paths.experiment_id = args.experiment
    if args.seed is not None:
        cfg.repro.seed = args.seed

    problems = cfg.validate()
    if problems:
        print("Configuration is invalid:")
        for problem in problems:
            print(f"  {problem}")
        return 2

    import logging

    setup_logging(
        logs_dir=Path(cfg.paths.outputs_dir) / "logs",
        level=getattr(logging, args.log_level),
        run_name=args.mode,
    )
    seed_report = set_all_seeds(
        cfg.repro.seed, cfg.repro.deterministic, cfg.repro.cudnn_benchmark
    )
    for note in seed_report.notes:
        logger.info("Reproducibility: %s", note)

    ctx = Context(cfg)
    ctx.stamp(args.mode, seed_report)

    try:
        return MODE_TABLE[args.mode](ctx, args)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
