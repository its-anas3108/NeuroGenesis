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
from typing import Any, Dict, List, Optional, Tuple

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

    @property
    def smoke_marker(self) -> Optional[Dict[str, Any]]:
        """Return the synthetic-data marker if this outputs tree is a smoke test."""
        if self._smoke is None:
            from tools.make_smoke_artifacts import is_smoke_output

            self._smoke = is_smoke_output(self.outputs) or {}
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
        cohort = self.cohort()

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
    roi_pipeline = ROIPipeline(ctx.cfg.preprocess, ctx.outputs, ctx.tracker)
    extractor = MorphometricFeatureExtractor(
        ctx.outputs / "features" / "workdir",
        gm_threshold=ctx.cfg.preprocess.gm_threshold,
    )

    frames: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    for _, row in processable.iterrows():
        subject_id = str(row["session_id"])
        logger.info("Processing %s", subject_id)
        pre = preprocessor.run(subject_id, Path(row["mri_path"]))
        pre.save(ctx.outputs / "preprocessing" / subject_id)
        if not pre.succeeded or pre.final_path is None:
            failures.append({"subject": subject_id, "stage": "preprocessing",
                             "error": pre.error or "unknown"})
            continue

        import nibabel as nib

        image = nib.load(pre.final_path)
        volume = np.asarray(image.get_fdata(), dtype=np.float32)
        seg, patches = roi_pipeline.run(subject_id, volume, image.affine)
        seg.save(ctx.outputs / "roi" / subject_id)
        if not seg.succeeded:
            failures.append({"subject": subject_id, "stage": "roi",
                             "error": seg.error or "unknown"})
            continue

        with ctx.tracker.stage("M8", subject_id) as record:
            etiv = row.get("eTIV")
            try:
                etiv_value: Optional[float] = float(etiv)
            except (TypeError, ValueError):
                etiv_value = None
            frames.append(
                extractor.extract_subject(patches, subject_id, etiv=etiv_value)
            )
            if record is not None:
                record.record_metric("n_rois", len(patches))

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

    cohort = ctx.cohort()
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
    print("\nTABLE 1  Dataset and split distribution")
    print(split.table1().to_string(index=False))
    for warning in split.warnings:
        print(f"WARNING: {warning}")

    scaler = MorphometricScaler().fit(
        features, train_session_ids=split.train_sessions
    )
    scaler.save(ctx.scaler_path)
    model = build_model(
        len(FEATURE_ORDER), ctx.cfg, variant, list(FEATURE_ORDER)
    )
    use_cnn = model.spec.use_cnn if use_cnn_override is None else use_cnn_override
    train_ds, val_ds, test_ds = ctx.build_datasets(split, use_cnn)

    print(f"\nVariant {variant}: {model.n_parameters():,} trainable parameters")
    print(f"Datasets  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}")
    print(f"Class counts  train={train_ds.class_counts()}  "
          f"val={val_ds.class_counts()}  test={test_ds.class_counts()}")

    trainer = Trainer(model, ctx.cfg)
    result = trainer.fit(
        make_loader(train_ds, ctx.cfg.train.batch_size, shuffle=True,
                    seed=ctx.cfg.repro.seed,
                    drop_last=len(train_ds) > 2 * ctx.cfg.train.batch_size),
        make_loader(val_ds, ctx.cfg.train.batch_size),
        checkpoint_path=ctx.checkpoint_path(variant),
        epochs=epochs,
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
    banner = ("SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT"
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
    cohort = ctx.cohort()
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

    banner = ("SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT"
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
    _, _, test_ds = ctx.build_datasets(split, model.spec.use_cnn)
    trainer = Trainer(model, ctx.cfg)
    evaluation = trainer.evaluate(make_loader(test_ds, ctx.cfg.train.batch_size))

    banner = ("SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT"
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
    pipeline = load_inference_pipeline(
        ctx.outputs, ctx.checkpoint_path(variant), ctx.cfg,
        scaler_path=ctx.scaler_path, split_path=ctx.split_path,
        smoke_marker=ctx.smoke_marker,
    )
    cohort = ctx.cohort()
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
    pipeline = load_inference_pipeline(
        ctx.outputs, ctx.checkpoint_path(variant), ctx.cfg,
        scaler_path=ctx.scaler_path, split_path=ctx.split_path,
        smoke_marker=ctx.smoke_marker,
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

    cohort = ctx.cohort()
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
    parser.add_argument("--experiment", type=str,
                        help="Experiment ID for the config and checkpoint stamp.")
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
