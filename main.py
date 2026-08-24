"""
NeuroGenesis Phase 1 — legacy imaging + visualization orchestrator
=================================================================

.. note::

   **`run.py` is the active orchestrator.** This script is retained because it
   drives the preserved Phase-1 imaging and visualization path end to end,
   including the publication figures from ``visualization/visualize.py`` that
   ``run.py --mode preprocess`` does not yet reproduce.

   What it does *not* do any more: the former "Stage 9.5" invoked the
   pre-refactor NeuroProp-X engine, the closed-form atrophy extrapolation, the
   Digital Twin and the synthetic-SHAP XAI module. Those consumed hard-coded
   speech-assessment scores that OASIS-1 does not contain and produced numbers
   with no learnable parameters behind them, so the stage has been removed
   rather than repointed. The active, trainable equivalents are:

   ===========================  ==================================================
   NeuroProp-X                  ``modules/m06_neuropropx``
   Graph classification         ``modules/m06_graph_learning``
   Stage propensity             ``modules/m07_stage_tgt``
   Explainable AI               ``modules/m08_xai``
   Reports                      ``modules/m11_report``
   ===========================  ==================================================

   Equivalent modern invocations::

       python run.py --mode preprocess      # M1-M8
       python run.py --mode train_full      # model training
       python run.py --mode report          # subject reports

   See ``MIGRATION_PLAN.md`` for the full disposition of every legacy file.

Original description
--------------------
Author : NeuroGenesis Research Team
Version: 1.0.0 (Phase 1 — Preprocessing)
Date   : 2024

Description:
    Executes the complete NeuroGenesis Phase 1 preprocessing pipeline
    for all subjects found in the dataset directory.

    Pipeline Stages:
        0. Environment setup (logging, output directories)
        1. MRI data loading (nibabel, metadata extraction)
        2. Quality control   (artifact detection, QC scoring)
        3. Preprocessing     (N4 bias → WM-norm → CLAHE → Aniso → Min-max)
        4. Skull stripping   (Nilearn → SimpleITK fallback)
        5. Spatial resize    (SimpleITK, 128×128×128 isotropic)
        6. ROI extraction    (Harvard-Oxford atlas, 5 speech regions)
        7. ROI patch crops   (Novel: 48×48×48 per ROI → (5,48,48,48) tensor)
        8. Feature extraction(Volume, GM, intensity, surface area, …)
        9. Graph construction(NetworkX speech connectivity graph)
       10. Visualisation     (Pipeline summary, ROI panel, feature table)
       11. Save all outputs  (NIfTI, CSV, JSON, NPY, PNG, adjacency matrix)

Configuration:
    Modify the CONFIG dictionary below. No hardcoded paths anywhere else.

Usage:
    python main.py

Outputs:
    outputs/
    ├── original/       — tri-plane MRI images + slice galleries
    ├── processed/      — preprocessing stage images + NIfTI files
    ├── segmented/      — ROI masks, overlays, statistics
    ├── roi_patches/    — per-ROI 3D patch .nii.gz files + mosaic
    ├── tensors/        — (5,48,48,48) .npy tensors (NeuroProp-X inputs)
    ├── graphs/         — connectivity graph PNG + adjacency matrix CSV
    ├── features/       — CSV, JSON feature tables per subject
    └── neurogenesis.log
"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# ████  CONFIGURATION  ████
# All paths and parameters are set here — no hardcoding elsewhere.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "dataset_dir":      Path("dataset/OASIS"),   # legacy / synthetic fallback
    "output_dir":       Path("outputs"),

    # ── Dataset Mode ───────────────────────────────────────────────────────
    # 'oasis1'    → use real OASIS-1 MRI + CSV via dataset/oasis1_loader.py
    # 'synthetic' → use existing dataset/OASIS/ directory (testing only)
    "dataset_mode":     "oasis1",
    "oasis_mri_dir":    Path("dataset/OASIS"),     # root folder with real MRI files
    "oasis_csv_path":   None,            # None = auto-discover first .csv in dataset/

    # ── Preprocessing ──────────────────────────────────────────────────────
    "target_shape":     (128, 128, 128),   # Resize target
    "interpolator":     "linear",          # 'linear' | 'bspline' | 'nearest'
    "gaussian_sigma":   1.0,               # Gaussian smoothing σ
    "aniso_iterations": 10,               # Anisotropic diffusion iterations
    "aniso_conductance":3.0,              # Anisotropic diffusion conductance
    "clahe_clip_limit": 0.03,             # CLAHE clip limit
    "morph_radius":     4,                # Skull strip morphological radius

    # ── ROI Patches (Novel ROI-Centric Pipeline) ───────────────────────────
    "patch_size":       (48, 48, 48),     # Per-ROI patch size
    "context_pad":      4,                # Voxel padding around ROI bbox

    # ── Quality Control ────────────────────────────────────────────────────
    "min_quality_score":  50.0,           # QC threshold (0–100)
    "skip_failed_qc":     False,          # Skip scans below QC threshold?

    # ── Feature Extraction ─────────────────────────────────────────────────
    "gm_threshold":     0.30,            # Grey matter intensity fraction

    # ── Atlas ──────────────────────────────────────────────────────────────
    "atlas_name":       "cort-maxprob-thr25-2mm",
    "atlas_dir":        Path("dataset/nilearn_data"),

    # ── Pipeline Flags ─────────────────────────────────────────────────────
    "save_nifti":       True,            # Save intermediate NIfTI files
    "save_figures":     True,            # Save all PNG figures
    "max_subjects":     10,              # Process 10 subjects (31 available)

}


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure root logger to write to both file and console.

    Args:
        output_dir : Directory where 'neurogenesis.log' is saved.

    Returns:
        Configured root logger.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "neurogenesis.log"

    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=date_fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("NeuroGenesis")
    logger.info("=" * 80)
    logger.info("  NeuroGenesis - legacy Phase-1 imaging pipeline")
    logger.info(f"  Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Log file   : {log_path}")
    logger.info("=" * 80)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Output directory factory
# ──────────────────────────────────────────────────────────────────────────────

def create_output_dirs(base: Path) -> Dict[str, Path]:
    """
    Create all output sub-directories and return a path map.

    Args:
        base : Root output directory.

    Returns:
        Dict mapping logical name → Path.
    """
    dirs = {
        "original":    base / "original",
        "processed":   base / "processed",
        "segmented":   base / "segmented",
        "roi_patches": base / "roi_patches",
        "tensors":     base / "tensors",
        "graphs":      base / "graphs",
        "features":    base / "features",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# Stage functions
# ──────────────────────────────────────────────────────────────────────────────

def stage_load(cfg: Dict, dirs: Dict, logger: logging.Logger):
    """
    Stage 1: Load all MRI files from the dataset directory.

    When cfg["dataset_mode"] == "oasis1", uses OASIS1Loader to discover real
    MRI files, match them to the OASIS-1 CSV, and enrich each scan's metadata
    with clinical/demographic fields.  All downstream stages are unaffected.

    When cfg["dataset_mode"] == "synthetic" (or any other value), falls back
    to the original behaviour: scan cfg["dataset_dir"] directly.

    Returns:
        (MRILoader instance, list of file paths)
    """
    from preprocessing.loader import MRILoader

    logger.info("▶ STAGE 1 — MRI Data Loading")

    # ── OASIS-1 real dataset mode ─────────────────────────────────────────
    oasis_pairs: Optional[List] = None   # list of (Path, clinical_dict)

    if cfg.get("dataset_mode", "synthetic") == "oasis1":
        try:
            import sys as _sys
            from pathlib import Path as _Path
            # Ensure dataset/ package is importable
            _dataset_dir = _Path("dataset")
            if str(_dataset_dir) not in _sys.path:
                _sys.path.insert(0, str(_dataset_dir))

            from dataset.oasis1_loader import OASIS1Loader

            oasis_loader = OASIS1Loader(
                mri_root    = cfg["oasis_mri_dir"],
                csv_path    = cfg.get("oasis_csv_path"),   # None = auto-discover
                dataset_dir = Path("dataset"),
                max_subjects= cfg.get("max_subjects"),
            )
            oasis_pairs = oasis_loader.load()

            if not oasis_pairs:
                logger.warning(
                    "  OASIS-1 loader returned no subjects. "
                    f"Check that MRI files exist under: {cfg['oasis_mri_dir']}"
                )
        except Exception as exc:
            logger.error(f"  OASIS1Loader failed: {exc} — falling back to dataset_dir scan.")
            oasis_pairs = None

    # ── Initialise MRILoader (unchanged for all modes) ────────────────────
    # Use oasis_mri_dir as dataset_dir for OASIS-1, otherwise legacy path
    effective_dir = (
        cfg["oasis_mri_dir"]
        if (cfg.get("dataset_mode") == "oasis1" and oasis_pairs is not None)
        else cfg["dataset_dir"]
    )

    loader = MRILoader(
        dataset_dir=effective_dir,
        output_dir=dirs["original"],
    )

    # ── Determine file list ───────────────────────────────────────────────
    if oasis_pairs is not None:
        # OASIS-1 mode: file list comes from oasis1_loader
        files = [fp for fp, _ in oasis_pairs]
    else:
        # Synthetic / legacy mode: scan the directory
        files = loader.scan_dataset()
        if cfg["max_subjects"] is not None:
            files = files[: cfg["max_subjects"]]
            logger.info(
                f"  Limited to {len(files)} subject(s) "
                f"(max_subjects={cfg['max_subjects']})"
            )

    if not files:
        logger.warning(
            "  No MRI files found. "
            f"Place .nii/.nii.gz files in {effective_dir} and re-run."
        )
        return loader, []

    logger.info(f"  Loading {len(files)} subject(s)…")
    for filepath in tqdm(files, desc="Loading MRI", unit="scan",
                         bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            loader.load_single(filepath)
        except Exception as exc:
            logger.warning(f"  Could not load {filepath.name}: {exc}")

    # ── Enrich scan metadata with OASIS-1 clinical fields ─────────────────
    # Merge clinical dict into each scan's existing metadata so downstream
    # stages automatically receive age, CDR, MMSE, etc.  No stage changes.
    if oasis_pairs is not None:
        clinical_map = {}  # patient_id -> clinical dict
        for fp, clinical in oasis_pairs:
            pid = loader._extract_patient_id(fp)
            clinical_map[pid] = clinical

        for pid, entry in loader.loaded_scans.items():
            clin = clinical_map.get(pid, {"metadata_matched": False})
            entry["metadata"].update(clin)
            if clin.get("metadata_matched"):
                logger.info(
                    f"  [OASIS1] {pid} │ age={clin.get('age')} │ "
                    f"CDR={clin.get('cdr')} │ MMSE={clin.get('mmse')}"
                )
            else:
                logger.warning(
                    f"  [OASIS1] {pid} │ no CSV match — clinical metadata unavailable"
                )

    loader.print_metadata_table()

    if cfg["save_figures"]:
        logger.info("  Generating tri-plane visualisations…")
        for pid in tqdm(loader.loaded_scans, desc="Tri-planes", unit="scan",
                        bar_format="{l_bar}{bar:30}{r_bar}"):
            loader.visualize_triplane(pid, save=True)
            loader.visualize_slice_gallery(pid, axis=2, save=True)

    # Save metadata (now includes clinical fields for OASIS-1 subjects)
    loader.save_metadata_json(dirs["features"] / "metadata.json")

    return loader, files


def stage_qc(cfg: Dict, dirs: Dict, loader, logger: logging.Logger) -> Dict[str, Any]:
    """
    Stage 2: Quality control on all loaded scans.

    Returns:
        Dict mapping patient_id → QCReport
    """
    from preprocessing.artifact_detector import ArtifactDetector

    logger.info("▶ STAGE 2 — MRI Quality Control")
    detector = ArtifactDetector(
        output_dir=dirs["processed"],
        min_quality_score=cfg["min_quality_score"],
    )

    qc_reports = {}
    for pid in tqdm(loader.loaded_scans, desc="QC Check", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        data = loader.loaded_scans[pid]["data"]
        try:
            report = detector.run_qc(data, pid, save_figure=cfg["save_figures"])
            qc_reports[pid] = report
            logger.info(f"  {report.summary()}")
        except Exception as exc:
            logger.warning(f"  QC failed for {pid}: {exc}")

    passed = sum(1 for r in qc_reports.values() if r.passed)
    logger.info(f"  QC complete │ {passed}/{len(qc_reports)} scans passed")
    return qc_reports


def stage_skull_strip(
    cfg: Dict, dirs: Dict, loader, qc_reports: Dict, logger: logging.Logger
) -> Dict[str, tuple]:
    """
    Stage 3: Skull stripping on RAW volumes (before normalisation).

    Skull stripping must run on raw intensities so that Otsu thresholding
    can correctly distinguish brain tissue from skull/background.

    Returns:
        Dict mapping patient_id → (masked_data, brain_mask)
    """
    from preprocessing.skull_strip import SkullStripper

    logger.info("▶ STAGE 3 — Skull Stripping (on raw intensities)")
    stripper = SkullStripper(
        output_dir=dirs["processed"],
        morph_radius=cfg["morph_radius"],
    )

    pids_to_process = [
        pid for pid in loader.loaded_scans
        if not cfg["skip_failed_qc"] or qc_reports.get(pid, type("R", (), {"passed": True})).passed
    ]

    stripped = {}
    for pid in tqdm(pids_to_process, desc="Skull Strip", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            raw_data = loader.loaded_scans[pid]["data"]
            affine   = loader.loaded_scans[pid]["affine"]
            masked, mask = stripper.strip(raw_data, pid, affine,
                                          save=cfg["save_figures"])
            stripped[pid] = (masked, mask)
        except Exception as exc:
            logger.error(f"  Skull strip failed for {pid}: {exc}")
            stripped[pid] = (loader.loaded_scans[pid]["data"], None)

    return stripped


def stage_preprocess(
    cfg: Dict, dirs: Dict, loader, stripped: Dict, logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Stage 4: Full normalisation pipeline on skull-stripped volumes.

    Normalisation runs AFTER skull stripping so that intensity statistics
    (white-matter peak, histogram) are computed on brain tissue only.

    Returns:
        Dict mapping patient_id → normalisation results dict
    """
    from preprocessing.normalization import MRINormalizer

    logger.info("▶ STAGE 4 — Preprocessing (N4 + WM-Norm + CLAHE + Aniso + MinMax)")
    normalizer = MRINormalizer(
        output_dir=dirs["processed"],
        gaussian_sigma=cfg["gaussian_sigma"],
        aniso_iterations=cfg["aniso_iterations"],
        aniso_conductance=cfg["aniso_conductance"],
        clahe_clip_limit=cfg["clahe_clip_limit"],
    )

    norm_results = {}
    for pid in tqdm(stripped, desc="Preprocessing", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            masked_data = stripped[pid][0]
            affine      = loader.loaded_scans[pid]["affine"]
            result = normalizer.run_full_pipeline(
                masked_data, pid, affine, save_nifti=cfg["save_nifti"]
            )
            norm_results[pid] = result
        except Exception as exc:
            logger.error(f"  Preprocessing failed for {pid}: {exc}")
            logger.debug(traceback.format_exc())

    return norm_results


def stage_resize(
    cfg: Dict, dirs: Dict, loader, norm_results: Dict, logger: logging.Logger
) -> Dict[str, tuple]:
    """
    Stage 5: Resample all volumes to the common target shape.

    Returns:
        Dict mapping patient_id → (resampled_data, new_affine)
    """
    from preprocessing.resize import MRIResizer

    logger.info(f"▶ STAGE 5 — Spatial Resampling → {cfg['target_shape']}")
    resizer = MRIResizer(
        target_shape=cfg["target_shape"],
        interpolator=cfg["interpolator"],
        output_dir=dirs["processed"],
    )

    resized = {}
    for pid in tqdm(norm_results, desc="Resampling", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            final_data = norm_results[pid]["final"]
            affine     = loader.loaded_scans[pid]["affine"]
            r_data, r_affine = resizer.resample(
                final_data, affine, pid, save=cfg["save_nifti"]
            )
            resized[pid] = (r_data, r_affine)
        except Exception as exc:
            logger.error(f"  Resampling failed for {pid}: {exc}")
            resized[pid] = (norm_results[pid]["final"], loader.loaded_scans[pid]["affine"])

    return resized


def stage_roi_extraction(
    cfg: Dict, dirs: Dict, loader, resized: Dict, logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Stage 6: Extract speech ROI masks from the resampled volumes.

    Returns:
        Dict mapping patient_id → {roi_name: binary_mask}
    """
    from segmentation.roi_extraction import ROIExtractor

    logger.info("▶ STAGE 6 — Speech ROI Extraction (Harvard-Oxford Atlas)")
    extractor = ROIExtractor(
        output_dir=dirs["segmented"],
        atlas_name=cfg["atlas_name"],
        data_dir=cfg.get("atlas_dir"),
    )

    # Load atlas once
    try:
        extractor.load_atlas()
    except Exception as exc:
        logger.error(f"  Atlas loading failed: {exc}")
        logger.warning(
            "  ROI extraction will be skipped. Ensure you have internet "
            "access for first-time atlas download."
        )
        return {pid: {} for pid in resized}

    all_masks = {}
    for pid in tqdm(resized, desc="ROI Extraction", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            r_data, r_affine = resized[pid]
            masks = extractor.extract_all(r_data, r_affine, pid,
                                          save=cfg["save_figures"])
            all_masks[pid] = masks
        except Exception as exc:
            logger.error(f"  ROI extraction failed for {pid}: {exc}")
            all_masks[pid] = {}

    return all_masks


def stage_roi_crops(
    cfg: Dict, dirs: Dict, loader, resized: Dict, all_masks: Dict, logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Stage 7 (Novel): Extract ROI-centric 3D patches and NeuroProp-X tensors.

    Returns:
        Dict mapping patient_id → {roi_name: patch_array}
    """
    from preprocessing.roi_crop import ROICropper

    logger.info(
        f"▶ STAGE 7 — ROI-Centric Patch Extraction (Novel) "
        f"│ patch_size={cfg['patch_size']}"
    )
    cropper = ROICropper(
        patch_size=cfg["patch_size"],
        context_pad=cfg["context_pad"],
        output_dir=dirs["roi_patches"],
        tensor_dir=dirs["tensors"],
    )

    all_patches = {}
    for pid in tqdm(resized, desc="ROI Crops", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        masks = all_masks.get(pid, {})
        if not masks:
            logger.warning(f"  No ROI masks for {pid} — skipping patch extraction.")
            all_patches[pid] = {}
            continue
        try:
            r_data, r_affine = resized[pid]
            tensor = cropper.extract_all(
                r_data, masks, pid, r_affine, save=cfg["save_figures"]
            )
            # Also store per-ROI patches dict
            patches_dict = {}
            for i, roi_name in enumerate(masks.keys()):
                patches_dict[roi_name] = tensor[i] if i < tensor.shape[0] else \
                    np.zeros(cfg["patch_size"], dtype=np.float32)
            all_patches[pid] = patches_dict
            logger.info(
                f"  ✓ {pid} │ tensor shape={tensor.shape} "
                f"│ saved to {dirs['tensors']}"
            )
        except Exception as exc:
            logger.error(f"  Patch extraction failed for {pid}: {exc}")
            all_patches[pid] = {}

    return all_patches


def stage_feature_extraction(
    cfg: Dict, dirs: Dict, loader, resized: Dict,
    all_patches: Dict, logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    """
    Stage 8: Compute features from ROI patches.

    Returns:
        Dict mapping patient_id → feature DataFrame
    """
    from features.feature_extractor import FeatureExtractor

    logger.info("▶ STAGE 8 — Feature Extraction")

    all_dfs: Dict[str, pd.DataFrame] = {}
    all_records: List[pd.DataFrame] = []

    for pid in tqdm(all_patches, desc="Features", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        patches = all_patches.get(pid, {})
        if not patches:
            logger.warning(f"  No patches for {pid} — skipping features.")
            continue

        try:
            # Derive voxel volume from resampled affine
            _, r_affine = resized.get(pid, (None, np.eye(4)))
            voxel_vol = float(abs(np.linalg.det(r_affine[:3, :3])))

            extractor = FeatureExtractor(
                output_dir=dirs["features"],
                voxel_volume_mm3=voxel_vol,
                gm_threshold=cfg["gm_threshold"],
            )

            df = extractor.extract_all(patches, pid)
            paths = extractor.save(df, pid)
            extractor.plot_feature_heatmap(df, pid)

            all_dfs[pid] = df
            all_records.append(df)
            logger.info(
                f"  ✓ {pid} │ {len(df)} ROIs │ CSV={paths.get('csv','')}"
            )

        except Exception as exc:
            logger.error(f"  Feature extraction failed for {pid}: {exc}")

    # Aggregate multi-subject feature table
    if all_records:
        from features.feature_extractor import FeatureExtractor
        agg_df = pd.concat(all_records, ignore_index=True)
        ext_tmp = FeatureExtractor(output_dir=dirs["features"])
        table_path = ext_tmp.save_feature_table(agg_df)
        logger.info(f"  Aggregate feature table saved → {table_path}")

    return all_dfs


def stage_graph(
    cfg: Dict, dirs: Dict, all_dfs: Dict, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage 9: Build and visualise speech connectivity graphs.

    Returns:
        Dict mapping patient_id → nx.DiGraph
    """
    from graph.graph_builder import BrainConnectivityGraph

    logger.info("▶ STAGE 9 — Brain Connectivity Graph Construction")
    builder = BrainConnectivityGraph(output_dir=dirs["graphs"])
    graphs = {}

    pids = list(all_dfs.keys()) or ["template"]
    for pid in tqdm(pids, desc="Graph Build", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            # Convert feature DataFrame to roi_features dict for node attrs
            roi_features = {}
            if pid in all_dfs:
                df = all_dfs[pid]
                if "roi_name" in df.columns:
                    for _, row in df.iterrows():
                        roi_features[row["roi_name"]] = row.to_dict()

            G = builder.build(pid, roi_features=roi_features)
            builder.visualize(pid, save=cfg["save_figures"])
            builder.save_adjacency_matrix(pid)
            builder.save_graph_json(pid)
            graphs[pid] = G

        except Exception as exc:
            logger.error(f"  Graph build failed for {pid}: {exc}")

    # Always build a template graph (even if no subjects loaded)
    if "template" not in graphs:
        try:
            G_template = builder.build("template")
            builder.visualize("template", save=cfg["save_figures"])
            builder.save_adjacency_matrix("template")
            graphs["template"] = G_template
        except Exception as exc:
            logger.warning(f"  Template graph failed: {exc}")

    return graphs


def stage_visualize(
    cfg: Dict, dirs: Dict,
    loader, resized: Dict, stripped: Dict,
    all_masks: Dict, all_patches: Dict,
    all_dfs: Dict, graphs: Dict,
    logger: logging.Logger,
    qc_reports: Optional[Dict] = None,
) -> None:
    """
    Stage 10: Generate comprehensive per-subject outputs and visualizations.
    """
    from visualization.visualize import NeuroGenesisVisualizer
    import pickle

    logger.info("▶ STAGE 10 — Final Visualisation & Subject Report Generation")
    viz = NeuroGenesisVisualizer(output_dir=dirs["original"].parent)

    for pid in tqdm(resized, desc="Visualising", unit="scan",
                    bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            # Create dedicated subject directory: outputs/<Subject_ID>/
            subj_dir = cfg["output_dir"] / pid
            subj_dir.mkdir(parents=True, exist_ok=True)

            original     = loader.loaded_scans[pid]["data"]
            preprocessed = resized[pid][0]
            masked       = stripped.get(pid, (None, None))[0]
            masks        = all_masks.get(pid, {})
            feature_df   = all_dfs.get(pid)
            graph        = graphs.get(pid) or graphs.get("template")
            patches      = all_patches.get(pid, {})
            qc_rep       = qc_reports.get(pid) if qc_reports else None

            # 1. Master pipeline summary figure
            viz.generate_pipeline_summary(
                patient_id=pid,
                original=original,
                preprocessed=preprocessed,
                masked=masked,
                masks=masks,
                feature_df=feature_df,
                graph=graph,
                patches=patches,
            )

            # Copy/save visualization into subject directory
            viz_src = cfg["output_dir"] / f"{pid}_pipeline_summary.png"
            if viz_src.exists():
                import shutil
                shutil.copy(viz_src, subj_dir / "visualization.png")

            # 2. Individual speech ROI overlays (Step 7)
            if masks:
                viz.plot_individual_roi_overlays(preprocessed, masks, pid, subj_dir=subj_dir)
                # 3. Combined speech network overlay (Step 8)
                viz.plot_combined_speech_network_overlay(preprocessed, masks, pid, subj_dir=subj_dir)
                # 4. Multi-view speech network (Step 9)
                viz.plot_speech_network_multiview(preprocessed, masks, pid, subj_dir=subj_dir)
                viz.plot_roi_panel(preprocessed, masks, pid)

            # 5. Feature table figure & CSV
            if feature_df is not None and not feature_df.empty:
                viz.plot_feature_table_figure(feature_df, pid)
                feature_df.to_csv(subj_dir / "features.csv", index=False)

            # 6. Save graph pickle & json in subject directory
            if graph is not None:
                with open(subj_dir / "graph.pkl", "wb") as fh:
                    pickle.dump(graph, fh)

            # 7. Save QC report JSON in subject directory
            if qc_rep is not None:
                with open(subj_dir / "qc.json", "w", encoding="utf-8") as fh:
                    json.dump(qc_rep.__dict__, fh, indent=2, default=str)

            # 8. Save per-subject summary JSON
            subj_meta = loader.loaded_scans[pid]["metadata"]
            subj_summary = {
                "subject_id": pid,
                "timestamp": datetime.now().isoformat(),
                "clinical_metadata": {
                    "age": subj_meta.get("age"),
                    "gender": subj_meta.get("gender"),
                    "cdr": subj_meta.get("cdr"),
                    "mmse": subj_meta.get("mmse"),
                    "education": subj_meta.get("education"),
                    "ses": subj_meta.get("ses"),
                },
                "mri_metadata": {
                    "shape": subj_meta.get("shape"),
                    "voxel_dims_mm": subj_meta.get("voxel_dims_mm"),
                    "intensity_range": [subj_meta.get("min_intensity"), subj_meta.get("max_intensity")],
                },
                "qc_passed": qc_rep.passed if qc_rep else True,
                "qc_score": qc_rep.quality_score if qc_rep else None,
                "roi_voxel_counts": {r: int(m.sum()) for r, m in masks.items()} if masks else {},
            }
            with open(subj_dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(subj_summary, fh, indent=2, default=str)

            # 9. Step 10: Generate debug_report.txt
            write_debug_report(
                pid=pid,
                loader=loader,
                resized=resized,
                masks=masks,
                patches=patches,
                df=feature_df,
                graph=graph,
                qc_report=qc_rep,
                subj_dir=subj_dir,
            )

        except Exception as exc:
            logger.warning(f"  Visualisation failed for {pid}: {exc}")
            logger.debug(traceback.format_exc())


def write_debug_report(
    pid: str,
    loader,
    resized: Dict,
    masks: Dict,
    patches: Dict,
    df: Optional[pd.DataFrame],
    graph: Optional[Any],
    qc_report: Optional[Any],
    subj_dir: Path,
) -> Path:
    """
    Step 10: Write comprehensive outputs/<Subject_ID>/debug_report.txt
    """
    lines = [
        "============================================================",
        f" NeuroGenesis Phase 1 — Subject Debug Report: {pid}",
        "============================================================",
        f"Timestamp           : {datetime.now().isoformat()}",
        f"Subject ID          : {pid}",
        "",
        "--- 1. MRI LOAD & REGISTRATION METADATA ---",
    ]
    meta = loader.loaded_scans.get(pid, {}).get("metadata", {})
    lines.append(f"MRI Filepath        : {meta.get('filepath')}")
    lines.append(f"MRI Dimensions      : {meta.get('shape')}")
    lines.append(f"Voxel Dimensions    : {meta.get('voxel_dims_mm')} mm")
    lines.append(f"Intensity Range     : [{meta.get('min_intensity')}, {meta.get('max_intensity')}]")
    lines.append(f"Clinical Age        : {meta.get('age')}")
    lines.append(f"Clinical Gender     : {meta.get('gender')}")
    lines.append(f"Clinical CDR        : {meta.get('cdr')}")
    lines.append(f"Clinical MMSE       : {meta.get('mmse')}")

    r_data, r_affine = resized.get(pid, (None, np.eye(4)))
    lines.append(f"Resampled Shape     : {r_data.shape if r_data is not None else 'None'}")
    lines.append(f"Resampled Affine    :\n{r_affine}")
    lines.append(f"Registration Status : Harvard-Oxford Atlas resampled to subject MRI space successfully.")

    lines.append("\n--- 2. QUALITY CONTROL REPORT ---")
    if qc_report:
        lines.append(f"QC Score            : {qc_report.quality_score:.2f} / 100")
        lines.append(f"QC Passed           : {qc_report.passed}")
        lines.append(f"SNR (dB)            : {qc_report.snr_db:.2f}")
        lines.append(f"Flags               : {qc_report.flags}")
    else:
        lines.append("QC Report unavailable.")

    lines.append("\n--- 3. ROI EXTRACTION & VOXEL COUNTS ---")
    if masks:
        for roi_name, mask in masks.items():
            n_vox = int(mask.sum())
            if n_vox > 0:
                pos = np.argwhere(mask > 0)
                bbox = (pos.min(axis=0).tolist(), pos.max(axis=0).tolist())
            else:
                bbox = "N/A (0 voxels)"
            lines.append(f"  {roi_name:28s} │ Voxels={n_vox:6d} │ BoundingBox={bbox}")
    else:
        lines.append("No ROI masks available.")

    lines.append("\n--- 4. ROI TENSOR STATISTICS ---")
    if patches:
        for roi_name, patch in patches.items():
            n_vox = int(np.count_nonzero(patch))
            min_i = float(patch.min()) if patch.size > 0 else 0.0
            max_i = float(patch.max()) if patch.size > 0 else 0.0
            mean_i = float(patch[patch > 0].mean()) if n_vox > 0 else 0.0
            std_i = float(patch[patch > 0].std()) if n_vox > 0 else 0.0
            lines.append(
                f"  {roi_name:28s} │ Shape={patch.shape} │ Min={min_i:.2f} │ "
                f"Max={max_i:.2f} │ Mean={mean_i:.2f} │ Std={std_i:.2f} │ NonZeroVoxels={n_vox}"
            )
    else:
        lines.append("No ROI tensor patches available.")

    lines.append("\n--- 5. FEATURE EXTRACTION METRICS ---")
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            lines.append(
                f"  {row['roi_name']:28s} │ Voxels={row.get('voxel_count', 0)} │ "
                f"BrainVol={row.get('brain_volume_mm3', 0.0):.1f} mm3 │ "
                f"GMVol={row.get('gm_volume_mm3', 0.0):.1f} mm3 │ "
                f"MeanInt={row.get('mean_intensity', 0.0):.2f} │ "
                f"StdInt={row.get('std_intensity', 0.0):.2f} │ "
                f"SurfaceArea={row.get('surface_area_vox', 0.0):.1f} vox2"
            )
    else:
        lines.append("Feature extraction failed or DataFrame empty.")

    lines.append("\n--- 6. GRAPH NODE ATTRIBUTES ---")
    if graph is not None:
        lines.append(f"Nodes: {list(graph.nodes())}")
        lines.append(f"Edges: {len(graph.edges())}")
        for n, data in graph.nodes(data=True):
            lines.append(f"  Node [{n}]: {data}")
    else:
        lines.append("Graph unavailable.")

    lines.append("\n============================================================")
    report_text = "\n".join(lines)
    report_file = subj_dir / "debug_report.txt"
    with open(report_file, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    return report_file


# ──────────────────────────────────────────────────────────────────────────────
# `stage_neuropropx` removed in the stage-aware refactor.
#
# It called the pre-refactor NeuroProp-X engine, the closed-form atrophy
# extrapolation mislabelled a "Temporal Graph Transformer", the Digital Twin and
# the XAI module whose own comment read "Synthetic SHAP value generation". Its
# inputs included `SpeechAssessmentScores(88.0, 86.0, 84.0, 87.0, 86.25)` -
# fabricated speech scores that OASIS-1 does not measure.
#
# The trainable replacements live in `modules/m06_neuropropx`,
# `modules/m06_graph_learning`, `modules/m07_stage_tgt`, `modules/m08_xai` and
# `modules/m11_report`, and are driven by `run.py`. The original code is
# preserved in git history and archived under
# `modules/m12_future_extensions/_legacy_backend/`.
# ──────────────────────────────────────────────────────────────────────────────


def print_pipeline_summary(
    cfg: Dict,
    dirs: Dict,
    loader,
    qc_reports: Dict,
    all_dfs: Dict,
    graphs: Dict,
    elapsed: float,
    logger: logging.Logger,
) -> None:
    """Print and log a final pipeline execution summary."""
    n_subjects = len(loader.loaded_scans)
    n_passed   = sum(1 for r in qc_reports.values() if r.passed)
    n_features = len(all_dfs)
    n_graphs   = len([g for k, g in graphs.items() if k != "template"])

    summary = {
        "pipeline":       "NeuroGenesis Phase 1",
        "timestamp":      datetime.now().isoformat(),
        "elapsed_s":      round(elapsed, 2),
        "subjects_loaded":n_subjects,
        "subjects_qc_passed": n_passed,
        "feature_tables": n_features,
        "graphs_built":   n_graphs,
        "outputs": {
            name: str(path) for name, path in dirs.items()
        },
        "config": {
            k: str(v) for k, v in cfg.items()
        },
    }

    # Save summary JSON
    summary_path = cfg["output_dir"] / "pipeline_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Print summary table
    sep = "═" * 70
    lines = [
        "",
        sep,
        "  NeuroGenesis Phase 1 — Pipeline Complete",
        sep,
        f"  Subjects loaded      : {n_subjects}",
        f"  QC passed            : {n_passed} / {n_subjects}",
        f"  Feature tables       : {n_features}",
        f"  Graphs built         : {n_graphs}",
        f"  Elapsed time         : {elapsed:.1f}s ({elapsed/60:.1f} min)",
        f"  Output directory     : {cfg['output_dir'].resolve()}",
        f"  Pipeline summary     : {summary_path}",
        sep,
        "  Next Step: Phase 2 — Integrate NeuroProp-X",
        f"  Input tensors ready in: {dirs['tensors']}",
        sep,
        "",
    ]

    for line in lines:
        print(line)
        logger.info(line)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the complete NeuroGenesis Phase 1 pipeline."""
    start_time = time.time()

    # ── Setup ───────────────────────────────────────────────────────────────
    cfg = CONFIG
    dirs = create_output_dirs(cfg["output_dir"])
    logger = setup_logging(cfg["output_dir"])

    logger.info(f"Dataset directory : {cfg['dataset_dir'].resolve()}")
    logger.info(f"Target shape      : {cfg['target_shape']}")
    logger.info(f"Patch size        : {cfg['patch_size']}")
    logger.info(f"Skip failed QC    : {cfg['skip_failed_qc']}")

    # ── Stages ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  NeuroGenesis - legacy Phase-1 imaging pipeline")
    print("  (run.py is the active orchestrator; see the module docstring)")
    print("═" * 70 + "\n")

    stages_progress = tqdm(
        total=10,
        desc="Pipeline",
        unit="stage",
        bar_format="[Overall] {l_bar}{bar:40}{r_bar}",
        position=0,
    )

    try:
        # Stage 1: Load
        loader, files = stage_load(cfg, dirs, logger)
        stages_progress.update(1)

        if not loader.loaded_scans:
            logger.warning(
                "\n  No MRI data loaded. Pipeline will complete with demo graph only.\n"
                f"  → Place .nii/.nii.gz files in: {cfg['dataset_dir'].resolve()}\n"
            )

        # Stage 2: QC
        qc_reports = stage_qc(cfg, dirs, loader, logger) if loader.loaded_scans else {}
        stages_progress.update(1)

        # Stage 3: Skull strip (on raw data — before normalisation)
        stripped = stage_skull_strip(cfg, dirs, loader, qc_reports, logger) \
            if loader.loaded_scans else {}
        stages_progress.update(1)

        # Stage 4: Preprocess (normalise the skull-stripped data)
        norm_results = stage_preprocess(cfg, dirs, loader, stripped, logger) \
            if stripped else {}
        stages_progress.update(1)

        # Stage 5: Resize
        resized = stage_resize(cfg, dirs, loader, norm_results, logger) \
            if norm_results else {}
        stages_progress.update(1)

        # Stage 6: ROI extraction
        all_masks = stage_roi_extraction(cfg, dirs, loader, resized, logger) \
            if resized else {}
        stages_progress.update(1)

        # Stage 7: ROI crops (Novel)
        all_patches = stage_roi_crops(cfg, dirs, loader, resized, all_masks, logger) \
            if all_masks else {}
        stages_progress.update(1)

        # Stage 8: Feature extraction
        all_dfs = stage_feature_extraction(cfg, dirs, loader, resized, all_patches, logger) \
            if all_patches else {}
        stages_progress.update(1)

        # Stage 9: Graph construction (runs even without subject data)
        graphs = stage_graph(cfg, dirs, all_dfs, logger)
        stages_progress.update(1)

        # Stage 9.5 removed. It invoked the pre-refactor NeuroProp-X engine,
        # the legacy atrophy extrapolation, the Digital Twin and the
        # synthetic-SHAP XAI module on hard-coded speech-assessment scores that
        # OASIS-1 does not contain. Use `python run.py --mode train_full` and
        # `--mode report` for the trainable replacements.

        # Stage 10: Final visualisation
        if loader.loaded_scans:
            stage_visualize(
                cfg, dirs, loader, resized, stripped,
                all_masks, all_patches, all_dfs, graphs, logger,
                qc_reports=qc_reports
            )
        stages_progress.update(1)


    except KeyboardInterrupt:
        logger.warning("\n  Pipeline interrupted by user.")
        stages_progress.close()
        sys.exit(0)
    except Exception as exc:
        logger.error(f"\n  Pipeline encountered a fatal error: {exc}")
        logger.error(traceback.format_exc())
        stages_progress.close()
        sys.exit(1)

    stages_progress.close()

    elapsed = time.time() - start_time
    print_pipeline_summary(
        cfg, dirs, loader, qc_reports, all_dfs, graphs, elapsed, logger
    )


if __name__ == "__main__":
    main()
