"""
NeuroGenesis — research pipeline inspection dashboard (Sections 39-61).
=====================================================================

This is **not** a final-results dashboard. It is an interactive inspection system
for the whole pipeline: every major module exposes its input, processing status,
output, numerical values, visualisation, downloadable artifact and an explanation
of what it did.

Three rules the app enforces:

**Status is measured, never asserted.** Every module badge comes from
:class:`~modules.common.run_state.RunStateTracker`, which records only what
actually executed. There is no hard-coded "completed" anywhere in this file.

**A missing artifact says so, and says how to produce it.** Absent inputs render
an explicit panel naming the ``run.py`` mode that generates them. No page ever
substitutes a plausible number.

**Provenance precedes every metric.** If the outputs tree is marked synthetic,
each page shows the warning banner before any value.

Launch with::

    python run.py --mode dashboard --outputs outputs
    # or directly:
    streamlit run dashboard/app.py -- --outputs outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components import (  # noqa: E402
    array_download,
    dataframe_download,
    download_button,
    formula,
    inject_theme,
    intensity_histogram,
    interpretation_note,
    matrix_view,
    module_header,
    page_title,
    provenance_banner,
    roi_bar,
    roi_metadata_card,
    roi_selector,
    slice_views,
    stage_detail,
    stage_probability_row,
    status_pill,
    unavailable,
    what_this_module_did,
)
from dashboard.state import DashboardState  # noqa: E402
from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.roi_constants import (  # noqa: E402
    N_ROI,
    ROI_ORDER,
    STAGE_ORDER,
    roi_short,
)
from modules.common.run_state import PIPELINE, PIPELINE_BY_CODE  # noqa: E402

st.set_page_config(
    page_title="NeuroGenesis — pipeline inspection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def parse_args() -> argparse.Namespace:
    """Parse the ``--outputs`` argument passed after Streamlit's ``--``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    known, _ = parser.parse_known_args()
    return known


# ──────────────────────────────────────────────────────────────────────────────
# Page: overview and pipeline tracker (Section 40)
# ──────────────────────────────────────────────────────────────────────────────

def page_overview(state: DashboardState, subject: Optional[str]) -> None:
    """Global pipeline tracker, readiness checklist and environment report."""
    page_title(
        "Pipeline tracker",
        "Execution status of every module, derived from recorded state — never "
        "hard-coded.",
    )
    provenance_banner(state)

    if not state.exists():
        unavailable(
            f"the outputs directory {state.outputs} does not exist",
            "python run.py --mode preprocess",
        )
        return

    status = state.pipeline_status(subject)
    completed = int((status["status"] == "COMPLETED").sum())
    failed = int((status["status"] == "FAILED").sum())
    skipped = int((status["status"] == "SKIPPED").sum())

    columns = st.columns(5)
    columns[0].metric("Modules", len(status))
    columns[1].metric("Completed", completed)
    columns[2].metric("Failed", failed)
    columns[3].metric("Skipped", skipped)
    columns[4].metric("Completion", f"{state.completion(subject):.0%}")
    st.progress(state.completion(subject))

    st.markdown("### M1 → M19 execution state")
    if subject:
        st.caption(
            f"Per-subject modules reflect `{subject}`. M17 (statistics) and M18 "
            "(ablation) are cohort-level and are read from the global state."
        )

    rows = []
    for _, row in status.iterrows():
        duration = row["duration_s"]
        rows.append({
            "": row["code"],
            "Module": row["title"],
            "Scope": row["scope"],
            "Status": row["status"],
            "Duration (s)": None if duration is None else round(duration, 3),
            "Artifacts": row["n_artifacts"],
            "Error": (row["error"] or "")[:70],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 height=560)

    with st.expander("Readiness checklist — which artifacts exist"):
        checklist = pd.DataFrame(state.readiness())
        checklist["present"] = checklist["present"].map(
            {True: "present", False: "absent"}
        )
        st.dataframe(checklist, use_container_width=True, hide_index=True)

    with st.expander("Environment — optional dependencies"):
        environment = state.environment()
        if not environment["imaging_can_run"]:
            st.warning(environment["imaging_message"])
        st.markdown("**Imaging stack**")
        st.dataframe(
            pd.DataFrame(
                [{"package": k, "available": v}
                 for k, v in environment["imaging"].items()]
            ),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            f"**SHAP**: {'available' if environment['shap'] else 'not installed'}"
            " — without it, attribution falls back to exact single-feature "
            "replacement, which is reported as `permutation`, never as SHAP."
        )
        st.dataframe(
            pd.DataFrame(
                [{"baseline": k, "available": v}
                 for k, v in environment["baselines"].items()]
            ),
            use_container_width=True, hide_index=True,
        )

    what_this_module_did(
        "The tracker reads `outputs/state/pipeline_global.json` and "
        "`outputs/state/subjects/<id>.json`. Those files are written by the "
        "`RunStateTracker` context manager that wraps each pipeline module: it "
        "records `RUNNING` on entry and `COMPLETED` or `FAILED` on exit. A "
        "module that was never entered stays `NOT_STARTED`, so this table "
        "cannot show progress that did not happen."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page: patient selection
# ──────────────────────────────────────────────────────────────────────────────

def page_patient(state: DashboardState, subject: Optional[str]) -> None:
    """Cohort browser, label distribution and split membership."""
    page_title("Patient selection",
               "Cohort composition, CDR-derived labels and split membership.")
    provenance_banner(state)

    cohort = state.cohort()
    if cohort is None:
        unavailable("the cohort table", "python run.py --mode preprocess")
        return

    columns = st.columns(4)
    columns[0].metric("Labelled sessions", len(cohort))
    columns[1].metric(
        "Unique subjects",
        int(cohort["subject_id"].nunique()) if "subject_id" in cohort else 0,
    )
    with_mri = int(cohort["has_mri"].sum()) if "has_mri" in cohort else 0
    columns[2].metric("With MRI on disk", with_mri)
    columns[3].metric("Stages", len(STAGE_ORDER))

    stage_columns = st.columns(len(STAGE_ORDER))
    for column, stage in zip(stage_columns, STAGE_ORDER):
        column.metric(stage, int((cohort["stage"] == stage).sum()))

    report = state.cohort_report()
    if report:
        with st.expander("Cohort assembly report", expanded=False):
            for warning in report.get("warnings", []):
                st.warning(warning)
            label_report = report.get("label_report") or {}
            st.markdown(
                f"**Label mapping** — CDR to stage. Missing-CDR policy: "
                f"`{label_report.get('missing_cdr_policy')}`, affecting "
                f"{label_report.get('n_missing_cdr')} session(s)."
            )
            st.json({k: v for k, v in report.items()
                     if k not in ("label_report", "warnings")})

    split = state.split_manifest()
    if split:
        st.markdown("### Table 1 — dataset and split distribution")
        rows = []
        for name in ("train", "val", "test"):
            counts = split.get("session_counts", {}).get(name, {})
            row = {"Split": name.capitalize()}
            row.update({s: counts.get(s, 0) for s in STAGE_ORDER})
            row["Total"] = sum(counts.get(s, 0) for s in STAGE_ORDER)
            rows.append(row)
        total = {"Split": "Total"}
        for stage in STAGE_ORDER:
            total[stage] = sum(r[stage] for r in rows)
        total["Total"] = sum(r["Total"] for r in rows)
        rows.append(total)
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption(
            "Splits partition **subjects**, not sessions, so no subject's scans "
            f"appear in two splits. Stratum rule: {split.get('stratum_rule')}."
        )
        for warning in split.get("warnings", []):
            st.warning(warning)
    else:
        unavailable("the split manifest", "python run.py --mode train_full")

    st.markdown("### Cohort table")
    display = cohort.copy()
    if subject:
        display.insert(
            0, "selected",
            display["session_id"].astype(str) == subject,
        )
    st.dataframe(display, use_container_width=True, hide_index=True, height=380)
    dataframe_download(cohort, "cohort")

    if subject:
        row = cohort[cohort["session_id"].astype(str) == subject]
        if not row.empty:
            st.markdown(f"### Selected subject — `{subject}`")
            st.dataframe(row.T.rename(columns={row.index[0]: "value"}),
                         use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pages M1-M5: imaging (Sections 41-45)
# ──────────────────────────────────────────────────────────────────────────────

def _imaging_unavailable(state: DashboardState, subject: Optional[str],
                         code: str) -> bool:
    """Render the imaging-unavailable explanation. Returns True if unavailable."""
    if subject and state.preprocessing(subject):
        return False
    environment = state.environment()
    reason = []
    if not environment["imaging_can_run"]:
        reason.append(environment["imaging_message"])
    if state.is_synthetic:
        reason.append(
            "This outputs tree was produced by the synthetic artifact "
            "generator, which starts at the ROI patch stage and therefore "
            "produces no imaging artifacts for M1-M5."
        )
    unavailable(
        f"{code} imaging artifacts for this subject",
        "python run.py --mode preprocess",
        reason=" ".join(reason) or "The preprocessing stage has not been run.",
    )
    return True


def page_m1(state: DashboardState, subject: Optional[str]) -> None:
    """M1 — MRI acquisition and loading."""
    record = state.stage_record("M1", subject)
    module_header("M1", "MRI acquisition / loading",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M1"].description)
    provenance_banner(state)
    if _imaging_unavailable(state, subject, "M1"):
        stage_detail(record)
        return

    manifest = state.preprocessing(subject)
    metadata = manifest.get("metadata", {})
    columns = st.columns(4)
    columns[0].metric("Subject", subject)
    columns[1].metric("Shape", str(metadata.get("shape")))
    columns[2].metric("Voxels", f"{metadata.get('nonzero_voxels', 0):,}")
    columns[3].metric("Mean intensity",
                      f"{metadata.get('mean_intensity', 0):.4f}")

    st.markdown("**Acquisition metadata**")
    st.json(metadata)
    st.caption(f"Source volume: `{manifest.get('source_path')}`")
    stage_detail(record)
    what_this_module_did(
        "Loads the T1 volume with nibabel, reads its header, affine and voxel "
        "spacing, and records the dimensions and intensity range that every "
        "later stage is compared against. Output feeds M2 (quality control)."
    )


def page_m2(state: DashboardState, subject: Optional[str]) -> None:
    """M2 — quality control."""
    record = state.stage_record("M2", subject)
    module_header("M2", "Quality control",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M2"].description)
    provenance_banner(state)
    if _imaging_unavailable(state, subject, "M2"):
        stage_detail(record)
        return

    manifest = state.preprocessing(subject)
    qc = manifest.get("qc") or {}
    passed = manifest.get("qc_passed")

    columns = st.columns(4)
    columns[0].metric("QC score", f"{qc.get('quality_score', float('nan')):.2f}")
    columns[1].metric("Threshold",
                      f"{state.cfg.preprocess.min_quality_score:.1f}")
    columns[2].metric("Outcome",
                      "PASS" if passed else ("FAIL" if passed is False else "-"))
    columns[3].metric("Flags", len(qc.get("flags", []) or []))

    if passed is False:
        st.error(
            "This subject is below the configured QC threshold. Downstream "
            "results for it should be treated as low confidence."
        )

    st.markdown("**Measured QC metrics**")
    st.json(qc)
    for warning in manifest.get("warnings", []):
        st.warning(warning)
    stage_detail(record)
    what_this_module_did(
        "Runs the preserved `ArtifactDetector`, which measures SNR, motion "
        "indicators, signal dropout, Gibbs ringing and saturation, and combines "
        "them into a 0-100 score. Every value shown is measured from the "
        "volume; none is assumed."
    )


def page_m3(state: DashboardState, subject: Optional[str]) -> None:
    """M3 — preprocessing chain, every intermediate stage exposed."""
    record = state.stage_record("M3", subject)
    module_header("M3", "MRI preprocessing",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M3"].description)
    provenance_banner(state)
    if _imaging_unavailable(state, subject, "M3"):
        stage_detail(record)
        return

    manifest = state.preprocessing(subject)
    stages = manifest.get("stages", [])
    if not stages:
        unavailable("preprocessing stage records",
                    "python run.py --mode preprocess")
        return

    st.markdown("### Stage-by-stage output")
    frame = pd.DataFrame([
        {
            "Stage": s["name"],
            "Description": s["description"],
            "Shape": str(s.get("shape")),
            "Intensity range": str(
                [round(v, 4) for v in (s.get("intensity_range") or [])]
            ),
            "Mean": None if s.get("mean_intensity") is None
            else round(s["mean_intensity"], 5),
            "Non-zero voxels": s.get("nonzero_voxels"),
            "Seconds": s.get("seconds"),
        }
        for s in stages
    ])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    dataframe_download(frame, f"{subject}_preprocessing_stages")

    for stage in stages:
        if stage.get("path") and Path(stage["path"]).exists():
            download_button(Path(stage["path"]),
                            label=f"Download {stage['name']}")

    stage_detail(record)
    what_this_module_did(
        "Applies, in order: N4 bias field correction, white-matter KDE peak "
        "normalization, CLAHE, Perona-Malik anisotropic diffusion and min-max "
        "intensity normalization. Each row above is the volume's measured state "
        "**after** that step, so a step that changed nothing is visible as such. "
        "A step that fails is recorded with its error and the previous volume "
        "is carried forward rather than the subject being lost."
    )


def page_m4(state: DashboardState, subject: Optional[str]) -> None:
    """M4 — skull stripping."""
    record = state.stage_record("M4", subject)
    module_header("M4", "Skull stripping",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M4"].description)
    provenance_banner(state)
    if _imaging_unavailable(state, subject, "M4"):
        stage_detail(record)
        return

    manifest = state.preprocessing(subject)
    before = manifest.get("brain_voxels_before")
    after = manifest.get("brain_voxels_after")
    columns = st.columns(3)
    columns[0].metric("Voxels before", f"{before:,}" if before else "-")
    columns[1].metric("Voxels after", f"{after:,}" if after else "-")
    columns[2].metric(
        "Retained",
        f"{after / before:.1%}" if before and after else "-",
    )
    if before and after and after / before > 0.95:
        st.warning(
            "More than 95% of voxels were retained, which usually means skull "
            "stripping had little effect. Inspect the mask before trusting the "
            "downstream ROI extraction."
        )
    if manifest.get("brain_mask_path"):
        st.caption(f"Brain mask: `{manifest['brain_mask_path']}`")
    stage_detail(record)
    what_this_module_did(
        "Runs the preserved `SkullStripper` (Nilearn with a SimpleITK "
        "fallback), producing a brain mask and the masked volume. The retained "
        "voxel fraction is the fastest check that the step did something."
    )


def page_m5(state: DashboardState, subject: Optional[str]) -> None:
    """M5 — spatial standardization."""
    record = state.stage_record("M5", subject)
    module_header("M5", "Spatial standardization",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M5"].description)
    provenance_banner(state)
    if _imaging_unavailable(state, subject, "M5"):
        stage_detail(record)
        return

    manifest = state.preprocessing(subject)
    metrics = (record or {}).get("metrics", {})
    columns = st.columns(3)
    columns[0].metric("Shape before", str(metrics.get("shape_before")))
    columns[1].metric("Target shape",
                      str(list(state.cfg.preprocess.target_shape)))
    columns[2].metric("Shape after", str(metrics.get("shape_after")))
    if manifest.get("final_path"):
        st.caption(f"Standardised volume: `{manifest['final_path']}`")
        path = Path(manifest["final_path"])
        if path.exists():
            download_button(path, "Download standardised volume")
    stage_detail(record)
    what_this_module_did(
        "Resamples every subject onto a common 128x128x128 grid with SimpleITK, "
        "so that ROI bounding boxes and patch coordinates are comparable across "
        "subjects. The affine is replaced with identity afterwards because the "
        "original affine no longer describes the resampled grid."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pages M6-M8 (Sections 46-48)
# ──────────────────────────────────────────────────────────────────────────────

def page_m6(state: DashboardState, subject: Optional[str]) -> None:
    """M6 — Harvard-Oxford speech ROI localization."""
    record = state.stage_record("M6", subject)
    module_header("M6", "Harvard-Oxford speech ROI extraction",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M6"].description)
    provenance_banner(state)

    st.markdown("### The five speech-related regions")
    st.caption(
        "The atlas is the anatomical localization mechanism, not the research "
        "contribution. NeuroProp-X begins after this module."
    )
    selected = roi_selector("m6_roi")
    roi_metadata_card(selected)

    manifest = state.segmentation(subject) if subject else None
    if manifest is None:
        unavailable(
            "atlas ROI localization for this subject",
            "python run.py --mode preprocess",
            reason=(
                "The synthetic artifact generator starts at the patch stage, so "
                "it produces no atlas masks."
                if state.is_synthetic else
                "The ROI extraction stage has not been run."
            ),
        )
        stage_detail(record)
        return

    st.markdown("### Measured ROI statistics")
    frame = pd.DataFrame(manifest.get("rois", []))
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        dataframe_download(frame, f"{subject}_roi_stats")
    if manifest.get("n_empty_rois"):
        st.error(
            f"{manifest['n_empty_rois']} ROI mask(s) are empty for this "
            "subject, which usually indicates atlas/subject misalignment."
        )
    for warning in manifest.get("warnings", []):
        st.warning(warning)
    stage_detail(record)
    what_this_module_did(
        "Resamples the Harvard-Oxford cortical atlas onto the subject grid and "
        "builds a binary mask for each of the five speech regions, then records "
        "each region's voxel count, volume, bounding box and centroid. Empty "
        "masks are reported rather than silently passed on."
    )


def page_m7(state: DashboardState, subject: Optional[str]) -> None:
    """M7 — ROI patch extraction and the (5, 48, 48, 48) tensor."""
    record = state.stage_record("M7", subject)
    module_header("M7", "ROI patch extraction",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M7"].description)
    provenance_banner(state)

    tensor = state.patch_tensor(subject) if subject else None
    if tensor is None:
        unavailable("the ROI patch tensor for this subject",
                    "python run.py --mode preprocess")
        stage_detail(record)
        return

    columns = st.columns(4)
    columns[0].metric("Tensor shape", str(tuple(tensor.shape)))
    columns[1].metric("Voxels", f"{tensor.size:,}")
    columns[2].metric("Intensity range",
                      f"[{tensor.min():.3f}, {tensor.max():.3f}]")
    columns[3].metric("Patch size",
                      str(list(state.cfg.preprocess.patch_size)))

    st.markdown(
        f"Extraction path: full MRI → atlas mask → 3-D bounding box → "
        f"{state.cfg.preprocess.context_pad}-voxel context padding → "
        f"resize to {tuple(state.cfg.preprocess.patch_size)} → stack into "
        f"**{tuple(tensor.shape)}**, ordered by `ROI_ORDER`."
    )

    selected = roi_selector("m7_roi")
    index = ROI_ORDER.index(selected)
    patch = tensor[index]
    columns = st.columns(4)
    columns[0].metric("Shape", str(tuple(patch.shape)))
    columns[1].metric("Non-zero voxels", f"{int((patch > 0.05).sum()):,}")
    columns[2].metric("Mean", f"{patch.mean():.4f}")
    columns[3].metric("SD", f"{patch.std():.4f}")

    left, right = st.columns([3, 2])
    with left:
        slice_views(patch, f"{roi_short(selected)} — tri-plane mid-slices")
    with right:
        intensity_histogram(patch, "Patch intensity distribution")

    st.markdown("### All five patches")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, N_ROI, figsize=(14, 3.0))
    for ax, (i, roi) in zip(axes, enumerate(ROI_ORDER)):
        middle = tensor[i][:, :, tensor.shape[3] // 2]
        ax.imshow(np.rot90(middle), cmap="gray")
        ax.set_title(roi_short(roi), fontsize=9)
        ax.axis("off")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    array_download(tensor, f"{subject}_roi_tensor")
    stage_detail(record)
    what_this_module_did(
        "Crops a context-padded cube around each ROI's bounding box and resizes "
        "it to 48x48x48, then stacks the five patches in `ROI_ORDER`. A missing "
        "region raises rather than being zero-filled, because a blank patch "
        "labelled with the subject's stage would silently corrupt training. "
        "Output feeds M8 (features) and M9 (3D CNN)."
    )


def page_m8(state: DashboardState, subject: Optional[str]) -> None:
    """M8 — morphometric feature extraction."""
    record = state.stage_record("M8", subject)
    module_header("M8", "Morphometric feature extraction",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M8"].description)
    provenance_banner(state)

    features = state.features()
    if features is None:
        unavailable("the feature table", "python run.py --mode preprocess")
        stage_detail(record)
        return

    from modules.m04_feature_extraction.feature_spec import (
        EXCLUDED_FEATURES,
        FEATURE_DEFINITIONS,
        FEATURE_ORDER,
    )

    quality = state.feature_quality()
    if quality:
        columns = st.columns(4)
        columns[0].metric("Sessions", quality.get("n_sessions", 0))
        columns[1].metric("Rows", quality.get("n_rows", 0))
        columns[2].metric("Usable features",
                          f"{len(quality.get('usable_features', []))}"
                          f"/{len(FEATURE_ORDER)}")
        columns[3].metric("Degraded", len(quality.get("degraded", {})))
        for name, reason in (quality.get("degraded") or {}).items():
            st.warning(f"**{name}** is degraded: {reason}")
        for warning in quality.get("warnings", []):
            st.warning(warning)

    if subject:
        subset = features[features["session_id"].astype(str) == subject]
        if not subset.empty:
            st.markdown(f"### Raw feature values — `{subject}`")
            display = subset.set_index("roi_name")[
                [c for c in FEATURE_ORDER if c in subset.columns]
            ].T
            display.columns = [roi_short(c) for c in display.columns]
            st.dataframe(
                display.style.background_gradient(cmap="Blues", axis=1)
                .format("{:.4g}"),
                use_container_width=True, height=520,
            )
            dataframe_download(subset, f"{subject}_features")

    scaler = state.scaler()
    if scaler and subject:
        with st.expander("Standardised values and the fitted scaler"):
            st.caption(
                "The scaler is fitted on the training split only. Centre is the "
                f"per-ROI median, scale is IQR/{1.349}. "
                f"Clip: {scaler.get('clip')}."
            )
            st.json(scaler.get("fit_report", {}))

    st.markdown("### Feature definitions")
    definitions = pd.DataFrame([
        {
            "Feature": name,
            "Label": FEATURE_DEFINITIONS[name]["label"],
            "Unit": FEATURE_DEFINITIONS[name]["unit"],
            "Category": FEATURE_DEFINITIONS[name]["category"],
            "Source": FEATURE_DEFINITIONS[name]["source"],
            "Definition": FEATURE_DEFINITIONS[name]["definition"],
        }
        for name in FEATURE_ORDER
    ])
    st.dataframe(definitions, use_container_width=True, hide_index=True,
                 height=400)

    with st.expander("Deliberately excluded features"):
        for name, reason in EXCLUDED_FEATURES.items():
            st.markdown(f"- **{name}**: {reason}")

    selected_feature = st.selectbox(
        "Inspect a feature's distribution across ROIs and stages",
        [c for c in FEATURE_ORDER if c in features.columns],
    )
    cohort = state.cohort()
    if cohort is not None and selected_feature:
        merged = features.merge(
            cohort[["session_id", "stage"]].astype({"session_id": str}),
            left_on=features["session_id"].astype(str),
            right_on="session_id", how="left", suffixes=("", "_c"),
        )
        pivot = merged.pivot_table(
            index="roi_name", columns="stage", values=selected_feature,
            aggfunc="mean",
        )
        pivot.index = [roi_short(i) for i in pivot.index]
        st.dataframe(
            pivot[[s for s in STAGE_ORDER if s in pivot.columns]]
            .style.background_gradient(cmap="RdYlBu_r", axis=1)
            .format("{:.4g}"),
            use_container_width=True,
        )
        st.caption(
            "Stage-wise means. These are cross-sectional group differences, not "
            "within-subject change."
        )

    dataframe_download(features, "morphometric_features")
    stage_detail(record)
    what_this_module_did(
        "Runs the preserved `FeatureExtractor` on each ROI patch to compute "
        "volumetric, intensity and textural features, then derives grey-matter "
        "fraction, eTIV-normalized volume, surface-to-volume ratio and the "
        "training-referenced atrophy index. The quality audit flags any feature "
        "that has silently degraded to a constant."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page M9 (Section 49)
# ──────────────────────────────────────────────────────────────────────────────

def page_m9(state: DashboardState, subject: Optional[str]) -> None:
    """M9 — 3D CNN spatial encoding, with intermediate tensor shapes."""
    record = state.stage_record("M9", subject)
    module_header("M9", "3D CNN spatial encoding",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M9"].description)
    provenance_banner(state)

    st.info(
        "The 3D CNN captures local three-dimensional spatial structure inside "
        "each ROI patch — the patterns hand-crafted morphometric features "
        "cannot express."
    )

    from modules.common.config import SpatialEncoderConfig
    from modules.m06_spatial_encoder.cnn3d import SpatialEncoder3D

    st.markdown("### Architecture and intermediate tensor shapes")
    encoder = SpatialEncoder3D(state.cfg.spatial_encoder)
    shapes = encoder.layer_shapes(state.cfg.preprocess.patch_size)
    st.dataframe(
        pd.DataFrame([
            {
                "Stage": s["name"],
                "Kind": s["kind"],
                "Output shape (per patch)": str(tuple(s["output_shape"])),
                "Elements": f"{s['n_elements']:,}",
                "Parameters": f"{s['n_params']:,}",
            }
            for s in shapes
        ]),
        use_container_width=True, hide_index=True,
    )
    columns = st.columns(3)
    columns[0].metric("Trainable parameters", f"{encoder.n_parameters():,}")
    columns[1].metric("Embedding dim", encoder.embed_dim)
    columns[2].metric("Encoder shared across ROIs",
                      str(state.cfg.spatial_encoder.shared_encoder))
    st.caption(
        "Shapes are measured by a real traced forward pass over a zero tensor, "
        "not hand-computed, so they cannot drift from the code."
    )

    embeddings, summary = state.cnn_embeddings(subject) if subject else (None, None)
    if embeddings is None:
        unavailable(
            "cached CNN embeddings for this subject",
            "python run.py --mode train_cnn",
            reason=(
                "Note that `--mode train_full` trains the encoder jointly and "
                "does not write cached embeddings; the cache exists for "
                "inspection."
            ),
        )
        stage_detail(record)
        return

    if summary and (record or {}).get("metrics", {}).get(
            "encoder_status") == "untrained":
        st.warning(
            "These embeddings were produced by an **untrained** encoder. They "
            "describe the random initialisation, not learned structure."
        )

    st.markdown("### Per-ROI embeddings")
    columns = st.columns(3)
    columns[0].metric("Shape", str(tuple(embeddings.shape)))
    columns[1].metric("Mean", f"{embeddings.mean():.4f}")
    columns[2].metric("SD", f"{embeddings.std():.4f}")

    statistics = (summary or {}).get("per_roi_statistics", {})
    if statistics:
        st.dataframe(
            pd.DataFrame([
                {"ROI": roi_short(roi), **{k: round(v, 5)
                                           for k, v in values.items()}}
                for roi, values in statistics.items()
            ]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Embedding similarity between regions")
    normalised = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )
    matrix_view(
        normalised @ normalised.T, "Cosine similarity",
        labels=[roi_short(r) for r in ROI_ORDER], cmap="Purples",
        download_name=f"{subject}_embedding_similarity",
    )

    st.markdown("### PCA projection of the five ROI embeddings")
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    if centred.shape[0] >= 2:
        _, singular, components = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ components[:2].T
        variance = singular ** 2
        explained = variance / variance.sum() if variance.sum() else variance
        st.dataframe(
            pd.DataFrame({
                "ROI": [roi_short(r) for r in ROI_ORDER],
                f"PC1 ({explained[0]:.1%})": projected[:, 0],
                f"PC2 ({explained[1]:.1%})": projected[:, 1],
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "PCA is used rather than UMAP or t-SNE: it is deterministic and its "
            "axes carry a stated explained-variance fraction, so apparent "
            "separation cannot be a hyper-parameter artifact."
        )

    array_download(embeddings, f"{subject}_cnn_embeddings")
    stage_detail(record)
    what_this_module_did(
        "Each 48x48x48 patch passes through three Conv3D-BatchNorm-ReLU blocks "
        "(two with max-pooling), global average pooling and a linear projection "
        "to a 128-dimensional embedding. One encoder is shared across ROIs, with "
        "a learned per-ROI embedding added to the projection, because per-ROI "
        "encoders would quintuple the parameters for the same data. Output "
        "feeds M10 and M11."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page M10 (Section 50)
# ──────────────────────────────────────────────────────────────────────────────

def page_m10(state: DashboardState, subject: Optional[str]) -> None:
    """M10 — brain graph construction and the anatomical prior."""
    record = state.stage_record("M10", subject)
    module_header("M10", "Brain graph construction",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M10"].description)
    provenance_banner(state)

    from modules.m05_graph_construction.anatomical_prior import (
        PRIOR_PROVENANCE,
        build_prior_matrix,
        describe_prior,
        edge_table,
    )

    st.markdown("### Nodes")
    st.dataframe(
        pd.DataFrame([
            {
                "Node": roi_short(roi),
                "Identifier": roi,
                **{k: v for k, v in
                   __import__("modules.common.roi_constants",
                              fromlist=["ROI_METADATA"]).ROI_METADATA[roi].items()
                   if k != "color"},
            }
            for roi in ROI_ORDER
        ]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Anatomical prior `A_prior`")
    st.code(describe_prior(), language="text")
    matrix_view(build_prior_matrix(), "A_prior (directed, asymmetric)",
                cmap="Blues", download_name="A_prior")

    st.markdown("**Edge list with named white-matter tracts**")
    edges = pd.DataFrame(edge_table())
    st.dataframe(edges, use_container_width=True, hide_index=True)
    dataframe_download(edges, "anatomical_prior_edges")

    interpretation_note(
        "Provenance: these are expert-assigned ordinal weights encoding the "
        "relative strength and directness of established speech-network "
        "pathways. They are <b>not</b> measured DTI fractional anisotropy "
        "values — no diffusion data exists in this study. The prior's empirical "
        "contribution is tested by ablations A1 and A3 rather than assumed."
    )
    with st.expander("Full provenance record"):
        st.json(PRIOR_PROVENANCE)

    metrics = (record or {}).get("metrics", {})
    if metrics:
        st.markdown("### Subject-specific graph")
        columns = st.columns(3)
        columns[0].metric("Nodes", metrics.get("n_nodes", N_ROI))
        columns[1].metric("Node input width",
                          metrics.get("node_input_dim", "-"))
        columns[2].metric("Prior edges", len(edges))

    stage_detail(record)
    what_this_module_did(
        "Builds the subject graph G = (V, E): five speech-ROI nodes whose "
        "features are the concatenation H_i = [X_i_morph || E_i_3D], and edges "
        "from the documented anatomical prior. This ordinary graph is the input "
        "to NeuroProp-X, which turns it into the disease-aware enhanced graph G*."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page M11 and submodules (Sections 51-55)
# ──────────────────────────────────────────────────────────────────────────────

def page_m11(state: DashboardState, subject: Optional[str]) -> None:
    """M11 — NeuroProp-X master page with SRVE / AP-LAF / ANP / SAGR panels."""
    record = state.stage_record("M11", subject)
    module_header("M11", "NeuroProp-X",
                  (record or {}).get("status", "NOT_STARTED"),
                  "Stage-Aware Regional Vulnerability and Adaptive Graph "
                  "Propagation — the proposed framework.")
    provenance_banner(state)

    st.markdown(
        """
```
Initial graph G  ->  M11.1 SRVE   ->  RV       regional vulnerability
                 ->  M11.2 AP-LAF ->  A_star   prior fused with learned attention
                 ->  M11.3 ANP    ->  P        propagation representation
                 ->  M11.4 SAGR   ->  X_star   enriched node features
                 =>  Enhanced graph  G* = (V, X_star, A_star, P)
```
"""
    )

    statuses = st.columns(5)
    for column, code in zip(statuses, ("M11", "M11.1", "M11.2", "M11.3", "M11.4")):
        sub = state.stage_record(code, subject) or {}
        column.markdown(
            f"**{code}**<br>{status_pill(sub.get('status', 'NOT_STARTED'))}",
            unsafe_allow_html=True,
        )

    report = state.subject_report(subject) if subject else None
    artifacts = state.neuropropx(subject) if subject else {}
    npx = ((report or {}).get("explanations") or {}).get("neuropropx") or {}

    if not npx and not any(v is not None for v in artifacts.values()):
        unavailable(
            "NeuroProp-X outputs for this subject",
            "python run.py --mode report --patient_id <ID>",
        )
        stage_detail(record)
        return

    interpretation_note(
        "Regional vulnerability and propagation scores are model-derived "
        "quantities describing how informative each region and pathway is for "
        "the current CN/MCI/AD discrimination. They are <b>not</b> biological "
        "probabilities of degeneration and <b>not</b> measurements of disease "
        "spread."
    )

    # ── M11.1 SRVE ────────────────────────────────────────────────────────
    with st.expander("M11.1 — SRVE: Stage-Aware Regional Vulnerability Estimation",
                     expanded=True):
        formula(
            r"RV_i = \sigma\left(w_v^\top \hat{x}_i + b_v\right)",
            "A shared vulnerability direction w_v with a per-ROI bias, learned "
            "by the classification objective. The pre-refactor implementation "
            "computed this from hard-coded constants and had no trainable "
            "parameters at all.",
        )
        vulnerability = npx.get("regional_vulnerability") or {}
        if vulnerability:
            roi_bar(vulnerability, "Regional vulnerability RV_i", "vulnerability")
            ranking = pd.DataFrame(npx.get("vulnerability_ranking") or [])
            if not ranking.empty:
                st.dataframe(ranking, use_container_width=True, hide_index=True)
                dataframe_download(ranking, f"{subject}_vulnerability")
        elif artifacts.get("vulnerability") is not None:
            st.dataframe(artifacts["vulnerability"], use_container_width=True,
                         hide_index=True)
        else:
            st.caption("No vulnerability values available.")

        trace = npx.get("srve_trace")
        if trace:
            st.markdown(
                f"**Feature → calculation → vulnerability trace for "
                f"{roi_short(trace['roi'])}**"
            )
            terms = pd.DataFrame([
                {
                    "Feature": name,
                    "Normalised value": trace["features"].get(name),
                    "Learned weight": trace["weights"].get(name),
                    "Contribution": trace["contributions"].get(name),
                }
                for name in trace["features"]
            ]).sort_values("Contribution", key=abs, ascending=False)
            st.dataframe(terms, use_container_width=True, hide_index=True)
            columns = st.columns(4)
            columns[0].metric("Sum of contributions",
                              f"{sum(trace['contributions'].values()):.5f}")
            columns[1].metric("Bias", f"{trace['bias']:.5f}")
            columns[2].metric("Logit", f"{trace['logit']:.5f}")
            columns[3].metric("RV", f"{trace['regional_vulnerability']:.5f}")
            st.caption(
                "The decomposition is exact: contributions plus bias equal the "
                "pre-sigmoid logit, because SRVE is linear before the sigmoid."
            )

    # ── M11.2 AP-LAF ──────────────────────────────────────────────────────
    with st.expander("M11.2 — AP-LAF: Anatomical-Prior and Learned-Attention "
                     "Fusion", expanded=True):
        formula(
            r"A^* = \alpha\, A_{\text{prior}} + (1-\alpha)\, A_{\text{att}},"
            r"\qquad \alpha = \sigma(a)",
            "Both operands are row-stochastic before mixing, so alpha reads "
            "directly as the share of adjacency mass contributed by anatomy.",
        )
        alpha = npx.get("alpha")
        if alpha is not None:
            columns = st.columns(3)
            columns[0].metric("alpha (prior weight)", f"{alpha:.4f}")
            columns[1].metric("1 - alpha (learned weight)", f"{1 - alpha:.4f}")
            columns[2].metric("alpha is learned", "yes")

        panels = [
            ("A_prior", npx.get("prior_adjacency") or artifacts.get("A_prior"),
             "Blues"),
            ("A_att", npx.get("attention_adjacency") or artifacts.get("A_att"),
             "Greens"),
            ("A_star", npx.get("adaptive_adjacency") or artifacts.get("A_star"),
             "Purples"),
        ]
        columns = st.columns(3)
        for column, (name, matrix, cmap) in zip(columns, panels):
            with column:
                if matrix is None:
                    st.caption(f"{name}: not available")
                    continue
                matrix_view(np.asarray(matrix), name, cmap=cmap,
                            download_name=f"{subject}_{name}")

    # ── M11.3 ANP ─────────────────────────────────────────────────────────
    with st.expander("M11.3 — ANP: Adaptive Neurodegeneration Propagation",
                     expanded=True):
        formula(
            r"P_{ij} = \sigma\left(\beta_1 RV_i + \beta_2 RV_j "
            r"+ \beta_3 A^*_{ij} + \beta_4 T_{ij} + b\right)",
            "Separate source and target coefficients preserve directionality; "
            "T_ij is a normalised hub-to-hub topology term.",
        )
        beta = npx.get("beta")
        if beta:
            st.dataframe(
                pd.DataFrame([{"Coefficient": k, "Value": round(v, 5)}
                              for k, v in beta.items()]),
                use_container_width=True, hide_index=True,
            )
        propagation = npx.get("propagation_matrix") or artifacts.get("propagation")
        if propagation is not None:
            matrix_view(np.asarray(propagation),
                        "Propagation representation P", cmap="Reds",
                        download_name=f"{subject}_propagation")
        pathways = pd.DataFrame(npx.get("top_pathways") or [])
        if not pathways.empty:
            st.markdown("**Highest-scoring directed pathways**")
            st.dataframe(pathways, use_container_width=True, hide_index=True)
        st.warning(
            "Label: model-derived propagation representation. NOT a biological "
            "disease-spread probability."
        )

    # ── M11.4 SAGR ────────────────────────────────────────────────────────
    with st.expander("M11.4 — SAGR: Stage-Aware Graph Representation",
                     expanded=True):
        st.markdown(
            "`X_i_star = [ X_i_morph || E_i_3D || RV_i || centrality ]`"
        )
        metadata = artifacts.get("metadata") or {}
        slices = metadata.get("component_slices") or (
            (record or {}).get("metrics", {}).get("component_slices")
        )
        sub_record = state.stage_record("M11.4", subject) or {}
        slices = slices or sub_record.get("metrics", {}).get("component_slices")
        if slices:
            st.dataframe(
                pd.DataFrame([
                    {"Component": name, "Start": bounds[0], "Stop": bounds[1],
                     "Width": bounds[1] - bounds[0]}
                    for name, bounds in slices.items()
                ]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Column ranges let the XAI layer attribute an importance value "
                "back to the kind of evidence it came from, rather than to an "
                "anonymous index."
            )
        x_star = artifacts.get("X_star")
        if x_star is not None:
            st.markdown(f"**Enhanced node feature matrix — shape "
                        f"{tuple(np.asarray(x_star).shape)}**")
            frame = pd.DataFrame(
                np.asarray(x_star),
                index=[roi_short(r) for r in ROI_ORDER],
            )
            st.dataframe(frame.style.format("{:.4f}"), use_container_width=True)
            array_download(np.asarray(x_star), f"{subject}_X_star")
        if metadata:
            with st.expander("Complete G* metadata"):
                st.json(metadata)

    stage_detail(record)
    what_this_module_did(
        "NeuroProp-X transforms the ordinary graph G into the disease-aware "
        "enhanced graph G* = (V, X_star, A_star, P). SRVE estimates which "
        "regions are discriminative, AP-LAF decides how much to trust anatomy "
        "versus learned attention, ANP builds a vulnerability-aware edge "
        "representation, and SAGR assembles the enriched node features. This is "
        "the framework's proposed algorithmic contribution; the underlying "
        "techniques are established and are not claimed as new."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pages M12-M14 (Sections 56-58)
# ──────────────────────────────────────────────────────────────────────────────

def page_m12(state: DashboardState, subject: Optional[str]) -> None:
    """M12 — SAEG-GATv2 attention, edge gates and classification."""
    record = state.stage_record("M12", subject)
    module_header("M12", "SAEG-GATv2",
                  (record or {}).get("status", "NOT_STARTED"),
                  "Stage-Aware Edge-Gated Graph Attention Network v2.")
    provenance_banner(state)

    formula(
        r"g_{ij} = \sigma\left(w_g^\top E_{ij} + b_g\right), \qquad "
        r"\tilde{\alpha}_{ij} = g_{ij}\,\alpha_{ij}",
        "Edge features E_ij = [A_prior_ij, A_att_ij, P_ij] come from "
        "NeuroProp-X. GATv2 (Brody et al., 2022) is the established foundation "
        "and is not claimed as novel; the proposed elements are the enriched "
        "node input, the three-channel edge feature and this gate.",
    )

    checkpoints = state.checkpoints()
    if checkpoints:
        from modules.model import NeuroGenesisModel

        variant = st.selectbox("Checkpoint", sorted(checkpoints),
                               key="m12_ckpt")
        try:
            model = NeuroGenesisModel.load_checkpoint(checkpoints[variant])
            summary = model.graph_encoder.summary()
            st.markdown("### Layer-by-layer architecture")
            architecture = summary.get("architecture")
            if architecture:
                st.dataframe(pd.DataFrame(architecture),
                             use_container_width=True, hide_index=True)
            columns = st.columns(4)
            columns[0].metric("Layers", summary.get("n_layers", "-"))
            columns[1].metric("Heads", summary.get("heads", "-"))
            columns[2].metric("Graph embedding dim",
                              summary.get("graph_embedding_dim", "-"))
            columns[3].metric("Edge gate active",
                              str(summary.get("edge_gate_active")))
            with st.expander("Encoder summary"):
                st.json(summary)
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Could not load the checkpoint: {exc}")
    else:
        unavailable("a model checkpoint", "python run.py --mode train_full")

    report = state.subject_report(subject) if subject else None
    graph = ((report or {}).get("explanations") or {}).get("graph") or {}

    if graph:
        st.markdown("### Learned attention and edge gates")
        attention = graph.get("edge_attention")
        gate = graph.get("edge_gate")
        columns = st.columns(2)
        with columns[0]:
            if attention is not None:
                matrix_view(np.asarray(attention),
                            "Head-averaged gated attention", cmap="Purples",
                            download_name=f"{subject}_attention")
        with columns[1]:
            if gate is not None:
                matrix_view(np.asarray(gate), "Edge gate g_ij", cmap="Oranges",
                            download_name=f"{subject}_edge_gate")
            else:
                st.caption("The edge gate is disabled in this variant.")

        if graph.get("node_importance"):
            roi_bar(graph["node_importance"],
                    "Node importance (incoming attention mass)", "attention")

        edges = pd.DataFrame(graph.get("top_edges") or [])
        if not edges.empty:
            st.markdown("**Most attended pathways**")
            st.dataframe(edges, use_container_width=True, hide_index=True)

        layers = pd.DataFrame(graph.get("layers") or [])
        if not layers.empty:
            with st.expander("Per-layer trace"):
                st.dataframe(layers, use_container_width=True, hide_index=True)

    if report and report.get("class_probabilities"):
        st.markdown("### Classification output")
        stage_probability_row(report["class_probabilities"],
                              report.get("current_stage"))
        confidence = report.get("confidence") or {}
        columns = st.columns(3)
        columns[0].metric("Margin", f"{confidence.get('margin', 0):.4f}")
        columns[1].metric("Normalised entropy",
                          f"{confidence.get('normalized_entropy', 0):.4f}")
        columns[2].metric("Assigned stage", report.get("current_stage", "-"))
        st.caption(confidence.get("note", ""))
    elif not graph:
        unavailable("per-subject graph outputs",
                    "python run.py --mode report --patient_id <ID>")

    stage_detail(record)
    what_this_module_did(
        "Two SAEG-GATv2 layers consume G*. Each computes GATv2 dynamic "
        "attention from node features, then multiplies it by a gate derived "
        "from the NeuroProp-X edge features. Because the gated attention is not "
        "renormalised, the gate can attenuate a node's total incoming message "
        "rather than only redistributing it. A mean+max readout produces the "
        "graph embedding Z_G."
    )


def page_m13(state: DashboardState, subject: Optional[str]) -> None:
    """M13 — multimodal fusion."""
    record = state.stage_record("M13", subject)
    module_header("M13", "Multimodal fusion",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M13"].description)
    provenance_banner(state)

    st.markdown("`Z_F = [Z_3D || Z_G]`  →  `Z_H = MLP(Z_F)`")
    st.info(
        "Z_H is shared by the current-stage classifier and the Stage-TGT. If "
        "each had its own trunk, the stage prototypes would live in a different "
        "space from the classifier's decision geometry and 'distance to the AD "
        "prototype' would bear no relation to 'probability of AD'."
    )

    checkpoints = state.checkpoints()
    if checkpoints:
        from modules.model import NeuroGenesisModel

        variant = st.selectbox("Checkpoint", sorted(checkpoints), key="m13_ckpt")
        try:
            model = NeuroGenesisModel.load_checkpoint(checkpoints[variant])
            summary = model.fusion.summary()
            columns = st.columns(4)
            columns[0].metric("Z_3D width", summary["spatial_dim"])
            columns[1].metric("Z_G width", summary["graph_dim"])
            columns[2].metric("Z_F width", summary["fused_dim"])
            columns[3].metric("Z_H width", summary["out_dim"])
            st.json(summary)
            st.markdown("### Parameters by component")
            st.dataframe(
                pd.DataFrame([
                    {"Component": k, "Parameters": v}
                    for k, v in model.component_parameters().items()
                ]),
                use_container_width=True, hide_index=True,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Could not load the checkpoint: {exc}")
    else:
        unavailable("a model checkpoint", "python run.py --mode train_full")

    stage_detail(record)
    what_this_module_did(
        "Projects the five per-ROI CNN embeddings into one spatial branch "
        "vector, concatenates it with the graph embedding and passes the result "
        "through a small two-layer MLP with LayerNorm and dropout. The head is "
        "kept small deliberately: it sees a fully-connected view of both "
        "branches at once and is the easiest place in the architecture to "
        "overfit."
    )


def page_m14(state: DashboardState, subject: Optional[str]) -> None:
    """M14 — Stage-TGT prototypes, transformer and propensity."""
    record = state.stage_record("M14", subject)
    module_header("M14", "Stage-TGT",
                  (record or {}).get("status", "NOT_STARTED"),
                  "Stage-Temporal Graph Transformer over ordered stage "
                  "representations.")
    provenance_banner(state)

    st.info(
        "The sequence axis is the **ordered stage vocabulary** CN → MCI → AD "
        "(clinical severity order), not observed time. This module requires no "
        "longitudinal data, forecasts no future MRI and simulates no future "
        "scans."
    )
    st.markdown(
        "```\ntokens:  [ c_CN , c_MCI , c_AD , Z_H ]  + stage positional "
        "embeddings\n```"
    )

    report = state.subject_report(subject) if subject else None
    detail = (report or {}).get("stage_propensity_detail") or {}
    geometry = (report or {}).get("stage_geometry") or {}

    if not detail:
        unavailable("Stage-TGT outputs for this subject",
                    "python run.py --mode report --patient_id <ID>")
        stage_detail(record)
        return

    st.markdown("### Stage-propensity outputs")
    columns = st.columns(4)
    columns[0].metric("Reference stage", detail.get("reference_stage", "-"))
    columns[1].metric("Advanced-stage alignment",
                      f"{detail.get('advanced_stage_alignment', 0):.4f}")
    columns[2].metric("AD-associated propensity",
                      f"{detail.get('ad_associated_propensity', 0):.4f}")
    transition = detail.get("stage_transition_propensity")
    columns[3].metric(
        "Stage-transition propensity",
        "undefined" if transition is None else f"{transition:.4f}",
    )
    st.caption(detail.get("transition_note", ""))

    alignment = detail.get("stage_alignment") or {}
    if alignment:
        st.markdown("**Stage alignment distribution**")
        st.dataframe(
            pd.DataFrame([{"Stage": s, "Alignment": alignment.get(s)}
                          for s in STAGE_ORDER]),
            use_container_width=True, hide_index=True,
        )

    if geometry:
        st.markdown("### Prototype geometry")
        ordered = geometry.get("ordering_respected")
        columns = st.columns(3)
        columns[0].metric("CN < MCI < AD respected", str(ordered))
        columns[1].metric("Temperature",
                          f"{geometry.get('temperature', 0):.3f}")
        columns[2].metric("d_model", geometry.get("d_model", "-"))
        st.dataframe(
            pd.DataFrame([
                {"Pair": k, "Distance": round(v, 4)}
                for k, v in (geometry.get("pairwise_distances") or {}).items()
            ]),
            use_container_width=True, hide_index=True,
        )
        if ordered is False:
            st.error(
                "The learned prototypes do not respect the stage ordering at "
                "this checkpoint. The propensity values above therefore reflect "
                "under-trained geometry and must not be read as a finding. "
                "Increase `loss.lambda_order`, train for more epochs, or raise "
                "`train.early_stopping_patience`."
            )
        st.caption(geometry.get("ordering_check", ""))

    with st.expander("Definitions of every propensity quantity", expanded=False):
        for name, text in (detail.get("definitions") or {}).items():
            st.markdown(f"- **{name}**: {text}")

    interpretation_note(detail.get("disclaimer", ""))
    stage_detail(record)
    what_this_module_did(
        "Builds a four-token sequence from the three learned stage prototypes "
        "plus the subject's shared representation, adds stage positional "
        "embeddings (which is what makes the axis ordered rather than a bag of "
        "three labels), and runs multi-head self-attention over it. The "
        "propensity head then derives a learned stage alignment, an expected "
        "ordinal position, and a purely geometric relative proximity to the AD "
        "prototype."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pages M15-M19 (Sections 59-61)
# ──────────────────────────────────────────────────────────────────────────────

def page_m15(state: DashboardState, subject: Optional[str]) -> None:
    """M15 — unified ROI ranking and stability."""
    record = state.stage_record("M15", subject)
    module_header("M15", "ROI ranking",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M15"].description)
    provenance_banner(state)

    formula(
        r"\text{ROI score}_i = \lambda_1 RV_i + \lambda_2 "
        r"\text{Attention}_i + \lambda_3 \text{SHAP}_i",
        "Each signal is min-max rescaled within the subject before combining, "
        "because the three live on incompatible scales. Missing signals cause "
        "the remaining weights to be renormalised rather than zeros to be "
        "substituted.",
    )

    report = state.subject_report(subject) if subject else None
    ranking = (report or {}).get("roi_ranking")
    if ranking:
        st.markdown(f"### Subject ranking — `{subject}`")
        frame = pd.DataFrame(ranking)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        roi_bar({r["roi"]: r["score"] for r in ranking},
                "Combined ROI importance", "score")
        dataframe_download(frame, f"{subject}_roi_ranking")
    else:
        unavailable("a per-subject ROI ranking",
                    "python run.py --mode report --patient_id <ID>")

    table6 = state.roi_ranking_table("table6_stagewise_ranking")
    if table6 is not None:
        st.markdown("### Table 6 — stage-wise ROI ranking")
        st.dataframe(table6, use_container_width=True, hide_index=True)
        dataframe_download(table6, "table6_stagewise_ranking")
    else:
        unavailable("Table 6 (stage-wise ROI ranking)",
                    "python run.py --mode xai")

    stability = state.roi_ranking()
    if stability:
        st.markdown("### Table 8 — ROI ranking stability")
        st.caption(
            f"Aggregated over {stability.get('n_runs')} run(s), top-"
            f"{stability.get('top_k')} selection frequency."
        )
        rows = pd.DataFrame(stability.get("rows") or [])
        if not rows.empty:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            dataframe_download(rows, "table8_ranking_stability")
        for note in stability.get("notes", []):
            st.warning(note)
    else:
        unavailable("Table 8 (ranking stability)", "python run.py --mode xai")

    figures = state.figures()
    if "fig14_ranking_stability" in figures:
        st.image(str(figures["fig14_ranking_stability"]),
                 use_container_width=True)

    stage_detail(record)
    what_this_module_did(
        "Combines NeuroProp-X regional vulnerability, SAEG-GATv2 node "
        "attention, morphometric feature attribution and (when available) 3D "
        "CNN occlusion importance into one score per region, then aggregates "
        "across subjects for the stage-wise ranking and across repeats for the "
        "stability table. With a small AD class, a single ranking is not a "
        "finding — the stability table is what makes it reportable."
    )


def page_m16(state: DashboardState, subject: Optional[str]) -> None:
    """M16 — explainable AI."""
    record = state.stage_record("M16", subject)
    module_header("M16", "Explainable AI",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M16"].description)
    provenance_banner(state)

    report = state.subject_report(subject) if subject else None
    explanations = (report or {}).get("explanations") or {}
    if not explanations:
        unavailable("explanations for this subject",
                    "python run.py --mode report --patient_id <ID>")
        stage_detail(record)
        return

    predicted = (report or {}).get("current_stage")
    if predicted:
        st.caption(
            f"Every signal below explains the model's assignment of "
            f"**{predicted}** for `{subject}`."
        )

    attribution = explanations.get("attribution")
    st.markdown("### A. Feature attribution")
    if attribution:
        method = attribution.get("method")
        if attribution.get("is_shap"):
            st.success("Method: **Kernel SHAP** (genuine Shapley values).")
        else:
            st.warning(
                f"Method: **{method}** — exact single-feature replacement. "
                "These are NOT Shapley values. The `shap` package was "
                "unavailable or failed. Permutation attribution measures the "
                "marginal effect of replacing one feature with its "
                "training-split reference values and does not decompose "
                "additively."
            )
        columns = st.columns(3)
        columns[0].metric("Target stage", attribution.get("target_stage", "-"))
        columns[1].metric("Background size", attribution.get("n_background", 0))
        columns[2].metric("Signed", str(attribution.get("signed")))

        top = pd.DataFrame(attribution.get("top_features") or [])
        if not top.empty:
            st.dataframe(top, use_container_width=True, hide_index=True)
        per_roi = attribution.get("per_roi") or {}
        if per_roi:
            roi_bar(per_roi, "Attribution magnitude per region", "magnitude")
        per_feature = attribution.get("per_feature") or {}
        if per_feature:
            st.markdown("**Attribution magnitude per feature**")
            st.bar_chart(
                pd.DataFrame(
                    {"magnitude": per_feature}
                ).sort_values("magnitude", ascending=False),
                horizontal=True,
            )
        for note in attribution.get("notes", []):
            st.caption(note)
    else:
        st.caption("No feature attribution available for this subject.")

    st.markdown("### B. Graph attention")
    graph = explanations.get("graph") or {}
    if graph.get("edge_attention") is not None:
        columns = st.columns(2)
        with columns[0]:
            matrix_view(np.asarray(graph["edge_attention"]),
                        "Edge importance (gated attention)", cmap="Purples")
        with columns[1]:
            if graph.get("node_importance"):
                roi_bar(graph["node_importance"], "Node importance",
                        "attention")
        edges = pd.DataFrame(graph.get("top_edges") or [])
        if not edges.empty:
            st.dataframe(edges, use_container_width=True, hide_index=True)
    else:
        st.caption("No attention recorded for this variant.")
    for note in graph.get("notes", []):
        st.caption(note)

    st.markdown("### C. NeuroProp-X")
    npx = explanations.get("neuropropx") or {}
    if npx.get("regional_vulnerability"):
        columns = st.columns(2)
        with columns[0]:
            roi_bar(npx["regional_vulnerability"], "Regional vulnerability",
                    "vulnerability")
        with columns[1]:
            if npx.get("propagation_matrix") is not None:
                matrix_view(np.asarray(npx["propagation_matrix"]),
                            "Propagation representation", cmap="Reds")
        interpretation_note(npx.get("disclaimer", ""))
    else:
        st.caption("This variant has no NeuroProp-X.")

    st.markdown("### D. 3D CNN occlusion")
    occlusion = explanations.get("cnn_occlusion")
    if occlusion:
        columns = st.columns(3)
        columns[0].metric("Method", occlusion.get("method", "-"))
        columns[1].metric("Baseline", occlusion.get("baseline", "-"))
        columns[2].metric("Baseline probability",
                          f"{occlusion.get('baseline_probability', 0):.4f}")
        st.dataframe(pd.DataFrame(occlusion.get("ranking") or []),
                     use_container_width=True, hide_index=True)
        for note in occlusion.get("notes", []):
            st.caption(note)
    else:
        st.caption("No occlusion analysis available for this subject.")

    stage_detail(record)
    what_this_module_did(
        "Extracts the explanation signals the model actually computed: "
        "attention and edge-gate matrices recorded during the forward pass, "
        "NeuroProp-X vulnerability and propagation values taken from the module "
        "outputs, feature attribution against a training-split background, and "
        "whole-ROI occlusion of the CNN branch. Nothing here is re-derived from "
        "a formula, and the attribution method is always named so a permutation "
        "result cannot be presented as SHAP."
    )


def page_m17(state: DashboardState, subject: Optional[str]) -> None:
    """M17 — stage-wise statistical analysis."""
    record = state.stage_record("M17", None)
    module_header("M17", "Statistics",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M17"].description)
    provenance_banner(state)

    statistics = state.statistics()
    if statistics is None:
        unavailable("the statistics report", "python run.py --mode statistics")
        stage_detail(record)
        return

    columns = st.columns(4)
    columns[0].metric("Tests run", statistics.get("n_tests_run", 0))
    columns[1].metric("Skipped", statistics.get("n_tests_skipped", 0))
    columns[2].metric("FDR-significant",
                      statistics.get("n_significant_fdr", 0))
    columns[3].metric("Correction", statistics.get("fdr_method", "-"))

    st.markdown("**Group sizes**")
    st.dataframe(
        pd.DataFrame([statistics.get("group_sizes", {})]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Statistical tests actually used**")
    st.dataframe(
        pd.DataFrame([
            {"Test": k, "Cells": v}
            for k, v in (statistics.get("test_counts") or {}).items()
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Test choice is data-driven per cell: Shapiro-Wilk (or D'Agostino above "
        "n=50) screens normality and Levene screens equality of variance, then "
        "Student's t, Welch's t or Mann-Whitney U is selected accordingly. Cells "
        "below the minimum group size are reported as `insufficient_data` with "
        "no p-value."
    )

    for note in statistics.get("notes", []):
        st.warning(note)

    table5 = state.statistics_table("table5_stagewise")
    if table5 is not None:
        st.markdown("### Table 5 — stage-wise ROI statistical analysis")
        st.dataframe(table5, use_container_width=True, hide_index=True,
                     height=420)
        dataframe_download(table5, "table5_stagewise")

    table7 = state.statistics_table("table7_morphometry")
    if table7 is not None:
        st.markdown("### Table 7 — morphometric differences by stage")
        st.dataframe(table7, use_container_width=True, hide_index=True,
                     height=420)
        dataframe_download(table7, "table7_morphometry")

    figures = state.figures()
    stagewise = {k: v for k, v in figures.items()
                 if k.startswith(("fig06", "fig12"))}
    if stagewise:
        st.markdown("### Stage-wise difference figures")
        for name, path in stagewise.items():
            st.image(str(path), caption=name, use_container_width=True)

    stage_detail(record)
    what_this_module_did(
        "For every ROI x feature, tests CN vs MCI, MCI vs AD and CN vs AD, "
        "choosing the test from measured normality and variance rather than "
        "assuming one. Effect sizes match the test (Hedges' g for parametric, "
        "rank biserial for rank-based). Benjamini-Hochberg FDR is applied across "
        "all tests as a single family, not per contrast."
    )


def page_m18(state: DashboardState, subject: Optional[str]) -> None:
    """M18 — ablation and baselines."""
    record = state.stage_record("M18", None)
    module_header("M18", "Ablation",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M18"].description)
    provenance_banner(state)

    ablation = state.ablation()
    if ablation is None:
        unavailable("the ablation results", "python run.py --mode ablation")
    else:
        columns = st.columns(3)
        columns[0].metric("Variants", ablation.get("n_variants", 0))
        columns[1].metric("Repeats", ablation.get("n_repeats", 0))
        columns[2].metric("Reference", ablation.get("reference_variant", "-"))

        for name, title in (
            ("table4_ablation", "Table 4 — NeuroProp-X ablation ladder"),
            ("table2_main_performance", "Table 2 — main classification "
                                        "performance"),
            ("table3_class_wise", "Table 3 — class-wise performance"),
            ("table9_model_comparison", "Table 9 — final model comparison"),
        ):
            frame = state.ablation_table(name)
            if frame is not None:
                st.markdown(f"### {title}")
                st.dataframe(frame, use_container_width=True, hide_index=True)
                dataframe_download(frame, name)

        for caveat in ablation.get("caveats", []):
            st.warning(caveat)
        failures = ablation.get("failures") or []
        if failures:
            st.error(f"{len(failures)} run(s) failed:")
            st.dataframe(pd.DataFrame(failures), use_container_width=True,
                         hide_index=True)

    baselines = state.baseline_table()
    if baselines is not None:
        st.markdown("### Classical baselines")
        st.dataframe(baselines, use_container_width=True, hide_index=True)
        dataframe_download(baselines, "baseline_comparison")
        payload = state.baselines() or {}
        for caveat in payload.get("caveats", []):
            st.caption(caveat)

    figures = state.figures()
    if "fig15_ablation" in figures:
        st.image(str(figures["fig15_ablation"]), use_container_width=True)

    stage_detail(record)
    what_this_module_did(
        "Trains every variant A0-A7 on the *same* repeated stratified "
        "subject-wise splits, with the same trainer and the same selection "
        "criteria, so a measured difference is attributable to the component "
        "that changed. Disabled components are replaced by shape-preserving "
        "stand-ins rather than removed, so the comparison isolates the "
        "component's information rather than a change in capacity. Comparisons "
        "against the reference variant are paired across splits and use the "
        "Wilcoxon signed-rank test."
    )


def page_m19(state: DashboardState, subject: Optional[str]) -> None:
    """M19 — final report."""
    record = state.stage_record("M19", subject)
    module_header("M19", "Final report",
                  (record or {}).get("status", "NOT_STARTED"),
                  PIPELINE_BY_CODE["M19"].description)
    provenance_banner(state)

    report = state.subject_report(subject) if subject else None
    if report is None:
        unavailable("a report for this subject",
                    f"python run.py --mode report --patient_id {subject or '<ID>'}")
        stage_detail(record)
        return

    st.markdown("### Result summary")
    columns = st.columns(4)
    columns[0].metric("Assigned stage", report.get("current_stage", "-"))
    propensity = report.get("ad_associated_propensity")
    columns[1].metric("AD-associated propensity",
                      "n/a" if propensity is None else f"{propensity:.4f}")
    transition = report.get("stage_transition_propensity")
    columns[2].metric("Stage-transition propensity",
                      "undefined" if transition is None else f"{transition:.4f}")
    columns[3].metric("Reference stage", report.get("reference_stage") or "-")

    if report.get("class_probabilities"):
        stage_probability_row(report["class_probabilities"],
                              report.get("current_stage"))

    ranking = report.get("roi_ranking") or []
    if ranking:
        top = [r["roi_short"] for r in ranking[:3]]
        st.markdown(f"**Top contributing regions:** {', '.join(top)}")
    edges = report.get("important_edges") or []
    if edges:
        pathways = ", ".join(e.get("pathway", "") for e in edges[:3])
        st.markdown(f"**Most informative pathways:** {pathways}")

    stage = report.get("current_stage")
    if stage and propensity is not None:
        st.markdown(
            f"> **Interpretation.** The subject is classified as {stage}. The "
            f"current representation shows an AD-associated propensity of "
            f"{propensity:.2f}"
            + (f", and {ranking[0]['roi_short']} is the highest-contributing "
               "speech-related region." if ranking else ".")
        )

    interpretation_note(report.get("disclaimer", ""))

    markdown = state.report_markdown(subject)
    if markdown:
        with st.expander("Full report", expanded=False):
            st.markdown(markdown)
        report_dir = state.outputs / "reports" / subject
        for suffix in (".md", ".html", ".json"):
            path = report_dir / f"{subject}_report{suffix}"
            if path.exists():
                download_button(path, f"Download {suffix[1:].upper()} report")

    with st.expander("Machine-readable record (Section 31)"):
        st.json({k: v for k, v in report.items()
                 if k not in ("explanations", "config")})

    stage_detail(record)
    what_this_module_did(
        "Assembles the fifteen report sections from real computed values, in "
        "Markdown, HTML and JSON. A section whose inputs are absent renders as "
        "an explicit 'not available' block rather than being dropped. Before "
        "anything is written, the assembled text is scanned for prohibited "
        "clinical phrasing; the check is negation-aware so disclaimers can name "
        "what the system does not claim, and it raises rather than writing a "
        "non-compliant document."
    )


def page_figures(state: DashboardState, subject: Optional[str]) -> None:
    """All generated figures."""
    page_title("Figures", "Every figure generated from real computed values.")
    provenance_banner(state)
    figures = state.figures()
    if not figures:
        unavailable("generated figures", "python run.py --mode figures")
        return
    st.caption(
        "Figures showing an explicit 'Not available' panel are honest reports of "
        "a missing input, not rendering failures."
    )
    for name, path in figures.items():
        st.markdown(f"**{name}**")
        st.image(str(path), use_container_width=True)
        download_button(path)


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────

PAGES = {
    "Pipeline tracker": page_overview,
    "Patient selection": page_patient,
    "M1  MRI viewer": page_m1,
    "M2  Quality control": page_m2,
    "M3  Preprocessing": page_m3,
    "M4  Skull stripping": page_m4,
    "M5  Spatial standardization": page_m5,
    "M6  Harvard-Oxford ROIs": page_m6,
    "M7  ROI patches": page_m7,
    "M8  Morphometric features": page_m8,
    "M9  3D CNN embeddings": page_m9,
    "M10 Brain graph": page_m10,
    "M11 NeuroProp-X": page_m11,
    "M12 SAEG-GATv2": page_m12,
    "M13 Multimodal fusion": page_m13,
    "M14 Stage-TGT propensity": page_m14,
    "M15 ROI ranking": page_m15,
    "M16 Explainable AI": page_m16,
    "M17 Statistics": page_m17,
    "M18 Ablation": page_m18,
    "M19 Final report": page_m19,
    "Figures": page_figures,
}


def main() -> None:
    """Render the dashboard."""
    inject_theme()
    args = parse_args()

    cfg = NeuroGenesisConfig()
    cfg.paths.outputs_dir = args.outputs
    state = DashboardState(outputs=args.outputs, cfg=cfg)

    st.sidebar.markdown("## NeuroGenesis")
    st.sidebar.caption(
        "Stage-aware speech-network NeuroAI framework — pipeline inspection."
    )
    outputs_text = st.sidebar.text_input("Outputs directory",
                                         value=str(args.outputs))
    if outputs_text and Path(outputs_text) != state.outputs:
        cfg.paths.outputs_dir = Path(outputs_text)
        state = DashboardState(outputs=Path(outputs_text), cfg=cfg)

    if state.is_synthetic:
        st.sidebar.error("SYNTHETIC SMOKE-TEST TREE")

    subjects = state.subjects()
    subject = None
    if subjects:
        subject = st.sidebar.selectbox("Subject", subjects)
        completion = state.completion(subject)
        st.sidebar.progress(completion, text=f"{completion:.0%} of modules run")
    else:
        st.sidebar.warning("No subjects found in this outputs tree.")

    page = st.sidebar.radio("Module", list(PAGES), index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Module status is read from recorded execution state. A module that did "
        "not run shows NOT_STARTED; nothing here is hard-coded."
    )

    PAGES[page](state, subject)

    st.markdown("---")
    st.caption(
        "NeuroGenesis is a research system. Outputs are model-derived "
        "associations and stage-propensity estimates from a single "
        "cross-sectional scan. They are not a clinical diagnosis, not a "
        "validated prediction of future disease conversion, and not a basis for "
        "any clinical decision."
    )


if __name__ == "__main__":
    main()
