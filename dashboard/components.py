"""
Shared Streamlit UI components.
==============================

Every reusable rendering primitive the M1-M19 pages need, so the page code stays
about *what* is shown rather than how.

Two components carry policy rather than styling:

:func:`unavailable` — the single way a page reports a missing artifact. It always
names the command that would produce it, so an empty panel is actionable instead
of merely blank. Section 28 requires "Not available / requires training data"
rather than an invented output, and this is that surface.

:func:`provenance_banner` — renders the synthetic-data warning. It is called at
the top of every page that displays a number, so no metric can be read without
its provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from modules.common.roi_constants import (
    ROI_METADATA,
    ROI_ORDER,
    STAGE_COLOR,
    STAGE_ORDER,
    roi_short,
)

STATUS_STYLE: Dict[str, str] = {
    "COMPLETED": "background:#dafbe1;color:#116329;border:1px solid #4ac26b",
    "RUNNING": "background:#fff8c5;color:#7d4e00;border:1px solid #d4a72c",
    "FAILED": "background:#ffebe9;color:#a40e26;border:1px solid #ff8182",
    "SKIPPED": "background:#eef1f4;color:#57606a;border:1px solid #afb8c1",
    "NOT_STARTED": "background:#f6f8fa;color:#6e7781;border:1px dashed #afb8c1",
}


def inject_theme() -> None:
    """Apply the shared dashboard styling."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.6rem; max-width: 1500px; }
          .ng-title { font-size: 1.55rem; font-weight: 700; margin-bottom: .1rem; }
          .ng-sub { color: #6a737d; font-size: .92rem; margin-bottom: 1rem; }
          .ng-pill { display:inline-block; padding:.14rem .55rem; border-radius:1rem;
                     font-size:.72rem; font-weight:700; letter-spacing:.02em; }
          .ng-banner { padding:.8rem 1rem; border-radius:.55rem; font-weight:600;
                       margin-bottom:1rem; }
          .ng-danger { background:#ffebe9; border:1px solid #ff8182; color:#a40e26; }
          .ng-info { background:#ddf4ff; border:1px solid #54aeff; color:#0969da; }
          .ng-warn { background:#fff8c5; border:1px solid #d4a72c; color:#7d4e00; }
          .ng-note { color:#6a737d; font-size:.82rem; font-style:italic; }
          .ng-card { border:1px solid #d0d7de; border-radius:.55rem;
                     padding:.85rem 1rem; margin-bottom:.7rem; }
          .ng-mono { font-family: ui-monospace, Consolas, monospace;
                     font-size:.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str = "") -> None:
    """Render a page heading."""
    st.markdown(f'<div class="ng-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ng-sub">{subtitle}</div>',
                    unsafe_allow_html=True)


def provenance_banner(state: Any) -> None:
    """Render the synthetic-data warning when the outputs tree is a smoke test.

    Called at the top of every page that shows a metric. Without it a viewer
    could read a perfect confusion matrix off synthetic blobs and take it for a
    result.
    """
    marker = state.smoke_marker
    if not marker:
        return
    st.markdown(
        '<div class="ng-banner ng-danger">SYNTHETIC SMOKE-TEST DATA &mdash; '
        "NOT A RESEARCH RESULT<br><span style='font-weight:400;font-size:.85rem'>"
        f"{marker.get('warning', '')}</span></div>",
        unsafe_allow_html=True,
    )


def unavailable(what: str, produced_by: str = "",
                reason: str = "") -> None:
    """Report a missing artifact, naming the command that produces it."""
    lines = [f"**Not available — {what}**"]
    if reason:
        lines.append(reason)
    if produced_by:
        lines.append(f"Produce it with: `{produced_by}`")
    st.info("\n\n".join(lines))


def status_pill(status: str) -> str:
    """Return an HTML pill for a module status."""
    style = STATUS_STYLE.get(status, STATUS_STYLE["NOT_STARTED"])
    return f'<span class="ng-pill" style="{style}">{status}</span>'


def module_header(code: str, title: str, status: str,
                  description: str = "") -> None:
    """Render a module heading with its real execution status."""
    st.markdown(
        f'<div class="ng-title">{code} &nbsp; {title} &nbsp; '
        f"{status_pill(status)}</div>",
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(f'<div class="ng-sub">{description}</div>',
                    unsafe_allow_html=True)


def stage_detail(record: Optional[Dict[str, Any]]) -> None:
    """Render a module's recorded metrics, artifacts, timing and errors.

    This is the Section 39 requirement: for every module, show the processing
    status, numerical values, downloadable artifacts and any error.
    """
    if not record:
        return

    columns = st.columns(4)
    columns[0].metric("Status", record.get("status", "NOT_STARTED"))
    duration = record.get("duration_s")
    columns[1].metric("Duration",
                      f"{duration:.3f}s" if duration is not None else "-")
    columns[2].metric("Artifacts", len(record.get("artifacts") or {}))
    columns[3].metric("Metrics", len(record.get("metrics") or {}))

    if record.get("error"):
        st.error(f"**{record['error']}**")
        if record.get("traceback"):
            with st.expander("Traceback"):
                st.code(record["traceback"], language="text")

    metrics = record.get("metrics") or {}
    if metrics:
        with st.expander("Recorded values", expanded=True):
            st.json(metrics)

    artifacts = record.get("artifacts") or {}
    if artifacts:
        st.markdown("**Artifacts**")
        for name, path in artifacts.items():
            file = Path(path)
            if file.exists():
                download_button(file, label=f"Download {name}")
            else:
                st.markdown(
                    f'<span class="ng-mono">{name}: {path} (file no longer '
                    "present)</span>", unsafe_allow_html=True,
                )

    for note in record.get("notes") or []:
        st.markdown(f'<div class="ng-note">{note}</div>',
                    unsafe_allow_html=True)


def download_button(path: Path, label: Optional[str] = None,
                    key: Optional[str] = None) -> None:
    """Offer a file for download, guarding against unreadable files."""
    path = Path(path)
    if not path.exists():
        return
    try:
        payload = path.read_bytes()
    except OSError as exc:
        st.caption(f"Cannot read {path.name}: {exc}")
        return
    st.download_button(
        label or f"Download {path.name}",
        data=payload,
        file_name=path.name,
        key=key or f"dl_{path.as_posix()}",
    )


def dataframe_download(frame: pd.DataFrame, name: str,
                       key: Optional[str] = None) -> None:
    """Offer an in-memory dataframe as a CSV download."""
    st.download_button(
        f"Download {name}.csv",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
        key=key or f"df_{name}",
    )


def array_download(array: np.ndarray, name: str) -> None:
    """Offer a NumPy array as a ``.npy`` download."""
    import io

    buffer = io.BytesIO()
    np.save(buffer, array)
    st.download_button(
        f"Download {name}.npy",
        data=buffer.getvalue(),
        file_name=f"{name}.npy",
        key=f"npy_{name}",
    )


def matrix_view(matrix: np.ndarray, title: str = "",
                labels: Optional[Sequence[str]] = None,
                fmt: str = "{:.3f}", cmap: str = "Reds",
                download_name: Optional[str] = None) -> None:
    """Render an ROI-by-ROI matrix as a styled table with a download."""
    labels = list(labels or [roi_short(r) for r in ROI_ORDER])
    if title:
        st.markdown(f"**{title}**")
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    st.dataframe(
        frame.style.background_gradient(cmap=cmap).format(fmt),
        use_container_width=True,
    )
    if download_name:
        dataframe_download(frame.reset_index(names="source"), download_name)


def roi_bar(values: Dict[str, float], title: str = "",
            value_label: str = "value", ascending: bool = False) -> None:
    """Render a per-ROI bar chart from a value dictionary."""
    if not values:
        st.caption("No values to display.")
        return
    frame = pd.DataFrame(
        {
            "ROI": [roi_short(r) for r in ROI_ORDER],
            value_label: [values.get(r, np.nan) for r in ROI_ORDER],
        }
    ).sort_values(value_label, ascending=ascending)
    if title:
        st.markdown(f"**{title}**")
    st.bar_chart(frame.set_index("ROI"), horizontal=True)


def slice_views(volume: np.ndarray, title: str = "",
                cmap: str = "gray") -> None:
    """Render axial, coronal and sagittal mid-slices of a 3-D volume."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        st.caption(f"Expected a 3-D volume; got shape {volume.shape}.")
        return
    if title:
        st.markdown(f"**{title}**")

    figure, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    planes = (
        ("Sagittal", volume[volume.shape[0] // 2, :, :]),
        ("Coronal", volume[:, volume.shape[1] // 2, :]),
        ("Axial", volume[:, :, volume.shape[2] // 2]),
    )
    for ax, (name, plane) in zip(axes, planes):
        ax.imshow(np.rot90(plane), cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def intensity_histogram(volume: np.ndarray, title: str = "") -> None:
    """Render an intensity histogram over non-background voxels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(volume, dtype=np.float32).ravel()
    values = values[np.isfinite(values) & (values > 1e-6)]
    if values.size == 0:
        st.caption("No non-background voxels to histogram.")
        return
    figure, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.hist(values, bins=80, color="#2f6feb", alpha=0.85)
    ax.set_xlabel("intensity", fontsize=8)
    ax.set_ylabel("voxels", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def formula(latex: str, explanation: str = "") -> None:
    """Render a formula with an optional plain-language explanation."""
    st.latex(latex)
    if explanation:
        st.markdown(f'<div class="ng-note">{explanation}</div>',
                    unsafe_allow_html=True)


def interpretation_note(text: str) -> None:
    """Render a mandatory interpretation caveat."""
    st.markdown(
        f'<div class="ng-banner ng-warn">{text}</div>', unsafe_allow_html=True
    )


def what_this_module_did(text: str) -> None:
    """Render the Section 39 requirement 7 explanation panel."""
    with st.expander("What this module did", expanded=False):
        st.markdown(text)


def stage_probability_row(probabilities: Dict[str, float],
                          predicted: Optional[str] = None) -> None:
    """Render CN/MCI/AD probabilities as a metric row."""
    columns = st.columns(len(STAGE_ORDER))
    for column, stage in zip(columns, STAGE_ORDER):
        value = probabilities.get(stage)
        column.metric(
            f"{stage}{'  (assigned)' if stage == predicted else ''}",
            "n/a" if value is None else f"{value:.4f}",
        )


def roi_selector(key: str, label: str = "Region") -> str:
    """Render an ROI selector and return the canonical ROI name."""
    short = st.radio(
        label, [roi_short(r) for r in ROI_ORDER], horizontal=True, key=key
    )
    for roi in ROI_ORDER:
        if roi_short(roi) == short:
            return roi
    return ROI_ORDER[0]


def roi_metadata_card(roi: str) -> None:
    """Render an ROI's anatomical metadata."""
    meta = ROI_METADATA.get(roi, {})
    st.markdown(
        f"""<div class="ng-card">
        <b>{meta.get('label', roi)}</b><br>
        <span class="ng-note">{meta.get('region', '')}</span><br>
        Function: {meta.get('function', '')}<br>
        Hemisphere: {meta.get('hemisphere', '')} &nbsp;|&nbsp;
        Brodmann: {meta.get('brodmann', '')}
        </div>""",
        unsafe_allow_html=True,
    )


__all__ = [
    "STATUS_STYLE",
    "inject_theme",
    "page_title",
    "provenance_banner",
    "unavailable",
    "status_pill",
    "module_header",
    "stage_detail",
    "download_button",
    "dataframe_download",
    "array_download",
    "matrix_view",
    "roi_bar",
    "slice_views",
    "intensity_histogram",
    "formula",
    "interpretation_note",
    "what_this_module_did",
    "stage_probability_row",
    "roi_selector",
    "roi_metadata_card",
]
