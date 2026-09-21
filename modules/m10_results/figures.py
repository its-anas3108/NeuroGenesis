"""
M10 — Research figures (Section 27).
===================================

Generates the sixteen figures the design requires, each from real computed
values. Three rules are enforced throughout:

**No figure is drawn from invented data.** Every function takes the actual
arrays or result objects. When the input is missing, the function draws an
explicit "Not available - requires training data" panel via
:func:`unavailable_panel` rather than plotting zeros or random values. A blank
labelled panel is honest; a plausible-looking curve from synthetic numbers is
not.

**Cross-sectional stage differences are never labelled as longitudinal
progression.** Section 27 is explicit about this. Titles and axis labels use
"stage-wise morphometric differences" and "stage-wise representation
differences". The words "progression", "trajectory", "forecast" and "over time"
do not appear in any figure this module produces, and
:func:`_check_title` enforces that at runtime.

**Smoke-test provenance is stamped on the figure.** When the output tree is
marked as synthetic, every figure carries a visible banner, so a figure cannot be
lifted into a slide deck without its provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import (
    N_ROI,
    ROI_ORDER,
    ROI_SHORT,
    STAGE_COLOR,
    STAGE_ORDER,
    roi_short,
)

logger = get_logger(__name__)

DPI = 200
FIG_BG = "#ffffff"
GRID = "#d9dde3"
TEXT = "#1b1f24"
MUTED = "#6a737d"
ACCENT = "#2f6feb"

#: Words that would misdescribe a cross-sectional figure as longitudinal.
_FORBIDDEN_TITLE_WORDS = (
    "progression", "trajectory", "forecast", "over time", "longitudinal",
    "future", "month",
)


def _check_title(title: str) -> str:
    """Reject figure titles that imply longitudinal measurement.

    Raises:
        ValueError: If the title contains a word from
            :data:`_FORBIDDEN_TITLE_WORDS`. This is a guard against a
            well-intentioned edit reintroducing "24-month progression" style
            labelling onto a cross-sectional figure.
    """
    lowered = title.lower()
    for word in _FORBIDDEN_TITLE_WORDS:
        if word in lowered:
            raise ValueError(
                f"Figure title {title!r} contains {word!r}, which implies "
                "longitudinal measurement. This study is cross-sectional; use "
                "'stage-wise' phrasing instead."
            )
    return title


def _style(ax: plt.Axes, title: str = "", xlabel: str = "",
           ylabel: str = "") -> None:
    """Apply the shared axis style."""
    if title:
        ax.set_title(_check_title(title), fontsize=11, color=TEXT, pad=10,
                     fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def unavailable_panel(ax: plt.Axes, title: str,
                      reason: str = "requires training data") -> None:
    """Draw an explicit unavailable panel instead of a fabricated plot."""
    ax.text(
        0.5, 0.56, "Not available", ha="center", va="center",
        fontsize=13, color=MUTED, fontweight="semibold", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.40, reason, ha="center", va="center", fontsize=9, color=MUTED,
        transform=ax.transAxes, wrap=True,
    )
    ax.set_title(_check_title(title), fontsize=11, color=TEXT, pad=10,
                 fontweight="semibold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linestyle((0, (4, 4)))


def _stamp(fig: Figure, smoke_marker: Optional[Dict[str, Any]]) -> None:
    """Stamp a synthetic-data banner across the figure when applicable."""
    if not smoke_marker:
        return
    fig.text(
        0.5, 0.985,
        "SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT",
        ha="center", va="top", fontsize=9, fontweight="bold",
        color="#b3261e",
        bbox=dict(facecolor="#fdecea", edgecolor="#b3261e", boxstyle="round,pad=0.35"),
    )


def _save(fig: Figure, out_dir: Path, name: str,
          smoke_marker: Optional[Dict[str, Any]] = None) -> Path:
    """Stamp, tighten and write a figure as PNG."""
    _stamp(fig, smoke_marker)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    top = 0.93 if smoke_marker else 0.97
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=DPI, facecolor=FIG_BG)
    plt.close(fig)
    logger.debug("Figure written: %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1-2. Training curves
# ──────────────────────────────────────────────────────────────────────────────

def training_curves(
    curves: Optional[Dict[str, List[Optional[float]]]],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figures 1 and 2: training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=FIG_BG)

    if not curves or not curves.get("epoch"):
        for ax, title in zip(axes, ("Loss curves", "Accuracy curves")):
            unavailable_panel(ax, title, "no training history recorded")
        return _save(fig, out_dir, "fig01_02_training_curves", smoke_marker)

    epochs = curves["epoch"]

    ax = axes[0]
    ax.plot(epochs, curves["train_loss"], color=ACCENT, linewidth=1.8,
            label="train")
    val_loss = [v for v in curves.get("val_loss", []) if v is not None]
    if val_loss:
        ax.plot(epochs, curves["val_loss"], color="#d29922", linewidth=1.8,
                linestyle="--", label="validation")
    _style(ax, "Training and validation loss", "epoch", "total loss")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(epochs, curves["train_accuracy"], color=ACCENT, linewidth=1.8,
            label="train accuracy")
    if any(v is not None for v in curves.get("val_accuracy", [])):
        ax.plot(epochs, curves["val_accuracy"], color="#d29922", linewidth=1.8,
                linestyle="--", label="validation accuracy")
    if any(v is not None for v in curves.get("val_balanced_accuracy", [])):
        ax.plot(epochs, curves["val_balanced_accuracy"], color="#3fb950",
                linewidth=1.5, linestyle=":", label="validation balanced acc.")
    _style(ax, "Training and validation accuracy", "epoch", "accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)

    return _save(fig, out_dir, "fig01_02_training_curves", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

def confusion_matrix_figure(
    matrix: Optional[np.ndarray],
    out_dir: Path,
    normalize: bool = True,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 3: CN/MCI/AD confusion matrix."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=FIG_BG)
    if matrix is None:
        unavailable_panel(ax, "CN / MCI / AD confusion matrix")
        return _save(fig, out_dir, "fig03_confusion_matrix", smoke_marker)

    matrix = np.asarray(matrix, dtype=np.float64)
    display = matrix.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        # A class with no test samples must stay blank, not become 0/0 -> nan
        # rendered as a colour.
        display = np.divide(display, row_sums, out=np.zeros_like(display),
                            where=row_sums > 0)

    image = ax.imshow(display, cmap="Blues", vmin=0,
                      vmax=1 if normalize else display.max())
    ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_yticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_xlabel("predicted stage", fontsize=9, color=MUTED)
    ax.set_ylabel("true stage", fontsize=9, color=MUTED)
    ax.set_title(_check_title("CN / MCI / AD confusion matrix"), fontsize=11,
                 color=TEXT, pad=10, fontweight="semibold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            label = f"{count}\n{display[i, j]:.0%}" if normalize else str(count)
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if display[i, j] > 0.55 else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.046,
                 label="row-normalised" if normalize else "count")
    return _save(fig, out_dir, "fig03_confusion_matrix", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ROC curves
# ──────────────────────────────────────────────────────────────────────────────

def roc_curves(
    y_true: Optional[Sequence[int]],
    y_prob: Optional[np.ndarray],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 4: one-vs-rest ROC curves per stage.

    Curves are computed directly from the score ranking rather than via
    scikit-learn, so a class absent from the labels yields no curve and is
    labelled as such instead of producing a diagonal that looks like a result.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.8), facecolor=FIG_BG)
    if y_true is None or y_prob is None:
        unavailable_panel(ax, "One-vs-rest ROC curves")
        return _save(fig, out_dir, "fig04_roc_curves", smoke_marker)

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64)
    drawn = 0

    for index, stage in enumerate(STAGE_ORDER):
        if index >= y_prob.shape[1]:
            continue
        positive = y_true == index
        if positive.sum() == 0 or (~positive).sum() == 0:
            continue
        scores = y_prob[:, index]
        order = np.argsort(-scores, kind="mergesort")
        hits = positive[order].astype(np.float64)
        tpr = np.concatenate([[0.0], np.cumsum(hits) / positive.sum()])
        fpr = np.concatenate(
            [[0.0], np.cumsum(1.0 - hits) / (~positive).sum()]
        )
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
            else float(np.trapz(tpr, fpr))
        ax.plot(fpr, tpr, linewidth=1.9, color=STAGE_COLOR[stage],
                label=f"{stage} (AUC {auc:.3f}, n={int(positive.sum())})")
        drawn += 1

    ax.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=1.0,
            label="chance")
    _style(ax, "One-vs-rest ROC curves", "false positive rate",
           "true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if drawn == 0:
        ax.text(0.5, 0.5, "No class had both positive and negative samples",
                ha="center", va="center", fontsize=9, color=MUTED,
                transform=ax.transAxes)
    return _save(fig, out_dir, "fig04_roc_curves", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 5-7. ROI maps and stage-wise differences
# ──────────────────────────────────────────────────────────────────────────────

def roi_value_map(
    values: Optional[Dict[str, float]],
    out_dir: Path,
    name: str,
    title: str,
    value_label: str = "value",
    cmap: str = "YlOrRd",
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figures 5 / 10: a per-ROI value map as a labelled horizontal bar chart.

    A schematic bar chart is used rather than a rendered brain overlay: the brain
    overlays require the actual MRI volumes and are produced by the preserved
    ``visualization`` package on the imaging path. This figure is the
    value-carrying summary that remains meaningful without imaging.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=FIG_BG)
    if not values:
        unavailable_panel(ax, title)
        return _save(fig, out_dir, name, smoke_marker)

    labels = [roi_short(r) for r in ROI_ORDER]
    series = [float(values.get(r, np.nan)) for r in ROI_ORDER]
    finite = [v for v in series if np.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    span = (hi - lo) or 1.0
    colours = plt.get_cmap(cmap)(
        [(v - lo) / span if np.isfinite(v) else 0.0 for v in series]
    )

    positions = np.arange(len(labels))
    ax.barh(positions, [v if np.isfinite(v) else 0.0 for v in series],
            color=colours, edgecolor=GRID, height=0.62)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    for y, v in zip(positions, series):
        text = f"{v:.3f}" if np.isfinite(v) else "n/a"
        ax.text(max(v if np.isfinite(v) else 0.0, 0) + 0.01 * span, y, text,
                va="center", fontsize=8, color=TEXT)
    _style(ax, title, value_label, "")
    return _save(fig, out_dir, name, smoke_marker)


def stagewise_feature_map(
    features: Optional[pd.DataFrame],
    cohort: Optional[pd.DataFrame],
    feature: str,
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 6: stage-wise morphometric differences for one feature.

    Deliberately titled "differences" and not "progression": each subject
    contributes exactly one scan, so the stage axis is a between-group contrast,
    never a within-subject change.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.0), facecolor=FIG_BG)
    if features is None or cohort is None or feature not in features.columns:
        unavailable_panel(
            ax, f"Stage-wise differences: {feature}",
            "requires an extracted feature table",
        )
        return _save(fig, out_dir, f"fig06_stagewise_{feature}", smoke_marker)

    stage_map = dict(zip(cohort["session_id"].astype(str),
                         cohort["stage"].astype(str)))
    table = features.copy()
    table["stage"] = table["session_id"].astype(str).map(stage_map)

    width = 0.24
    positions = np.arange(N_ROI)
    for offset, stage in enumerate(STAGE_ORDER):
        means, errors = [], []
        for roi in ROI_ORDER:
            values = pd.to_numeric(
                table.loc[(table["roi_name"] == roi) & (table["stage"] == stage),
                          feature],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
            errors.append(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        ax.bar(positions + (offset - 1) * width,
               [m if np.isfinite(m) else 0.0 for m in means],
               width=width, yerr=errors, capsize=2.5,
               color=STAGE_COLOR[stage], edgecolor=GRID, label=stage,
               error_kw={"linewidth": 0.9, "ecolor": MUTED})

    ax.set_xticks(positions, [roi_short(r) for r in ROI_ORDER])
    _style(ax, f"Stage-wise morphometric differences: {feature}",
           "speech-related ROI", feature)
    ax.legend(frameon=False, fontsize=8, title="stage",
              title_fontsize=8)
    return _save(fig, out_dir, f"fig06_stagewise_{feature}", smoke_marker)


def stagewise_importance_heatmap(
    rankings: Optional[Dict[str, Any]],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 7: stage-wise ROI importance heatmap."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4), facecolor=FIG_BG)
    if not rankings:
        unavailable_panel(ax, "Stage-wise ROI importance",
                          "requires a trained model and ROI ranking")
        return _save(fig, out_dir, "fig07_stagewise_roi_importance",
                     smoke_marker)

    stages = [s for s in STAGE_ORDER if s in rankings]
    matrix = np.full((len(stages), N_ROI), np.nan)
    for i, stage in enumerate(stages):
        scores = getattr(rankings[stage], "scores", None) or {}
        for j, roi in enumerate(ROI_ORDER):
            if roi in scores:
                matrix[i, j] = float(scores[roi])

    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(N_ROI), [roi_short(r) for r in ROI_ORDER])
    ax.set_yticks(range(len(stages)), stages)
    ax.set_title(_check_title("Stage-wise ROI importance"), fontsize=11,
                 color=TEXT, pad=10, fontweight="semibold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if matrix[i, j] < 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, label="combined importance")
    return _save(fig, out_dir, "fig07_stagewise_roi_importance", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 8-9. Graph visualisations
# ──────────────────────────────────────────────────────────────────────────────

def _node_layout() -> Dict[str, Tuple[float, float]]:
    """Fixed 2-D layout approximating left-hemisphere speech-network anatomy.

    Positions are held constant across every graph figure so that the anatomical
    and adaptive graphs can be compared directly. A force-directed layout would
    move nodes between figures and make visual comparison misleading.
    """
    return {
        "Inferior_Frontal_Gyrus": (-1.00, 0.55),
        "Broca_Area": (-0.45, 0.15),
        "Insula": (0.00, -0.35),
        "Wernicke_Area": (0.60, 0.20),
        "Superior_Temporal_Gyrus": (1.00, -0.35),
    }


def _draw_graph(ax: plt.Axes, adjacency: np.ndarray, title: str,
                threshold: float = 1e-6, max_width: float = 4.0) -> None:
    """Draw one directed weighted graph over the fixed anatomical layout."""
    layout = _node_layout()
    peak = float(np.nanmax(np.abs(adjacency))) or 1.0

    for i, source in enumerate(ROI_ORDER):
        for j, target in enumerate(ROI_ORDER):
            if i == j:
                continue
            weight = float(adjacency[i, j])
            if abs(weight) <= threshold:
                continue
            x0, y0 = layout[source]
            x1, y1 = layout[target]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=ACCENT,
                    alpha=float(np.clip(abs(weight) / peak, 0.12, 0.95)),
                    linewidth=max(0.4, max_width * abs(weight) / peak),
                    shrinkA=17, shrinkB=17,
                    connectionstyle="arc3,rad=0.13",
                ),
            )

    for roi in ROI_ORDER:
        x, y = layout[roi]
        ax.scatter([x], [y], s=1500, color="#f2f5f9", edgecolors=ACCENT,
                   linewidths=1.4, zorder=3)
        ax.text(x, y, ROI_SHORT[roi], ha="center", va="center", fontsize=8,
                color=TEXT, zorder=4, fontweight="semibold")

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.85, 1.05)
    ax.set_title(_check_title(title), fontsize=10, color=TEXT, pad=8,
                 fontweight="semibold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def graph_comparison(
    prior: Optional[np.ndarray],
    attention: Optional[np.ndarray],
    adaptive: Optional[np.ndarray],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figures 8 and 9: anatomical, learned-attention and adaptive graphs."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=FIG_BG)
    panels = (
        (prior, "Anatomical prior  A_prior"),
        (attention, "Learned attention  A_att"),
        (adaptive, "Adaptive graph  A* = alpha*A_prior + (1-alpha)*A_att"),
    )
    for ax, (matrix, title) in zip(axes, panels):
        if matrix is None:
            unavailable_panel(ax, title, "requires a trained model")
        else:
            _draw_graph(ax, np.asarray(matrix, dtype=np.float64), title)
    return _save(fig, out_dir, "fig08_09_graph_comparison", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 10-11. NeuroProp-X heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def matrix_heatmap(
    matrix: Optional[np.ndarray],
    out_dir: Path,
    name: str,
    title: str,
    cbar_label: str,
    cmap: str = "Reds",
    footnote: Optional[str] = None,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figures 10 / 11: an ROI-by-ROI matrix heatmap with cell annotations."""
    fig, ax = plt.subplots(figsize=(5.8, 5.0), facecolor=FIG_BG)
    if matrix is None:
        unavailable_panel(ax, title, "requires a trained model")
        return _save(fig, out_dir, name, smoke_marker)

    matrix = np.asarray(matrix, dtype=np.float64)
    image = ax.imshow(matrix, cmap=cmap)
    labels = [roi_short(r) for r in ROI_ORDER]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("target ROI", fontsize=9, color=MUTED)
    ax.set_ylabel("source ROI", fontsize=9, color=MUTED)
    ax.set_title(_check_title(title), fontsize=11, color=TEXT, pad=10,
                 fontweight="semibold")

    peak = float(np.nanmax(matrix)) or 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if matrix[i, j] > 0.6 * peak else TEXT)
    fig.colorbar(image, ax=ax, fraction=0.046, label=cbar_label)
    if footnote:
        fig.text(0.5, 0.015, footnote, ha="center", fontsize=7.5, color=MUTED,
                 wrap=True)
    return _save(fig, out_dir, name, smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Feature distributions
# ──────────────────────────────────────────────────────────────────────────────

def feature_violin(
    features: Optional[pd.DataFrame],
    cohort: Optional[pd.DataFrame],
    feature: str,
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 12: per-ROI, per-stage distribution of one feature."""
    fig, axes = plt.subplots(1, N_ROI, figsize=(15.0, 3.6), facecolor=FIG_BG,
                             sharey=True)
    if features is None or cohort is None or feature not in features.columns:
        for ax, roi in zip(axes, ROI_ORDER):
            unavailable_panel(ax, roi_short(roi), "no feature table")
        return _save(fig, out_dir, f"fig12_distribution_{feature}", smoke_marker)

    stage_map = dict(zip(cohort["session_id"].astype(str),
                         cohort["stage"].astype(str)))
    table = features.copy()
    table["stage"] = table["session_id"].astype(str).map(stage_map)

    for ax, roi in zip(axes, ROI_ORDER):
        groups, labels, colours = [], [], []
        for stage in STAGE_ORDER:
            values = pd.to_numeric(
                table.loc[(table["roi_name"] == roi) & (table["stage"] == stage),
                          feature],
                errors="coerce",
            ).dropna().values
            # A violin needs at least two distinct values; below that, draw the
            # points instead of silently omitting the group.
            if values.size >= 2 and float(np.std(values)) > 0:
                groups.append(values)
                labels.append(f"{stage}\nn={values.size}")
                colours.append(STAGE_COLOR[stage])
            elif values.size:
                ax.scatter([len(labels) + 1] * values.size, values, s=14,
                           color=STAGE_COLOR[stage], zorder=3)
                groups.append(np.array([values.mean(), values.mean()]))
                labels.append(f"{stage}\nn={values.size}")
                colours.append(STAGE_COLOR[stage])

        if groups:
            parts = ax.violinplot(groups, showmeans=True, showextrema=False)
            for body, colour in zip(parts["bodies"], colours):
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(GRID)
            if "cmeans" in parts:
                parts["cmeans"].set_color(TEXT)
                parts["cmeans"].set_linewidth(1.0)
            ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        _style(ax, roi_short(roi), "", feature if roi == ROI_ORDER[0] else "")

    fig.suptitle(
        _check_title(f"Stage-wise distribution of {feature} by ROI"),
        fontsize=11, color=TEXT, fontweight="semibold",
    )
    return _save(fig, out_dir, f"fig12_distribution_{feature}", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 17. Morphometric differences (Figures & Visual Analytics)
# ──────────────────────────────────────────────────────────────────────────────

#: (feature column, panel title, y-axis label, unit conversion applied to the
#: raw feature values before plotting). Four real, already-computed features
#: from modules.m04_feature_extraction.feature_spec — no new feature is
#: derived here. Surface area is deliberately left in voxel units rather than
#: converted to a physical area: post-resample voxel spacing is anisotropic
#: per axis, so a scalar voxel->cm^2 conversion would need a real per-face
#: marching-cubes correction the pipeline does not compute. Converting
#: without it would silently invent a number, so the axis is labelled
#: honestly instead.
_MORPHOMETRIC_PANELS: Tuple[Tuple[str, str, str, float], ...] = (
    ("cortical_thickness_mm", "Cortical Thickness", "mm", 1.0),
    ("brain_volume_mm3", "Regional Volume", "cm$^3$", 1.0 / 1000.0),
    ("surface_area_vox", "Surface Area", "voxels", 1.0),
    ("gm_fraction", "GM Fraction", "fraction", 1.0),
)


def _boxplot_panel(ax: plt.Axes, table: pd.DataFrame, feature: str,
                   scale: float, title: str, ylabel: str) -> None:
    """Draw one grouped (ROI x stage) boxplot panel onto ``ax``."""
    width = 0.24
    positions = np.arange(N_ROI)
    for offset, stage in enumerate(STAGE_ORDER):
        group_positions, group_data = [], []
        for i, roi in enumerate(ROI_ORDER):
            values = pd.to_numeric(
                table.loc[(table["roi_name"] == roi) & (table["stage"] == stage),
                          feature],
                errors="coerce",
            ).dropna().values * scale
            if values.size:
                group_positions.append(positions[i] + (offset - 1) * width)
                group_data.append(values)
        if not group_data:
            continue
        bp = ax.boxplot(
            group_data, positions=group_positions, widths=width * 0.85,
            patch_artist=True, showfliers=False,
            medianprops={"color": TEXT, "linewidth": 1.1},
            whiskerprops={"color": MUTED}, capprops={"color": MUTED},
        )
        for box in bp["boxes"]:
            box.set_facecolor(STAGE_COLOR[stage])
            box.set_alpha(0.6)
            box.set_edgecolor(GRID)

    ax.set_xticks(positions, [roi_short(r) for r in ROI_ORDER])
    ax.set_xlim(-0.6, N_ROI - 0.4)
    _style(ax, title, "", ylabel)


def morphometric_group_comparison(
    features: Optional[pd.DataFrame],
    cohort: Optional[pd.DataFrame],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 17: morphometric differences across CN/MCI/AD, four features at once.

    Combines four already-computed per-ROI features (cortical thickness,
    regional volume, surface area, grey-matter fraction) into one figure, one
    boxplot panel per feature, each grouped by ROI and stage. Reuses the exact
    ``features``/``cohort`` join :func:`stagewise_feature_map` and
    :func:`feature_violin` already use; this figure differs from those only in
    combining four features side by side as boxplots, matching how the
    stage-wise contrast is presented alongside the single-feature bar/violin
    figures already produced by ``--mode figures``.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), facecolor=FIG_BG)
    if features is None or cohort is None:
        for ax, (_, title, unit, _) in zip(axes, _MORPHOMETRIC_PANELS):
            unavailable_panel(ax, title, "requires an extracted feature table")
        return _save(fig, out_dir, "fig17_morphometric_differences", smoke_marker)

    stage_map = dict(zip(cohort["session_id"].astype(str),
                         cohort["stage"].astype(str)))
    table = features.copy()
    table["stage"] = table["session_id"].astype(str).map(stage_map)

    for ax, (feature, title, unit, scale) in zip(axes, _MORPHOMETRIC_PANELS):
        if feature not in table.columns:
            unavailable_panel(ax, title, f"'{feature}' not in feature table")
            continue
        _boxplot_panel(ax, table, feature, scale,
                      f"{title}\n({unit})", unit)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STAGE_COLOR[s], alpha=0.6,
                      edgecolor=GRID)
        for s in STAGE_ORDER
    ]
    fig.legend(handles, STAGE_ORDER, loc="upper right", ncol=1,
              frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.96))
    fig.suptitle(
        _check_title("Morphometric differences across CN / MCI / AD"),
        fontsize=12, color=TEXT, fontweight="semibold",
    )
    return _save(fig, out_dir, "fig17_morphometric_differences", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 13. Embedding projection
# ──────────────────────────────────────────────────────────────────────────────

def _pca_scatter(ax: plt.Axes, embeddings: Optional[np.ndarray],
                 labels: Optional[Sequence[int]], title: str) -> None:
    """Project ``embeddings`` to 2-D via SVD and scatter onto ``ax``, coloured
    by stage. Shared by :func:`embedding_projection` and
    :func:`embedding_projection_pair` so both draw from one PCA
    implementation.

    PCA is computed directly from the SVD rather than via UMAP or t-SNE. Those
    methods have stochastic, hyper-parameter-sensitive layouts that invite
    over-reading of apparent cluster separation; PCA is a deterministic linear
    projection whose axes carry a stated explained-variance fraction.
    """
    if embeddings is None or labels is None:
        unavailable_panel(ax, title, "requires a trained model")
        return

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    labels_arr = np.asarray(labels, dtype=np.int64).ravel()

    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        unavailable_panel(ax, title,
                          f"too few samples to project ({matrix.shape[0]})")
        return

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ components[:2].T
    variance = singular ** 2
    explained = variance / variance.sum() if variance.sum() > 0 else variance

    for index, stage in enumerate(STAGE_ORDER):
        mask = labels_arr == index
        if not mask.any():
            continue
        ax.scatter(projected[mask, 0], projected[mask, 1], s=34,
                   color=STAGE_COLOR[stage], edgecolors="white", linewidths=0.6,
                   label=f"{stage} (n={int(mask.sum())})", alpha=0.9)

    _style(ax, title,
           f"PC1 ({explained[0]:.1%} variance)",
           f"PC2 ({explained[1]:.1%} variance)")
    ax.legend(frameon=False, fontsize=8)


def embedding_projection(
    embeddings: Optional[np.ndarray],
    labels: Optional[Sequence[int]],
    out_dir: Path,
    name: str = "fig13_embedding_projection",
    title: str = "Representation projection (PCA)",
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 13: 2-D PCA projection of a representation, coloured by stage."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=FIG_BG)
    _pca_scatter(ax, embeddings, labels, title)
    return _save(fig, out_dir, name, smoke_marker)


def embedding_projection_pair(
    z_g: Optional[np.ndarray],
    z_h: Optional[np.ndarray],
    labels: Optional[Sequence[int]],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 19: PCA projections of the graph embedding Z_G and the shared
    representation Z_H, side by side, coloured by real CN/MCI/AD labels.

    Two panels only, deliberately. A faculty reference figure this project
    was asked to reproduce also shows a third "Spatial Embedding (Z_3D)"
    panel — the output of the 3D-CNN branch. That branch was removed from
    this architecture entirely (a prior, explicit decision), so Z_3D does not
    exist as a real quantity anymore; fabricating a third panel to match the
    reference layout was rejected in favour of showing exactly the two
    representations this architecture actually computes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), facecolor=FIG_BG)
    _pca_scatter(axes[0], z_g, labels, "Graph Embedding (Z_G)")
    _pca_scatter(axes[1], z_h, labels, "Fused Embedding (Z_H)")
    fig.suptitle(
        _check_title("Embedding visualisation (PCA)"),
        fontsize=12, color=TEXT, fontweight="semibold",
    )
    fig.text(
        0.5, 0.005,
        "No third 'Spatial Embedding' panel: this architecture has no 3D-CNN "
        "/ spatial branch.",
        ha="center", fontsize=7.5, color=MUTED,
    )
    return _save(fig, out_dir, "fig19_embedding_projection", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 18. SRVE regional vulnerability maps
# ──────────────────────────────────────────────────────────────────────────────

def srve_vulnerability_maps(
    background: Optional[np.ndarray],
    roi_masks: Optional[Dict[str, np.ndarray]],
    stage_vulnerability: Optional[Dict[str, Dict[str, Optional[float]]]],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 18: SRVE regional vulnerability, averaged by stage, on real
    anatomy.

    Args:
        background: A 2-D grayscale slice (standard/atlas space), shared
            across all three panels.
        roi_masks: ``{roi_name: 2-D boolean mask}`` in the same grid as
            ``background`` (extracted once from a template, not from any
            individual subject).
        stage_vulnerability: ``{stage: {roi_name: mean_RV_or_None}}`` — real
            per-subject SRVE regional-vulnerability scores (Section 11.1),
            averaged over every subject with that real stage label in this
            cohort. ``None`` for a cell means no subject in that stage had a
            score for that ROI, not zero vulnerability.

    Region location is standard-space and shared; only the fill colour (a
    real, computed group average) differs between the three panels. This is
    not a claim that vulnerability is spatially localised beyond the ROI's
    extent — it is the same five discrete regional scores shown elsewhere in
    this report, placed on real anatomy instead of a bar chart.
    """
    # A dedicated 4th subplot hosts the colorbar (``width_ratios`` makes it
    # narrow) rather than the usual `fig.colorbar(sm, ax=list(axes))`, which
    # steals space from the three panels *after* layout and is not
    # compatible with the `tight_layout` call `_save` always makes — that
    # combination previously produced a colorbar that visually overlapped
    # the AD panel.
    fig, all_axes = plt.subplots(
        1, 4, figsize=(14.5, 4.8), facecolor=FIG_BG,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.06]},
    )
    axes, cax = all_axes[:3], all_axes[3]
    if background is None or roi_masks is None or stage_vulnerability is None:
        cax.set_axis_off()
        for ax, stage in zip(axes, STAGE_ORDER):
            unavailable_panel(
                ax, stage,
                "requires standard-space ROI masks and SRVE vulnerability",
            )
        return _save(fig, out_dir, "fig18_srve_vulnerability_maps", smoke_marker)

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for ax, stage in zip(axes, STAGE_ORDER):
        ax.imshow(background, cmap="gray", origin="lower")
        scores = stage_vulnerability.get(stage, {})
        overlay = np.zeros((*background.shape, 4), dtype=np.float64)
        any_score = False
        for roi in ROI_ORDER:
            mask = roi_masks.get(roi)
            score = scores.get(roi)
            if mask is None or score is None or not mask.any():
                continue
            any_score = True
            color = cmap(norm(score))
            overlay[mask] = (color[0], color[1], color[2], 0.65)
        if any_score:
            ax.imshow(overlay, origin="lower")
        ax.set_title(_check_title(stage), fontsize=12, color=TEXT, pad=8,
                     fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="mean regional vulnerability (RV)")
    fig.suptitle(
        _check_title("SRVE regional vulnerability maps, averaged by stage"),
        fontsize=12, color=TEXT, fontweight="semibold",
    )
    return _save(fig, out_dir, "fig18_srve_vulnerability_maps", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 14. ROI ranking stability
# ──────────────────────────────────────────────────────────────────────────────

def ranking_stability_plot(
    stability_rows: Optional[List[Dict[str, Any]]],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 14: mean rank with SD, plus selection frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), facecolor=FIG_BG)
    if not stability_rows:
        for ax, title in zip(axes, ("ROI mean rank", "ROI selection frequency")):
            unavailable_panel(ax, title, "requires repeated ROI rankings")
        return _save(fig, out_dir, "fig14_ranking_stability", smoke_marker)

    rows = [r for r in stability_rows if r.get("mean_rank") is not None]
    if not rows:
        for ax, title in zip(axes, ("ROI mean rank", "ROI selection frequency")):
            unavailable_panel(ax, title, "no ROI received a rank")
        return _save(fig, out_dir, "fig14_ranking_stability", smoke_marker)

    labels = [r["roi_short"] for r in rows]
    positions = np.arange(len(rows))

    ax = axes[0]
    ax.errorbar(
        [r["mean_rank"] for r in rows], positions,
        xerr=[r.get("rank_sd") or 0.0 for r in rows],
        fmt="o", color=ACCENT, ecolor=MUTED, capsize=3, markersize=7,
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.5, N_ROI + 0.5)
    _style(ax, "ROI mean rank (lower is more important)",
           "mean rank +/- SD across subjects", "")

    ax = axes[1]
    ax.barh(positions, [r["selection_frequency"] for r in rows],
            color="#3fb950", edgecolor=GRID, height=0.6)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    for y, r in zip(positions, rows):
        ax.text(r["selection_frequency"] + 0.015, y,
                f"{r['selection_frequency']:.0%}", va="center", fontsize=8,
                color=TEXT)
    _style(ax, "Top-k selection frequency", "fraction of repeats", "")
    return _save(fig, out_dir, "fig14_ranking_stability", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 15. Ablation comparison
# ──────────────────────────────────────────────────────────────────────────────

def ablation_plot(
    summaries: Optional[Dict[str, Any]],
    out_dir: Path,
    metric: str = "macro_f1",
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 15: ablation ladder with error bars."""
    fig, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=FIG_BG)
    if not summaries:
        unavailable_panel(ax, "Ablation comparison",
                          "requires a completed ablation study")
        return _save(fig, out_dir, "fig15_ablation", smoke_marker)

    order = [k for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A7")
             if k in summaries]
    means, errors, labels = [], [], []
    for key in order:
        summary = summaries[key]
        mean = summary.value(metric, "mean")
        if mean is None:
            continue
        means.append(mean)
        errors.append(summary.value(metric, "sd") or 0.0)
        labels.append(key)

    if not means:
        unavailable_panel(ax, "Ablation comparison",
                          f"metric {metric} was undefined for every variant")
        return _save(fig, out_dir, "fig15_ablation", smoke_marker)

    positions = np.arange(len(labels))
    colours = ["#8b949e"] * len(labels)
    if labels and labels[-1] == "A7":
        colours[-1] = ACCENT
    ax.bar(positions, means, yerr=errors, capsize=3.5, color=colours,
           edgecolor=GRID, width=0.62,
           error_kw={"linewidth": 0.9, "ecolor": MUTED})
    ax.set_xticks(positions, labels)
    for x, (mean, err) in enumerate(zip(means, errors)):
        ax.text(x, mean + err + 0.015, f"{mean:.3f}", ha="center", fontsize=8,
                color=TEXT)
    _style(ax, f"Ablation comparison ({metric}, mean +/- SD over repeats)",
           "configuration", metric)
    ax.set_ylim(0, min(1.05, max(m + e for m, e in zip(means, errors)) + 0.12))
    return _save(fig, out_dir, "fig15_ablation", smoke_marker)


# ──────────────────────────────────────────────────────────────────────────────
# 16. Stage-transition propensity
# ──────────────────────────────────────────────────────────────────────────────

def propensity_figure(
    predictions: Optional[pd.DataFrame],
    out_dir: Path,
    smoke_marker: Optional[Dict[str, Any]] = None,
) -> Path:
    """Figure 16: stage-propensity distributions by true stage."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), facecolor=FIG_BG)
    columns = ("ad_associated_propensity", "advanced_stage_alignment")
    titles = ("AD-associated propensity by stage",
              "Advanced-stage alignment by stage")

    if predictions is None or predictions.empty:
        for ax, title in zip(axes, titles):
            unavailable_panel(ax, title, "requires a trained Stage-TGT branch")
        return _save(fig, out_dir, "fig16_stage_propensity", smoke_marker)

    for ax, column, title in zip(axes, columns, titles):
        if column not in predictions.columns:
            unavailable_panel(ax, title, f"column {column} is absent")
            continue
        groups, labels, colours = [], [], []
        for stage in STAGE_ORDER:
            values = pd.to_numeric(
                predictions.loc[predictions["true_stage"] == stage, column],
                errors="coerce",
            ).dropna().values
            if values.size:
                groups.append(values)
                labels.append(f"{stage}\nn={values.size}")
                colours.append(STAGE_COLOR[stage])
        if not groups:
            unavailable_panel(ax, title, "no values available")
            continue
        for x, (values, colour) in enumerate(zip(groups, colours), start=1):
            jitter = (np.random.default_rng(x).random(values.size) - 0.5) * 0.18
            ax.scatter(np.full(values.size, x) + jitter, values, s=26,
                       color=colour, alpha=0.75, edgecolors="white",
                       linewidths=0.5)
            ax.hlines(values.mean(), x - 0.24, x + 0.24, color=TEXT,
                      linewidth=1.6)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        _style(ax, title, "true stage", column.replace("_", " "))

    fig.text(
        0.5, 0.015,
        "Model-derived stage propensity from a single cross-sectional scan. Not "
        "a validated conversion probability and not a prediction of future "
        "diagnosis.",
        ha="center", fontsize=7.5, color=MUTED, wrap=True,
    )
    return _save(fig, out_dir, "fig16_stage_propensity", smoke_marker)


__all__ = [
    "DPI",
    "unavailable_panel",
    "training_curves",
    "confusion_matrix_figure",
    "roc_curves",
    "roi_value_map",
    "stagewise_feature_map",
    "stagewise_importance_heatmap",
    "graph_comparison",
    "matrix_heatmap",
    "feature_violin",
    "embedding_projection",
    "ranking_stability_plot",
    "ablation_plot",
    "propensity_figure",
]
