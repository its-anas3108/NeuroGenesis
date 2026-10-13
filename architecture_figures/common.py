"""Shared plotting helpers and real-data loaders for the NeuroGenesis
architecture figure set (Figures 1-8). Every loader reads actual on-disk
project outputs for subject OAS1_0001_MR1 (real cohort stage = CN) from
outputs_oasis1_pilot31/ — nothing here is synthetic or fabricated. Where an
artifact genuinely does not exist, callers must render `missing_panel`
labelled "ACTUAL OUTPUT NOT FOUND" rather than inventing a value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
OASIS_OUT = ROOT / "outputs_oasis1_pilot31"
SID = "OAS1_0001_MR1"

DPI = 300
TEXT = "#1b1f24"
MUTED = "#5b6570"
BORDER = "#9aa4b2"

COLOR = {
    "input": "#eef1f6", "preproc": "#dbe9fb", "roi": "#dff3e3",
    "morph": "#fdeecb", "npx": "#f3ddf7", "gat": "#ffe1e1",
    "fusion": "#e1f0ff", "out": "#fff3cf", "xai": "#e4e1fb",
}

ROI_FULL = ["Broca_Area", "Wernicke_Area", "Insula",
           "Inferior_Frontal_Gyrus", "Superior_Temporal_Gyrus"]
ROI_SHORT = ["Broca", "Wernicke", "Insula", "IFG", "STG"]

FEATURE_COLS = [
    "voxel_count", "brain_volume_mm3", "gm_volume_mm3", "mean_intensity",
    "std_intensity", "skewness", "kurtosis", "entropy", "surface_area_vox",
    "cortical_thickness_mm", "gm_fraction", "normalized_volume",
    "surface_to_volume_ratio", "atrophy_index",
]


def box(fig, xywh, text, color, fontsize=7.6, weight="semibold"):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.06), 0.96, 0.88, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1, edgecolor=BORDER, facecolor=color,
    ))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize,
            color=TEXT, fontweight=weight, wrap=True)
    return ax


def arrow(fig, p0, p1, color="#4a5563", lw=1.5):
    ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, transform=fig.transFigure, zorder=5,
    ))
    ax.patch.set_alpha(0)


def missing_panel(fig, xywh, title: str, reason: str):
    ax = fig.add_axes(xywh)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(BORDER); s.set_linewidth(0.7)
    ax.set_facecolor("#f7f7f8")
    ax.text(0.5, 0.55, "ACTUAL OUTPUT\nNOT FOUND", ha="center", va="center",
            transform=ax.transAxes, fontsize=6.5, color="#b3261e", fontweight="bold")
    ax.text(0.5, 0.22, reason, ha="center", va="center",
            transform=ax.transAxes, fontsize=5.2, color=MUTED, wrap=True)
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    return ax


def img_panel(fig, xywh, kind: str, path: Optional[Path], title: str,
             reason: str = "", cmap="gray"):
    if path is None or not Path(path).exists():
        return missing_panel(fig, xywh, title, reason)
    ax = fig.add_axes(xywh)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(BORDER); s.set_linewidth(0.7)
    if kind == "nifti":
        data = np.asarray(nib.load(str(path)).get_fdata())
        sl = np.rot90(np.take(data, data.shape[2] // 2, axis=2))
        ax.imshow(sl, cmap=cmap)
    elif kind == "png":
        ax.imshow(mpimg.imread(str(path)))
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    return ax


def heatmap_panel(fig, xywh, matrix, labels, title, cmap="viridis",
                  fmt="{:.2f}", fontsize=5.0, cbar=False):
    ax = fig.add_axes(xywh)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=4.6, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=4.6)
    if fmt:
        vmax = np.nanmax(matrix) if np.isfinite(np.nanmax(matrix)) else 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=fontsize,
                        color="white" if val > vmax * 0.55 else TEXT)
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=4.5)
    return ax


def bar_panel(fig, xywh, labels, values, title, color="#2f6feb", fmt="{:.2f}",
             horizontal=True):
    ax = fig.add_axes(xywh)
    if horizontal:
        ax.barh(range(len(labels)), values, color=color, edgecolor="none")
        ax.set_yticks(range(len(labels)), labels, fontsize=5.2)
        ax.invert_yaxis()
        vmax = max(values) if max(values) > 0 else 1
        ax.set_xlim(0, vmax * 1.35)
        for i, v in enumerate(values):
            ax.text(v + vmax * 0.03, i, fmt.format(v), va="center", fontsize=4.8)
    else:
        ax.bar(range(len(labels)), values, color=color, edgecolor="none")
        ax.set_xticks(range(len(labels)), labels, fontsize=5.2)
        vmax = max(values) if max(values) > 0 else 1
        ax.set_ylim(0, vmax * 1.35)
        for i, v in enumerate(values):
            ax.text(i, v + vmax * 0.03, fmt.format(v), ha="center", fontsize=4.8)
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    ax.tick_params(labelsize=4.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def text_panel(fig, xywh, title, lines, fontsize=6.0, color=COLOR["out"]):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.0, edgecolor=BORDER, facecolor=color,
    ))
    n_rows = sum(max(1, line.count("\n") + 1) for line in lines) if lines else 0
    n_rows += 1 if title else 0
    avail = 0.90 - 0.04
    step = min(0.085, avail / max(n_rows, 1))
    if title:
        ax.text(0.04, 0.90, title, ha="left", va="top", fontsize=fontsize + 0.6,
                color=TEXT, fontweight="bold")
    y = 0.90 - (step + 0.02 if title else 0.03)
    for line in lines:
        ax.text(0.04, y, line, ha="left", va="top", fontsize=fontsize, color=TEXT)
        y -= step * max(1, line.count("\n") + 1)
    return ax


def footer(fig, extra: str = ""):
    fig.text(
        0.5, 0.006,
        f"Subject: {SID} (real cohort stage = CN)  |  Source: outputs_oasis1_pilot31/  "
        f"|  All values real, none fabricated; unavailable artifacts are labelled "
        f"explicitly.{'  |  ' + extra if extra else ''}",
        ha="center", fontsize=6.6, color=MUTED,
    )


def save(fig, name: str):
    path = OUT / name
    plt.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


# ─────────────────────────── real-data loaders ────────────────────────────

def load_record():
    records = json.loads(
        (OASIS_OUT / "roi_ranking" / "per_subject_records.json").read_text()
    )
    return next(r for r in records if r["subject_id"] == SID)


def load_morph_matrix():
    features = pd.read_csv(OASIS_OUT / "features" / "morphometric_features.csv")
    sub = features[features["session_id"] == SID]
    return sub.set_index("roi_name").loc[ROI_FULL, FEATURE_COLS]


def load_npx():
    d = OASIS_OUT / "neuropropx"
    rv = np.load(d / "vulnerability" / SID / f"{SID}_regional_vulnerability.npy")
    a_prior = np.load(d / "adaptive_adjacency" / SID / f"{SID}_A_prior.npy") \
        if (d / "adaptive_adjacency" / SID / f"{SID}_A_prior.npy").exists() else None
    a_att = np.load(d / "adaptive_adjacency" / SID / f"{SID}_A_att.npy") \
        if (d / "adaptive_adjacency" / SID / f"{SID}_A_att.npy").exists() else None
    a_star = np.load(d / "adaptive_adjacency" / SID / f"{SID}_A_star.npy")
    p_mat = np.load(d / "propagation" / SID / f"{SID}_propagation_score.npy")
    eg = d / "enriched_graph" / SID
    x_star = np.load(eg / f"{SID}_X_star.npy") if (eg / f"{SID}_X_star.npy").exists() else None
    edge_features = np.load(eg / f"{SID}_edge_features.npy") \
        if (eg / f"{SID}_edge_features.npy").exists() else None
    g_star = json.loads((eg / f"{SID}_G_star.json").read_text()) \
        if (eg / f"{SID}_G_star.json").exists() else None
    return dict(rv=rv, a_prior=a_prior, a_att=a_att, a_star=a_star, p=p_mat,
               x_star=x_star, edge_features=edge_features, g_star=g_star)
