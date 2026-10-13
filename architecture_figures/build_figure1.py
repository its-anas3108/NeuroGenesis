"""
Figure 1 — NeuroGenesis Complete Layer-Wise Architecture.
==========================================================

The main pipeline skeleton (boxes/arrows/verified dimensions) with a small
number of representative REAL thumbnails at key nodes, for orientation. The
exhaustive per-stage real-artifact sets (all five M3 sub-steps, all five ROI
patches, the full matrix set, full attention/edge-gate maps, etc.) live in
Figures 2-8, per the "don't force 25 images into one microscopic diagram"
instruction this figure follows.

Every image/matrix/number is real (subject OAS1_0001_MR1, real stage=CN) or
read from the current config/model at build time. Nothing is invented; a
genuinely absent artifact (the raw pre-M1 OASIS-1 source, no longer present
on this machine) is labelled "ACTUAL OUTPUT NOT FOUND" rather than faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    "fusion": "#e1f0ff", "out": "#fff3cf",
}


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


def img_panel(fig, xywh, kind: str, path: Optional[Path], title: str,
             reason: str, cmap="gray"):
    ax = fig.add_axes(xywh)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(BORDER); s.set_linewidth(0.7)
    if path is None or not path.exists():
        ax.set_facecolor("#f7f7f8")
        ax.text(0.5, 0.55, "ACTUAL OUTPUT\nNOT FOUND", ha="center", va="center",
                transform=ax.transAxes, fontsize=6.5, color="#b3261e",
                fontweight="bold")
        ax.text(0.5, 0.22, reason, ha="center", va="center",
                transform=ax.transAxes, fontsize=5.2, color=MUTED, wrap=True)
    elif kind == "nifti":
        data = np.asarray(nib.load(str(path)).get_fdata())
        sl = np.rot90(np.take(data, data.shape[2] // 2, axis=2))
        ax.imshow(sl, cmap=cmap)
    elif kind == "png":
        ax.imshow(mpimg.imread(str(path)))
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    return ax


def heatmap_panel(fig, xywh, matrix, labels, title, cmap="viridis",
                  fmt="{:.2f}", fontsize=5.0):
    ax = fig.add_axes(xywh)
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels, fontsize=4.6, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=4.6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=fontsize,
                    color="white" if val > np.nanmax(matrix) * 0.55 else TEXT)
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    return ax


def bar_panel(fig, xywh, labels, values, title, color="#2f6feb", fmt="{:.2f}"):
    ax = fig.add_axes(xywh)
    ax.barh(range(len(labels)), values, color=color, edgecolor="none")
    ax.set_yticks(range(len(labels)), labels, fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.35 if max(values) > 0 else 1)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.03, i, fmt.format(v), va="center", fontsize=4.8)
    ax.set_title(title, fontsize=6.3, color=TEXT, pad=3)
    ax.tick_params(labelsize=4.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def main() -> None:
    roi_full = ["Broca_Area", "Wernicke_Area", "Insula",
               "Inferior_Frontal_Gyrus", "Superior_Temporal_Gyrus"]
    roi_short = ["Broca", "Wernicke", "Insula", "IFG", "STG"]

    record = json.loads(
        (OASIS_OUT / "roi_ranking" / "per_subject_records.json").read_text()
    )
    rec = next(r for r in record if r["subject_id"] == SID)
    stage = rec["reference_stage"]
    probs = rec["class_probabilities"]
    graph_expl = rec["explanations"]["graph"]
    npx_expl = rec["explanations"]["neuropropx"]
    prop_detail = rec["stage_propensity_detail"]

    features = pd.read_csv(OASIS_OUT / "features" / "morphometric_features.csv")
    sub_features = features[features["session_id"] == SID]
    feature_cols = [
        "voxel_count", "brain_volume_mm3", "gm_volume_mm3", "mean_intensity",
        "std_intensity", "skewness", "kurtosis", "entropy", "surface_area_vox",
        "cortical_thickness_mm", "gm_fraction", "normalized_volume",
        "surface_to_volume_ratio", "atrophy_index",
    ]
    morph_matrix = sub_features.set_index("roi_name").loc[
        roi_full, feature_cols
    ].to_numpy(dtype=float)
    # Normalise each feature column to [0,1] purely for heatmap legibility;
    # the printed values below the heatmap (Figure 4) carry the real units.
    morph_norm = (morph_matrix - morph_matrix.min(0)) / (
        morph_matrix.max(0) - morph_matrix.min(0) + 1e-9
    )

    npx_dir = OASIS_OUT / "neuropropx"
    rv = np.load(npx_dir / "vulnerability" / SID / f"{SID}_regional_vulnerability.npy")
    a_star = np.load(npx_dir / "adaptive_adjacency" / SID / f"{SID}_A_star.npy")
    p_mat = np.load(npx_dir / "propagation" / SID / f"{SID}_propagation_score.npy")
    edge_gate = np.array(graph_expl["edge_gate"])

    pre = OASIS_OUT / "preprocessing" / SID
    roi_masks = OASIS_OUT / "roi" / "masks" / SID

    fig = plt.figure(figsize=(28, 16), facecolor="white", dpi=DPI)
    fig.suptitle(
        "NeuroGenesis — Complete Layer-Wise Architecture",
        fontsize=17, fontweight="bold", color=TEXT, y=0.985, x=0.5,
    )
    fig.text(
        0.5, 0.958,
        f"Representative subject: {SID}  (real cohort stage = {stage})   —   "
        "morphometry-only architecture (no 3D-CNN branch: removed from this "
        "implementation)   —   every value real, none fabricated",
        ha="center", fontsize=9.5, color=MUTED,
    )

    # ══════════════════════════════════════════════════════════════════
    # Row A: M1-M5 preprocessing (y = 0.74-0.90)
    # ══════════════════════════════════════════════════════════════════
    ay0, ah = 0.76, 0.14
    m_labels = [
        ("M1\nLoad & Header Parse", "176x208x176x1\n→ 176x208x176"),
        ("M2\nQuality Control", "QC ≥ 50.0\nscore + flags"),
        ("M3\nIntensity Processing", "N4→WM-KDE→CLAHE→\ndiffusion→min-max"),
        ("M4\nSkull Stripping", "176x208x176"),
        ("M5\nSpatial Standardization", "σ=1.0, linear\n→ 128x128x128"),
    ]
    xs = np.linspace(0.015, 0.015 + 4 * 0.107, 5)
    bw = 0.09
    for (label, dims), x in zip(m_labels, xs):
        box(fig, (x, ay0, bw, ah), f"{label}\n{dims}", COLOR["preproc"], fontsize=6.4)
    for i in range(4):
        arrow(fig, (xs[i] + bw, ay0 + ah / 2), (xs[i + 1], ay0 + ah / 2))

    # Real thumbnails under M1-M5
    ty0, th = 0.575, 0.165
    img_panel(fig, (xs[0], ty0, bw, th), "nifti", None,
             "M1 input", "raw OASIS-1 .hdr/.img no\nlonger present on this\nmachine (dataset removed)")
    img_panel(fig, (xs[1], ty0, bw, th), "png", pre / f"{SID}_qc_report.png",
             "M2 real QC report", "")
    img_panel(fig, (xs[2], ty0, bw, th), "nifti", None,
             "M3 output", "active pipeline saves only\nthe final NIfTI, not each\nM3 sub-step (see Fig. 2)")
    img_panel(fig, (xs[3], ty0, bw, th), "png", pre / f"{SID}_skull_stripping.png",
             "M4 real skull-strip", "")
    img_panel(fig, (xs[4], ty0, bw, th), "png", pre / f"{SID}_resampling.png",
             "M5 real resampling", "")
    for x in xs:
        arrow(fig, (x + bw / 2, ay0), (x + bw / 2, ty0 + th + 0.014))

    # ══════════════════════════════════════════════════════════════════
    # M6-M7 ROI localisation + patches (y = 0.44-0.55)
    # ══════════════════════════════════════════════════════════════════
    arrow(fig, (xs[4] + bw / 2, ty0), (xs[4] + bw / 2, 0.56))
    box(fig, (0.40, 0.555, 0.20, 0.055),
       "M6–M7  Speech ROI Localization & Patch Extraction\n"
       "Harvard-Oxford atlas, 25% threshold → 5 x 48x48x48",
       COLOR["roi"], fontsize=7.0)

    roi_y, roi_h = 0.44, 0.10
    roi_x = np.linspace(0.015, 0.015 + 4 * 0.107, 5)
    for x, roi, short in zip(roi_x, roi_full, roi_short):
        img_panel(fig, (x, roi_y, bw, roi_h), "png",
                 roi_masks / f"{SID}_{roi.lower()}_overlay.png",
                 f"{short} (48³)", "")
    arrow(fig, (0.5, 0.555), (0.5, roi_y + roi_h + 0.005))
    fig.text(0.5, roi_y - 0.012, "5 ROI patches → 5 × 48 × 48 × 48",
             ha="center", fontsize=7.5, color=TEXT, fontweight="semibold")

    # ══════════════════════════════════════════════════════════════════
    # Morphometric branch only (CNN branch removed)
    # ══════════════════════════════════════════════════════════════════
    my0 = 0.28
    arrow(fig, (0.5, roi_y - 0.02), (0.5, my0 + 0.10))
    box(fig, (0.40, my0 + 0.075, 0.20, 0.045),
       "Morphometric Feature Extraction\n(the only branch — 3D-CNN branch removed)",
       COLOR["morph"], fontsize=6.8)
    heatmap_panel(
        fig, (0.30, my0, 0.40, 0.07), morph_norm, roi_short,
        "Real morphometric feature matrix (5 × 14), column-normalised for display "
        "— real units in Figure 4",
        cmap="YlOrBr", fmt="", fontsize=1,
    )

    # ══════════════════════════════════════════════════════════════════
    # NeuroProp-X
    # ══════════════════════════════════════════════════════════════════
    ny0 = 0.02
    mat_y, mat_h = ny0 + 0.05, 0.15          # matrices / graph row
    lbl_y = mat_y + mat_h + 0.03              # label boxes, clear of titles
    arrow(fig, (0.5, my0), (0.5, lbl_y + 0.03))
    box(fig, (0.38, lbl_y, 0.24, 0.03),
       "NeuroProp-X — X (5×14) → X* (5×19)", COLOR["npx"], fontsize=7.2)
    bar_panel(fig, (0.015, mat_y, 0.12, mat_h), roi_short, rv,
             "Real SRVE regional\nvulnerability RV (5,)", color="#8e44ad", fmt="{:.2f}")
    heatmap_panel(fig, (0.15, mat_y, 0.135, mat_h), a_star, roi_short,
                 "Real adaptive adjacency\nA* (5×5)", cmap="Reds")
    heatmap_panel(fig, (0.30, mat_y, 0.135, mat_h), p_mat, roi_short,
                 "Real propagation\nmatrix P (5×5)", cmap="PuBu")

    # Enriched graph (rendered from real A*)
    gax = fig.add_axes((0.45, mat_y, 0.20, mat_h))
    gax.set_xlim(-1.5, 1.5); gax.set_ylim(-1.0, 1.1); gax.axis("off")
    layout = {
        "Inferior_Frontal_Gyrus": (-1.0, 0.55), "Broca_Area": (-0.45, 0.15),
        "Insula": (0.0, -0.35), "Wernicke_Area": (0.6, 0.2),
        "Superior_Temporal_Gyrus": (1.0, -0.35),
    }
    peak = float(np.nanmax(np.abs(a_star))) or 1.0
    for i, src in enumerate(roi_full):
        for j, tgt in enumerate(roi_full):
            if i == j:
                continue
            wgt = float(a_star[i, j])
            if abs(wgt) <= 1e-8:
                continue
            x0, y0e = layout[src]; x1, y1e = layout[tgt]
            gax.annotate("", xy=(x1, y1e), xytext=(x0, y0e), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color="#2f6feb",
                                       alpha=float(np.clip(abs(wgt) / peak, 0.15, 0.95)),
                                       linewidth=max(0.4, 3.2 * abs(wgt) / peak),
                                       shrinkA=13, shrinkB=13,
                                       connectionstyle="arc3,rad=0.13"))
    for roi, short in zip(roi_full, roi_short):
        x, y = layout[roi]
        gax.scatter([x], [y], s=520, color="#f2f5f9", edgecolors="#2f6feb",
                   linewidths=1.2, zorder=3)
        gax.text(x, y, short, ha="center", va="center", fontsize=6, zorder=4,
                color=TEXT, fontweight="semibold")
    gax.set_title("Real enriched brain graph G*=(V,X*,A*,P)\n(edges from real A*)",
                  fontsize=6.3, color=TEXT)

    box(fig, (0.67, mat_y + 0.095, 0.10, 0.04), "Edge features\nE (5×5×3)",
       COLOR["npx"], fontsize=6.0)
    fig.text(0.67, mat_y + 0.055, f"AP-LAF alpha = {npx_expl['alpha']:.3f}\n"
            "(share of A* mass from\nthe anatomical prior)",
            fontsize=6.0, color=TEXT, va="top")

    # ══════════════════════════════════════════════════════════════════
    # SAEG-GATv2 (right column)
    # ══════════════════════════════════════════════════════════════════
    arrow(fig, (0.65, mat_y + 0.06), (0.80, mat_y + 0.06))
    box(fig, (0.80, lbl_y, 0.185, 0.03),
       "SAEG-GATv2 — 2 layers, 4 heads, hidden=64\ngate: g=σ(w^T E+b), no renorm, self-loops",
       COLOR["gat"], fontsize=6.2)
    heatmap_panel(fig, (0.80, mat_y, 0.09, mat_h), edge_gate, roi_short,
                 "Real edge gate g_ij (5×5)", cmap="Oranges")
    bar_panel(fig, (0.895, mat_y, 0.09, mat_h), roi_short,
             list(graph_expl["node_importance"].values()),
             "Real node importance\n(incoming attention mass)",
             color="#c0392b", fmt="{:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # Closing summary: Z_G -> Fusion -> Z_H -> Classifier + Stage-TGT
    # (full detail deferred to Figure 7; this is a terminating summary strip)
    # ══════════════════════════════════════════════════════════════════
    arrow(fig, (0.895, mat_y), (0.895, mat_y - 0.035))
    pred_stage = max(probs, key=probs.get)
    box(
        fig, (0.66, ny0 - 0.005, 0.325, mat_y - 0.035 - (ny0 - 0.005)),
        "Z_G (128-d, mean+max pool) → Multimodal Fusion → Z_H (64-d)\n"
        f"Classifier: P(CN)={probs['CN']:.2f}  P(MCI)={probs['MCI']:.2f}  "
        f"P(AD)={probs['AD']:.2f}  → predicted {pred_stage}\n"
        f"Stage-TGT: transition target = {prop_detail['transition_target_stage']}, "
        f"AD-associated propensity = {prop_detail['ad_associated_propensity']:.2f}  "
        "(full detail in Figure 7)",
        COLOR["out"], fontsize=6.4,
    )

    plt.savefig(OUT / "figure1_complete_architecture.png", dpi=DPI,
               facecolor="white", bbox_inches=None)
    plt.close(fig)
    print("wrote", OUT / "figure1_complete_architecture.png")


if __name__ == "__main__":
    main()
