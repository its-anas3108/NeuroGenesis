"""Figure 4 — Morphometric Feature Extraction (real 5x14 matrix), real data only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BORDER, COLOR, FEATURE_COLS, MUTED, OASIS_OUT, ROI_FULL, ROI_SHORT, SID,
    TEXT, bar_panel, box, footer, load_morph_matrix, save, text_panel,
)

FEATURE_UNITS = {
    "voxel_count": "voxels", "brain_volume_mm3": "mm³", "gm_volume_mm3": "mm³",
    "mean_intensity": "a.u.", "std_intensity": "a.u.", "skewness": "—",
    "kurtosis": "—", "entropy": "bits", "surface_area_vox": "voxels",
    "cortical_thickness_mm": "mm", "gm_fraction": "ratio",
    "normalized_volume": "ratio", "surface_to_volume_ratio": "1/mm",
    "atrophy_index": "ratio",
}


def full_table(fig, xywh, matrix: pd.DataFrame):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    rows = [[short] + [
        ("n/a" if pd.isna(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.1f}")
        for v in matrix.loc[roi]
    ] for roi, short in zip(ROI_FULL, ROI_SHORT)]
    col_labels = ["ROI"] + [f"{c}\n({FEATURE_UNITS[c]})" for c in FEATURE_COLS]
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.0)
    tbl.scale(1, 2.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COLOR["morph"]); cell.set_text_props(fontweight="bold", fontsize=5.6)
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#fdf6e8")
        if c == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("Real morphometric feature matrix X (5 ROIs × 14 features), "
                 "real units — read directly from morphometric_features.csv",
                 fontsize=8.5, color=TEXT, pad=10)
    return ax


def main() -> None:
    matrix = load_morph_matrix()

    fig = plt.figure(figsize=(26, 20), facecolor="white", dpi=300)
    fig.suptitle("Figure 4 — Morphometric Feature Extraction (X: 5 × 14)",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.975,
             f"Subject {SID}  —  the only feature branch in this architecture "
             "(3D-CNN branch removed); features computed per-ROI from the M7 patches",
             ha="center", fontsize=10, color=MUTED)

    box(fig, (0.32, 0.90, 0.36, 0.045),
       "Morphometric Feature Extraction\n14 features computed per ROI from each 48³ patch",
       COLOR["morph"], fontsize=9)

    full_table(fig, (0.02, 0.62, 0.96, 0.24), matrix)

    key_feats = [
        ("cortical_thickness_mm", "Cortical Thickness (mm)", "#8e5a2f"),
        ("brain_volume_mm3", "Regional Volume (mm³)", "#2f6feb"),
        ("gm_fraction", "GM Fraction (ratio)", "#2f9e5c"),
        ("surface_area_vox", "Surface Area (voxels)", "#c0392b"),
        ("entropy", "Intensity Entropy (bits)", "#8e44ad"),
        ("mean_intensity", "Mean Intensity (a.u.)", "#d68910"),
    ]
    xs = np.linspace(0.02, 0.02 + 5 * 0.163, 6)
    for x, (col, title, color) in zip(xs, key_feats):
        bar_panel(fig, (x, 0.34, 0.145, 0.22), ROI_SHORT,
                 matrix[col].to_numpy(dtype=float), title, color=color,
                 fmt="{:.2f}", horizontal=False)

    text_panel(
        fig, (0.02, 0.06, 0.45, 0.24), "All 14 real features (definitions)",
        ["voxel_count, brain_volume_mm3, gm_volume_mm3 — real segmentation counts/volumes",
         "mean_intensity, std_intensity, skewness, kurtosis, entropy — real intensity statistics",
         "surface_area_vox — real marching-cubes surface voxel count (no anisotropic\n"
         "voxel-size correction applied — kept in voxel units, not cm²)",
         "cortical_thickness_mm — real distance-transform thickness estimate",
         "gm_fraction — real GM/total volume ratio",
         "normalized_volume, surface_to_volume_ratio — real derived ratios",
         "atrophy_index — NaN for this subject (no reference/normative baseline\n"
         "available for a single-timepoint OASIS-1 cross-sectional scan; not fabricated)"],
        fontsize=7.0, color=COLOR["morph"],
    )
    text_panel(
        fig, (0.53, 0.06, 0.45, 0.24), "Where this matrix goes next",
        [f"X (5×14, real) → input to NeuroProp-X (Figure 5)",
         "NeuroProp-X appends SAGR centrality features → X* (5×19, real,",
         "verified from modules/m06_neuropropx/neuropropx_engine.py)",
         "No concatenation with any spatial/CNN embedding occurs — the",
         "3D-CNN branch was removed from this implementation; X is the",
         "complete node feature source for the entire downstream graph."],
        fontsize=7.4, color=COLOR["npx"],
    )

    footer(fig, "X: 5 ROIs × 14 real features per ROI, real units, no normalization "
          "applied at this stage")
    save(fig, "figure4_morphometric_features.png")


if __name__ == "__main__":
    main()
