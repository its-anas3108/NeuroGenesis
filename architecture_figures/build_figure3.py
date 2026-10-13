"""Figure 3 — M6-M7 Speech ROI Localization & Patch Extraction, real data only."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BORDER, COLOR, MUTED, OASIS_OUT, OUT, ROI_FULL, ROI_SHORT, SID, TEXT,
    box, footer, img_panel, save, text_panel,
)


def stats_table(fig, xywh, rois):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    rows = []
    for r in rois:
        bb = r["bounding_box"]
        rows.append([
            r["roi_name"].replace("_", " "),
            f"{r['voxel_count']:,}",
            f"{r['volume_mm3']:,.0f}",
            f"({bb[0][0]}-{bb[0][1]}, {bb[1][0]}-{bb[1][1]}, {bb[2][0]}-{bb[2][1]})",
            f"({r['centroid'][0]:.1f}, {r['centroid'][1]:.1f}, {r['centroid'][2]:.1f})",
        ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["ROI (Harvard-Oxford atlas)", "Voxel count", "Volume (mm³)",
                  "Bounding box (x, y, z)", "Centroid (voxel coords)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COLOR["roi"]); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#f2f7f3")
    ax.set_title("Real per-ROI segmentation statistics (from segmentation.json, "
                 "atlas: cort-maxprob-thr25-2mm)", fontsize=7.4, color=TEXT, pad=8)
    return ax


def main() -> None:
    roi_masks = OASIS_OUT / "roi" / "masks" / SID
    seg = json.loads(
        (OASIS_OUT / "roi" / SID / f"{SID}_segmentation.json").read_text()
    )
    tensor = np.load(
        OASIS_OUT / "roi" / "patches" / SID / f"{SID}_roi_tensor.npy"
    )

    fig = plt.figure(figsize=(24, 22), facecolor="white", dpi=300)
    fig.suptitle("Figure 3 — M6–M7 Speech ROI Localization & Patch Extraction",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.975,
             f"Subject {SID}  —  Harvard-Oxford cortical atlas, 25% probability "
             "threshold, registered to each subject's standardized 128³ volume",
             ha="center", fontsize=10, color=MUTED)

    box(fig, (0.30, 0.925, 0.40, 0.035),
       "M6  Atlas-Based Speech ROI Localization  →  M7  48³ Patch Extraction",
       COLOR["roi"], fontsize=9)

    img_panel(fig, (0.06, 0.66, 0.40, 0.24), "png",
             roi_masks / f"{SID}_all_rois_combined.png",
             "Real combined ROI overlay (all 5 speech ROIs)")
    img_panel(fig, (0.52, 0.66, 0.42, 0.24), "png",
             roi_masks / f"{SID}_roi_statistics.png",
             "Real ROI statistics panel (as saved by the pipeline)")

    xs = np.linspace(0.015, 0.015 + 4 * 0.198, 5)
    pw = 0.175
    overlay_files = ["broca_area", "wernicke_area", "insula",
                     "inferior_frontal_gyrus", "superior_temporal_gyrus"]
    for x, fname, short in zip(xs, overlay_files, ROI_SHORT):
        img_panel(fig, (x, 0.40, pw, 0.22), "png",
                 roi_masks / f"{SID}_{fname}_overlay.png", f"{short} overlay")

    fig.text(0.5, 0.375, "Real 48×48×48 patches extracted at each ROI centroid "
             "(roi_tensor.npy, mid-axial slice shown)", ha="center",
             fontsize=8.5, color=TEXT, fontweight="semibold")
    for i, (x, short) in enumerate(zip(xs, ROI_SHORT)):
        ax = fig.add_axes((x + (pw - 0.14) / 2, 0.14, 0.14, 0.20))
        ax.imshow(np.rot90(tensor[i, :, :, 24]), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(BORDER); s.set_linewidth(0.7)
        ax.set_title(f"{short} patch (48³)", fontsize=6.8, color=TEXT, pad=3)

    stats_table(fig, (0.03, 0.02, 0.94, 0.10), seg["rois"])

    footer(fig, "5 ROIs × 48³ → patch tensor shape (5, 48, 48, 48), "
          "real (from roi_tensor.npy)")
    save(fig, "figure3_roi_localization.png")


if __name__ == "__main__":
    main()
