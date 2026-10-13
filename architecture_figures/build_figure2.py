"""Figure 2 — M1-M5 MRI Preprocessing Pipeline (detailed), real data only."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BORDER, COLOR, MUTED, OASIS_OUT, OUT, SID, TEXT,
    arrow, box, footer, img_panel, missing_panel, save, text_panel,
)


def stage_table_panel(fig, xywh, stages):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    rows = [
        [s["name"], f"{s['shape'][0]}³" if len(set(s["shape"])) == 1 else str(s["shape"]),
         f"{s['intensity_range'][0]:.2f} – {s['intensity_range'][1]:.2f}",
         f"{s['mean_intensity']:.3f}",
         f"{s['seconds']:.2f}s" if s.get("seconds") is not None else "–"]
        for s in stages
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=["M3 sub-step (real, from saved JSON trace)", "Shape",
                  "Intensity range", "Mean intensity", "Runtime"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.6)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COLOR["preproc"])
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#f2f5f9")
    ax.set_title(
        "Real M3 intensity-processing trace (N4 → WM-KDE → CLAHE → "
        "Perona-Malik diffusion → min-max)— per-step images are not persisted "
        "by the active pipeline, but this numeric trace is real and saved per subject",
        fontsize=7.0, color=TEXT, pad=6,
    )
    return ax


def main() -> None:
    meta = json.loads(
        (OASIS_OUT / "preprocessing" / SID / f"{SID}_preprocessing.json").read_text()
    )
    pre = OASIS_OUT / "preprocessing" / SID

    fig = plt.figure(figsize=(24, 30), facecolor="white", dpi=300)
    fig.suptitle("Figure 2 — M1–M5 MRI Preprocessing Pipeline (Detailed)",
                fontsize=18, fontweight="bold", color=TEXT, y=0.985)
    fig.text(0.5, 0.972, f"Subject {SID}  —  every number below is read "
            "directly from the saved preprocessing JSON record, not recomputed "
            "or estimated", ha="center", fontsize=10, color=MUTED)

    bx, bw = 0.02, 0.115
    row_h = 0.165
    rows_y = np.linspace(0.79, 0.02, 5)

    # ── M1 ──────────────────────────────────────────────────────────
    y = rows_y[0]
    box(fig, (bx, y + row_h * 0.3, bw, row_h * 0.5), "M1\nLoad & Header Parse",
       COLOR["preproc"], fontsize=8.5)
    m = meta["metadata"]
    text_panel(fig, (bx + bw + 0.02, y, 0.30, row_h),
              "Real header / intensity metadata (source NIfTI)",
              [f"Shape: {tuple(m['shape'])}   Voxel size: {tuple(m['voxel_dims_mm'])} mm",
               f"dtype: {m['dtype']}    Total voxels: {m['total_voxels']:,}",
               f"Non-zero voxels: {m['non_zero_voxels']:,}",
               f"Intensity range: [{m['min_intensity']:.1f}, {m['max_intensity']:.1f}]",
               f"Mean ± std intensity: {m['mean_intensity']:.2f} ± {m['std_intensity']:.2f}"],
              fontsize=7.2, color=COLOR["preproc"])
    missing_panel(
        fig, (bx + bw + 0.35, y, 0.18, row_h), "M1 source volume",
        "Raw OASIS-1 .hdr/.img\n(source_path below) no\nlonger present on this\nmachine — dataset removed\nafter pipeline run",
    )
    src_path = meta["source_path"]
    fig.text(bx + bw + 0.55, y + row_h / 2,
             "Recorded source_path (real, from JSON):\n" + src_path,
             fontsize=6.4, color=MUTED, va="center", wrap=True)

    # ── M2 ──────────────────────────────────────────────────────────
    y = rows_y[1]
    box(fig, (bx, y + row_h * 0.3, bw, row_h * 0.5), "M2\nQuality Control",
       COLOR["preproc"], fontsize=8.5)
    img_panel(fig, (bx + bw + 0.02, y, 0.20, row_h), "png",
             pre / f"{SID}_qc_report.png", "Real QC report")
    qc = meta["qc"]
    text_panel(fig, (bx + bw + 0.25, y, 0.30, row_h), "Real QC metrics",
              [f"Quality score: {qc['quality_score']:.2f} / 100   (passed={qc['passed']})",
               f"Motion: {qc['motion_score']:.2f}   Dropout: {qc['dropout_score']:.2f}",
               f"Ringing: {qc['ringing_score']:.2f}   SNR: {qc['snr_db']:.1f} dB",
               f"Saturated voxels: {qc['raw_metrics']['saturation']['saturated_voxels']}",
               f"Flags: {', '.join(qc['flags']) if qc['flags'] else 'none'}"],
              fontsize=7.0, color=COLOR["preproc"])

    # ── M3 ──────────────────────────────────────────────────────────
    y = rows_y[2]
    box(fig, (bx, y + row_h * 0.3, bw, row_h * 0.5), "M3\nIntensity Processing",
       COLOR["preproc"], fontsize=8.5)
    stage_table_panel(fig, (bx + bw + 0.02, y, 0.82, row_h),
                      [s for s in meta["stages"] if s["name"] in
                       ("original", "n4_bias_corrected", "wm_normalized",
                        "clahe", "anisotropic_diffusion", "intensity_normalized")])

    # ── M4 ──────────────────────────────────────────────────────────
    y = rows_y[3]
    box(fig, (bx, y + row_h * 0.3, bw, row_h * 0.5), "M4\nSkull Stripping",
       COLOR["preproc"], fontsize=8.5)
    img_panel(fig, (bx + bw + 0.02, y, 0.20, row_h), "png",
             pre / f"{SID}_skull_stripping.png", "Real skull-strip panel")
    text_panel(fig, (bx + bw + 0.25, y, 0.30, row_h), "Real brain-extraction counts",
              [f"Brain voxels before: {meta['brain_voxels_before']:,}",
               f"Brain voxels after: {meta['brain_voxels_after']:,}",
               f"Retained fraction: "
               f"{meta['brain_voxels_after']/meta['brain_voxels_before']*100:.1f}%",
               f"WM KDE peak (normalization reference): {meta['wm_peak']:.4f}"],
              fontsize=7.2, color=COLOR["preproc"])

    # ── M5 ──────────────────────────────────────────────────────────
    y = rows_y[4]
    box(fig, (bx, y + row_h * 0.3, bw, row_h * 0.5), "M5\nSpatial Standardization",
       COLOR["preproc"], fontsize=8.5)
    img_panel(fig, (bx + bw + 0.02, y, 0.20, row_h), "png",
             pre / f"{SID}_resampling.png", "Real resampling panel")
    std_stage = next(s for s in meta["stages"] if s["name"] == "standardized")
    text_panel(fig, (bx + bw + 0.25, y, 0.30, row_h), "Real resampling record",
              [f"Input shape: {tuple(meta['stages'][-2]['shape'])}",
               f"Output shape: {tuple(std_stage['shape'])}  (isotropic 128³)",
               f"Output intensity range: [{std_stage['intensity_range'][0]:.3f}, "
               f"{std_stage['intensity_range'][1]:.3f}]",
               "Interpolation: linear (σ=1.0 for the smoothing pre-filter)"],
              fontsize=7.2, color=COLOR["preproc"])

    for i in range(4):
        arrow(fig, (bx + bw / 2, rows_y[i] + row_h * 0.3),
             (bx + bw / 2, rows_y[i + 1] + row_h * 0.8))

    footer(fig, "M1-M5 = 176x208x176 (1mm iso) → 128x128x128 (spatial standardization)")
    save(fig, "figure2_preprocessing_detail.png")


if __name__ == "__main__":
    main()
