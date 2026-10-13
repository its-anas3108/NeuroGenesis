"""Figure 5 — NeuroProp-X (SRVE + AP-LAF + ANP + SAGR), real data only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BORDER, COLOR, MUTED, ROI_FULL, ROI_SHORT, SID, TEXT,
    bar_panel, box, footer, heatmap_panel, load_npx, load_record, save, text_panel,
)


def main() -> None:
    npx = load_npx()
    record = load_record()
    g_star = npx["g_star"]

    fig = plt.figure(figsize=(26, 22), facecolor="white", dpi=300)
    fig.suptitle("Figure 5 — NeuroProp-X: X (5×14) → X* (5×19)",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.977,
             f"Subject {SID}  —  SRVE (vulnerability) → AP-LAF (adaptive "
             "adjacency) → ANP (propagation) → SAGR (enriched features)",
             ha="center", fontsize=10, color=MUTED)

    # ── 4-stage sub-pipeline boxes ─────────────────────────────────
    stage_labels = [
        "1. SRVE\nStructural-Regional\nVulnerability Estimation",
        "2. AP-LAF\nAnatomical-Prior Learned\nAdaptive Fusion",
        "3. ANP\nAdaptive Network\nPropagation",
        "4. SAGR\nStructure-Aware\nGraph Readout / enrichment",
    ]
    sx = np.linspace(0.03, 0.03 + 3 * 0.245, 4)
    for x, label in zip(sx, stage_labels):
        box(fig, (x, 0.925, 0.21, 0.045), label, COLOR["npx"], fontsize=8.0)

    # ── SRVE: regional vulnerability ───────────────────────────────
    rv = npx["rv"]
    bar_panel(fig, (0.02, 0.77, 0.20, 0.12), ROI_SHORT, rv,
             "1. Real SRVE regional vulnerability RV (5,)", color="#8e44ad")
    vr = g_star["vulnerability_ranking"]
    text_panel(fig, (0.02, 0.63, 0.20, 0.12), "Real vulnerability ranking",
              [f"#{v['rank']}  {v['roi_short']}: {v['vulnerability']:.3f}" for v in vr],
              fontsize=6.6, color=COLOR["npx"])

    # ── AP-LAF: A_prior, A_att, A_star ─────────────────────────────
    heatmap_panel(fig, (0.25, 0.77, 0.155, 0.14), npx["a_prior"], ROI_SHORT,
                 "2a. Real anatomical prior A_prior", cmap="Blues")
    heatmap_panel(fig, (0.43, 0.77, 0.155, 0.14), npx["a_att"], ROI_SHORT,
                 "2b. Real learned attention A_att", cmap="Greens")
    heatmap_panel(fig, (0.61, 0.77, 0.155, 0.14), npx["a_star"], ROI_SHORT,
                 "2c. Real fused A* = f(A_prior, A_att)", cmap="Reds")
    beta = g_star["beta"]
    text_panel(
        fig, (0.79, 0.77, 0.19, 0.14), "Real AP-LAF fusion coefficients",
        [f"alpha (prior mass) = {g_star['alpha']:.4f}",
         f"beta1 (src vuln) = {beta['beta1_source_vulnerability']:.4f}",
         f"beta2 (tgt vuln) = {beta['beta2_target_vulnerability']:.4f}",
         f"beta3 (adapt.adj) = {beta['beta3_adaptive_adjacency']:.4f}",
         f"beta4 (topology) = {beta['beta4_topology']:.4f}",
         f"bias = {beta['bias']:.4f}"],
        fontsize=6.8, color=COLOR["npx"],
    )

    # ── ANP: propagation ────────────────────────────────────────────
    heatmap_panel(fig, (0.25, 0.60, 0.155, 0.14), npx["p"], ROI_SHORT,
                 "3. Real propagation matrix P (5×5)", cmap="PuBu")
    text_panel(
        fig, (0.43, 0.60, 0.165, 0.14), "3. ANP — propagation",
        ["P_ij = how much of ROI j's",
         "vulnerability signal reaches",
         "ROI i after iterative",
         "propagation over A*."],
        fontsize=6.6, color=COLOR["npx"],
    )
    heatmap_panel(fig, (0.61, 0.60, 0.10, 0.14), npx["edge_features"][:, :, 0], ROI_SHORT,
                 "Edge feat. ch.0", cmap="Oranges", fmt="")
    heatmap_panel(fig, (0.715, 0.60, 0.10, 0.14), npx["edge_features"][:, :, 1], ROI_SHORT,
                 "Edge feat. ch.1", cmap="Oranges", fmt="")
    heatmap_panel(fig, (0.82, 0.60, 0.10, 0.14), npx["edge_features"][:, :, 2], ROI_SHORT,
                 "Edge feat. ch.2", cmap="Oranges", fmt="")
    fig.text(0.87, 0.575, "Real E (5×5×3)", ha="center", fontsize=6.6, color=MUTED)

    # ── SAGR: enriched X* matrix ────────────────────────────────────
    x_star = npx["x_star"]
    ax = fig.add_axes((0.06, 0.36, 0.88, 0.15))
    im = ax.imshow(x_star, cmap="YlOrBr", aspect="auto")
    ax.set_yticks(range(5), ROI_SHORT, fontsize=7)
    ax.set_xticks(range(19), [f"f{i}" for i in range(19)], fontsize=5.5)
    slices = g_star["component_slices"]
    for name, (a, b) in slices.items():
        ax.axvline(a - 0.5, color=TEXT, linewidth=1.0)
        ax.text((a + b - 1) / 2, -0.9, name.replace("_", " "), ha="center",
               fontsize=6.8, color=TEXT, fontweight="semibold")
    ax.axvline(18.5, color=TEXT, linewidth=1.0)
    ax.set_title("4. Real enriched node feature matrix X* (5×19) — morphometry "
                "[0:14] | regional vulnerability [14:15] | SAGR centrality [15:19]",
                fontsize=8.5, color=TEXT, pad=14)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=5)

    # ── Enriched graph (larger) ──────────────────────────────────────
    gax = fig.add_axes((0.06, 0.03, 0.42, 0.30))
    gax.set_xlim(-1.6, 1.6); gax.set_ylim(-1.1, 1.2); gax.axis("off")
    layout = {
        "Inferior_Frontal_Gyrus": (-1.0, 0.55), "Broca_Area": (-0.45, 0.15),
        "Insula": (0.0, -0.35), "Wernicke_Area": (0.6, 0.2),
        "Superior_Temporal_Gyrus": (1.0, -0.35),
    }
    a_star = npx["a_star"]
    peak = float(np.nanmax(np.abs(a_star))) or 1.0
    for i, src in enumerate(ROI_FULL):
        for j, tgt in enumerate(ROI_FULL):
            if i == j:
                continue
            wgt = float(a_star[i, j])
            if abs(wgt) <= 1e-8:
                continue
            x0, y0e = layout[src]; x1, y1e = layout[tgt]
            gax.annotate("", xy=(x1, y1e), xytext=(x0, y0e), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color="#2f6feb",
                                       alpha=float(np.clip(abs(wgt) / peak, 0.15, 0.95)),
                                       linewidth=max(0.5, 4.0 * abs(wgt) / peak),
                                       shrinkA=17, shrinkB=17,
                                       connectionstyle="arc3,rad=0.13"))
    for roi, short in zip(ROI_FULL, ROI_SHORT):
        x, y = layout[roi]
        size = 700 + 2200 * float(g_star["regional_vulnerability"][roi])
        gax.scatter([x], [y], s=size, color="#f2f5f9", edgecolors="#8e44ad", linewidths=1.6, zorder=3)
        gax.text(x, y, short, ha="center", va="center", fontsize=8, zorder=4,
                color=TEXT, fontweight="semibold")
    gax.set_title("Real enriched brain graph G* = (V, X*, A*, P)\nnode size ∝ real "
                  "regional vulnerability, edges = real A*", fontsize=8.5, color=TEXT)

    text_panel(
        fig, (0.52, 0.03, 0.44, 0.30), "NeuroProp-X summary (all real, verified from code)",
        ["Input:  X (5×14, real morphometric features, Figure 4)",
         "Output: X* (5×19), A* (5×5), P (5×5), E (5×5×3)",
         "",
         "SRVE computes RV per ROI from morphometric deviation signals.",
         "AP-LAF fuses the Harvard-Oxford anatomical prior A_prior with a",
         "learned attention matrix A_att into A* (alpha = prior-mass share).",
         "ANP iteratively propagates RV over A* to obtain P.",
         "SAGR appends RV + 4 real graph-centrality features to X,",
         "producing the 19-dim enriched node representation X*.",
         "",
         f"AP-LAF alpha for this subject: {g_star['alpha']:.3f} "
         "(share of A* mass from the anatomical prior, vs. learned attention)",
         "",
         "X*, A*, P, E → input to SAEG-GATv2 (Figure 6)"],
        fontsize=7.4, color=COLOR["npx"],
    )

    footer(fig, "node_feature_dim=19 (verified: modules/m06_neuropropx/"
          "neuropropx_engine.py); n_nodes=5")
    save(fig, "figure5_neuropropx.png")


if __name__ == "__main__":
    main()
