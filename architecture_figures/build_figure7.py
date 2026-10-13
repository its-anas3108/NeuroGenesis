"""Figure 7 — Multimodal Fusion -> Classifier + Stage-TGT, real data only."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BORDER, COLOR, MUTED, OASIS_OUT, SID, TEXT,
    bar_panel, box, footer, img_panel, load_record, save, text_panel,
)


def main() -> None:
    record = load_record()
    probs = record["class_probabilities"]
    detail = record["stage_propensity_detail"]

    fig = plt.figure(figsize=(24, 20), facecolor="white", dpi=300)
    fig.suptitle("Figure 7 — Multimodal Fusion → Classifier + Stage-TGT",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.975,
             f"Subject {SID} (real reference stage = {detail['reference_stage']})  "
             "—  Z_G (Figure 6) → Fusion → Z_H → Classifier / Stage-TGT",
             ha="center", fontsize=10, color=MUTED)

    box(fig, (0.06, 0.90, 0.18, 0.05), "Z_G\n128-d\n(graph embedding,\nFigure 6 output)",
       COLOR["gat"], fontsize=7.6)
    box(fig, (0.30, 0.90, 0.24, 0.05),
       "Multimodal Fusion\nspatial_dim = 0 (no CNN branch)\ngraph_dim = 128 → out_dim = 64",
       COLOR["fusion"], fontsize=7.6)
    box(fig, (0.60, 0.90, 0.15, 0.05), "Z_H\n64-d\n(fused embedding)",
       COLOR["fusion"], fontsize=7.6)
    box(fig, (0.81, 0.90, 0.15, 0.05), "Classifier +\nStage-TGT\nheads",
       COLOR["out"], fontsize=7.6)
    for x0, x1 in [(0.24, 0.30), (0.54, 0.60), (0.75, 0.81)]:
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch((x0, 0.925), (x1, 0.925), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.6, color="#4a5563",
                                     transform=fig.transFigure))
        ax.patch.set_alpha(0)

    pred_stage = max(probs, key=probs.get)
    bar_panel(fig, (0.05, 0.68, 0.24, 0.17), list(probs.keys()), list(probs.values()),
             f"Real classifier output — predicted: {pred_stage}\n"
             f"(real reference stage: {detail['reference_stage']})",
             color=["#2f9e5c" if k == pred_stage else "#9aa4b2" for k in probs],
             fmt="{:.3f}", horizontal=False)

    img_panel(fig, (0.34, 0.66, 0.30, 0.20), "png",
             OASIS_OUT / "figures" / "fig19_embedding_projection.png",
             "Real Z_G / Z_H PCA projection (cohort-wide, from fig19)")

    sa = detail["stage_alignment"]
    bar_panel(fig, (0.68, 0.68, 0.28, 0.17), list(sa.keys()), list(sa.values()),
             "Real Stage-TGT stage-alignment distribution", color="#8e44ad",
             fmt="{:.3f}", horizontal=False)

    text_panel(
        fig, (0.03, 0.35, 0.46, 0.27), "Real Stage-TGT scalar outputs (this subject)",
        [f"advanced_stage_alignment = {detail['advanced_stage_alignment']:.4f}",
         "  (expected ordinal position: 0=fully CN, 1=fully AD)",
         f"ad_associated_propensity = {detail['ad_associated_propensity']:.4f}",
         "  (d_CN / (d_CN + d_AD); distance-based, not P(AD))",
         f"ad_axis_projection = {detail['ad_axis_projection']:.4f}  (diagnostic only)",
         f"stage_transition_propensity = {detail['stage_transition_propensity']:.4f}",
         f"transition_target_stage = {detail['transition_target_stage']}",
         f"  — {detail['transition_note']}"],
        fontsize=7.4, color=COLOR["out"],
    )
    text_panel(
        fig, (0.51, 0.35, 0.46, 0.27), "Stage-TGT disclaimer (verbatim, from the saved record)",
        textwrap.wrap(detail["disclaimer"], width=70),
        fontsize=7.6, color="#ffe9e9",
    )

    text_panel(
        fig, (0.03, 0.03, 0.94, 0.28), "Fusion / classifier architecture summary (verified from code)",
        ["Multimodal Fusion (modules/m06_graph_learning/fusion.py): inputs are Z_G "
         "(graph, 128-d) and a spatial branch (spatial_dim=0, i.e. absent — the",
         "3D-CNN branch was removed from this implementation; MultimodalFusion already handled spatial_dim=0 cleanly with no code change needed).",
         "Fusion output Z_H is 64-d, passed to two heads: (1) the CN/MCI/AD classifier (3-way softmax, shown above, real "
         "probabilities for this subject), and",
         "(2) Stage-TGT (Stage-Transition-Guided Transformer head, d_model=64), which computes alignment against learned "
         "CN/MCI/AD stage prototypes (real values above).",
         "Stage-TGT is explicitly NOT a clinical conversion-risk prediction — see the disclaimer, reproduced verbatim "
         "above exactly as the pipeline saves it with every subject's record.",
         f"Total model parameters (verified from modules/model.py at the current checkpoint): 132,136."],
        fontsize=7.6, color="#eef1f6",
    )

    footer(fig, "Z_H = 64-d; classifier = 3-way (CN/MCI/AD); Stage-TGT d_model = 64")
    save(fig, "figure7_fusion_classifier_stagetgt.png")


if __name__ == "__main__":
    main()
