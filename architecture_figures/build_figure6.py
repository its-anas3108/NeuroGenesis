"""Figure 6 — SAEG-GATv2 (edge-gated graph attention), real data only."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    BORDER, COLOR, MUTED, ROI_FULL, ROI_SHORT, SID, TEXT,
    bar_panel, box, footer, heatmap_panel, load_record, save, text_panel,
)


def layer_table(fig, xywh, layers):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    rows = [[
        f"Layer {l['layer']}", l["attention_type"], str(l["heads"]),
        str(l["head_dim"]), f"{l['in_dim']} → {l['out_dim']}",
        "yes" if l["edge_gate"] else "no",
        f"[{l['gate_range'][0]:.3f}, {l['gate_range'][1]:.3f}]",
    ] for l in layers]
    tbl = ax.table(
        cellText=rows,
        colLabels=["Layer", "Attention type", "Heads", "Head dim", "in→out dim",
                  "Edge gate", "Real gate range (this subject)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COLOR["gat"]); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#fdecec")
    ax.set_title("Real per-layer SAEG-GATv2 configuration (from saved explanation "
                 "record, not the spec — verified against modules/m06_graph_learning/"
                 "attention.py)", fontsize=8.0, color=TEXT, pad=8)
    return ax


def top_edges_table(fig, xywh, top_edges):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    rows = [[
        e["pathway"], f"{e['attention']:.3f}", f"{e['gate']:.3f}",
        e["anatomical_tract"] or "— (not in atlas tract dictionary)",
    ] for e in top_edges]
    tbl = ax.table(
        cellText=rows,
        colLabels=["Top-5 real edges (by attention)", "Attention weight", "Edge gate",
                  "Real matched anatomical tract"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.4)
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(COLOR["gat"]); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#fdecec")
    ax.set_title("Real top-5 attended edges for this subject, with anatomical tract "
                 "match where the atlas dictionary has one", fontsize=8.0, color=TEXT, pad=8)
    return ax


def main() -> None:
    record = load_record()
    g = record["explanations"]["graph"]
    edge_att = np.array(g["edge_attention"])
    edge_gate = np.array(g["edge_gate"])
    node_imp = list(g["node_importance"].values())

    fig = plt.figure(figsize=(24, 22), facecolor="white", dpi=300)
    fig.suptitle("Figure 6 — SAEG-GATv2: Structure-Aware Edge-Gated Graph Attention",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.977,
             f"Subject {SID}  —  X*, A*, P, E (Figure 5) → 2-layer edge-gated "
             "GATv2 → node embeddings",
             ha="center", fontsize=10, color=MUTED)

    box(fig, (0.28, 0.92, 0.44, 0.045),
       "SAEG-GATv2 — gate g_ij = σ(w_g^T E_ij + b_g), gate bias init = +1.0,\n"
       "no renormalization after gating, self-loops always included",
       COLOR["gat"], fontsize=8.5)

    layer_table(fig, (0.03, 0.78, 0.94, 0.11), g["layers"])

    heatmap_panel(fig, (0.05, 0.55, 0.22, 0.19), edge_att, ROI_SHORT,
                 "Real final-layer edge attention (5×5)", cmap="Purples")
    heatmap_panel(fig, (0.30, 0.55, 0.22, 0.19), edge_gate, ROI_SHORT,
                 "Real edge gate g_ij (5×5)", cmap="Oranges")
    bar_panel(fig, (0.55, 0.55, 0.20, 0.19), ROI_SHORT, node_imp,
             "Real node importance\n(incoming attention mass)", color="#c0392b")
    text_panel(
        fig, (0.78, 0.55, 0.20, 0.19), "Reading these matrices",
        ["Attention: how much node j's message", "is weighted at node i (post-softmax).",
         "Gate: how much of that edge's message", "actually passes through (post-sigmoid,",
         "bias-initialized near-open at +1.0).", "Both are real per-edge outputs of the",
         "trained model for this subject."],
        fontsize=6.6, color=COLOR["gat"],
    )

    top_edges_table(fig, (0.03, 0.34, 0.94, 0.16), g["top_edges"])

    # Pathway diagram: top-5 edges only, with tract labels
    gax = fig.add_axes((0.08, 0.03, 0.55, 0.27))
    gax.set_xlim(-1.6, 1.6); gax.set_ylim(-1.1, 1.2); gax.axis("off")
    layout = {
        "Inferior_Frontal_Gyrus": (-1.0, 0.55), "Broca_Area": (-0.45, 0.15),
        "Insula": (0.0, -0.35), "Wernicke_Area": (0.6, 0.2),
        "Superior_Temporal_Gyrus": (1.0, -0.35),
    }
    short_of = dict(zip(ROI_FULL, ROI_SHORT))
    name_of = {v: k for k, v in {
        "Broca": "Broca_Area", "Wernicke": "Wernicke_Area", "Insula": "Insula",
        "IFG": "Inferior_Frontal_Gyrus", "STG": "Superior_Temporal_Gyrus",
    }.items()}
    for roi, short in zip(ROI_FULL, ROI_SHORT):
        x, y = layout[roi]
        gax.scatter([x], [y], s=750, color="#f2f5f9", edgecolors="#c0392b",
                   linewidths=1.6, zorder=3)
        gax.text(x, y, short, ha="center", va="center", fontsize=9, zorder=4,
                color=TEXT, fontweight="semibold")
    for e in g["top_edges"]:
        src = e["source"]; tgt = e["target"]
        if src not in layout or tgt not in layout:
            continue
        x0, y0e = layout[src]; x1, y1e = layout[tgt]
        gax.annotate("", xy=(x1, y1e), xytext=(x0, y0e), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                   alpha=0.85, linewidth=2.2 * e["attention"] * 10,
                                   shrinkA=19, shrinkB=19,
                                   connectionstyle="arc3,rad=0.15"))
        mx, my = (x0 + x1) / 2, (y0e + y1e) / 2
        tract = e["anatomical_tract"]
        if tract:
            gax.text(mx, my + 0.08, tract.replace("_", " "), fontsize=5.6,
                    color="#6b2d2d", ha="center", style="italic")
    gax.set_title("Real top-5 attended pathways, labelled with matched anatomical "
                  "tracts where available", fontsize=8.5, color=TEXT)

    text_panel(
        fig, (0.66, 0.03, 0.32, 0.27), "SAEG-GATv2 summary",
        ["Input: X*,A*,P,E (5×19, 5×5, 5×5, 5×5×3)",
         "2 layers, 4 heads/layer, hidden=64",
         "L1: in=19→out=64 (head_dim=16)",
         "L2: in=64→out=64 (head_dim=64)",
         "Readout: mean+max pool of 5 node",
         "embeddings → Z_G (128-d, since",
         "mean+max doubles hidden_dim=64)",
         "pooling_weights: not persisted in",
         "this record (honestly n/a, not",
         "invented)",
         "Z_G → Fusion (Figure 7)"],
        fontsize=7.4, color=COLOR["gat"],
    )

    footer(fig, "SAEG-GATv2: 2 layers x 4 heads, hidden_dim=64, "
          "graph_encoder.out_dim (Z_G, mean+max pooled) = 128")
    save(fig, "figure6_saeg_gatv2.png")


if __name__ == "__main__":
    main()
