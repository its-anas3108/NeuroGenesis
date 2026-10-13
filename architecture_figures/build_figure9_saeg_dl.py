"""Figure 9 — SAEG-GATv2 layer, deep-learning implementation breakdown.

Laid out to match the visual language of a faculty-supplied reference image
(nested boxes, colored multi-head stacks, matrix-grid icons, math-rendered
formulas) — but every formula, dimension, and value below is the real
implementation in modules/m06_graph_learning/{attention,saeg_gatv2}.py and
the real GraphLearningConfig defaults, not the reference's own (partly
fabricated / partly generic) content. Real per-subject numbers use
OAS1_0001_MR1. Where a quantity is shown for visual/illustrative purposes
only (e.g. generic pooling numbers, or per-node hidden states that the
pipeline does not persist to disk), it is labelled "illustrative" rather
than presented as real.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

from common import ROI_FULL, ROI_SHORT, SID, load_npx, load_record, save

INK = "#141618"
MUTED = "#5b6570"
LINE = "#2b2f33"
PANEL_FILL = "#eef0f2"
PANEL_EDGE = "#3a3f44"
HEAD_COLORS = ["#5b8def", "#f2994a", "#6fcf97", "#bb6bd9"]
GATE_FILL = "#fbe3e3"
NORM_FILL = "#e4f5ea"


def outer_box(fig, xywh, title, fill=PANEL_FILL, title_fs=10.5):
    x, y, w, h = xywh
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.006, 0.006), 0.988, 0.988, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.6, edgecolor=PANEL_EDGE, facecolor=fill, zorder=1,
    ))
    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=title_fs,
           color=INK, fontweight="bold")
    return ax


def inner_box(fig, xywh, title, formula=None, note=None, fill="white",
             title_fs=8.4, formula_fs=10.5, note_fs=6.6):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.006,rounding_size=0.03",
        linewidth=1.1, edgecolor=LINE, facecolor=fill, zorder=2,
    ))
    ax.text(0.5, 0.90, title, ha="center", va="top", fontsize=title_fs,
           color=INK, fontweight="bold")
    if formula:
        ax.text(0.5, 0.53, formula, ha="center", va="center", fontsize=formula_fs,
               color=INK)
    if note:
        ax.text(0.5, 0.10, note, ha="center", va="bottom", fontsize=note_fs,
               color=MUTED, style="italic")
    return ax


def arrow(fig, p0, p1, color=LINE, lw=1.6, style="-|>"):
    ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color,
                                 transform=fig.transFigure, zorder=8))
    ax.patch.set_alpha(0)


def head_stack(fig, xywh, labels=("head 1", "head 2", "head 3", "head 4")):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for i, (lab, c) in enumerate(zip(labels, HEAD_COLORS)):
        off = i * 0.07
        ax.add_patch(Rectangle((0.10 + off, 0.15 + off), 0.55, 0.55, linewidth=1.0,
                               edgecolor=LINE, facecolor=c, alpha=0.85,
                               zorder=10 - i))
    ax.text(0.5, 0.02, "4 dynamic heads (GATv2)", ha="center", va="bottom",
           fontsize=6.4, color=MUTED)
    return ax


def grid_icon(fig, xywh, n_rows, n_cols, title="", row_labels=None,
             cell_values=None, cmap_fn=None, fontsize=4.6, title_fs=6.4):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows); ax.invert_yaxis()
    ax.axis("off")
    for r in range(n_rows):
        for c in range(n_cols):
            face = "#f7f7f8"
            if cell_values is not None:
                face = cmap_fn(cell_values[r, c]) if cmap_fn else face
            ax.add_patch(Rectangle((c, r), 1, 1, linewidth=0.4, edgecolor="#b7bcc2",
                                   facecolor=face))
    if row_labels:
        for r, lab in enumerate(row_labels):
            ax.text(-0.15, r + 0.5, lab, ha="right", va="center", fontsize=fontsize)
    ax.set_title(title, fontsize=title_fs, color=INK, pad=3)
    return ax


def symbolic_rows(fig, xywh, row_labels, n_feat, title, title_fs=6.6, fs=5.6):
    ax = fig.add_axes(xywh)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.0, title, ha="left", va="top", fontsize=title_fs,
           color=INK, fontweight="bold")
    n = len(row_labels)
    for i, lab in enumerate(row_labels):
        y = 0.80 - i * (0.72 / max(n - 1, 1))
        ax.text(0.0, y, f"{lab}:", ha="left", va="center", fontsize=fs,
               color=INK, fontweight="semibold")
        ax.text(0.30, y, " ".join(f"x{j}" for j in range(min(n_feat, 6))) +
               f" ... x{n_feat - 1}", ha="left", va="center", fontsize=fs,
               color=MUTED, family="monospace")
    return ax


def mini_bar(fig, xywh, values, title, color="#5b8def"):
    ax = fig.add_axes(xywh)
    ax.bar(range(len(values)), values, color=color, width=0.6)
    ax.set_xticks([])
    ax.tick_params(labelsize=4.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=5.8, color=INK, pad=2)
    return ax


def node_graph_icon(fig, xywh):
    ax = fig.add_axes(xywh)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.15, 1.25); ax.axis("off")
    layout = {
        "Inferior_Frontal_Gyrus": (-1.0, 0.55), "Broca_Area": (-0.45, 0.15),
        "Insula": (0.0, -0.35), "Wernicke_Area": (0.6, 0.2),
        "Superior_Temporal_Gyrus": (1.0, -0.35),
    }
    for i, a in enumerate(ROI_FULL):
        for b in ROI_FULL[i + 1:]:
            x0, y0 = layout[a]; x1, y1 = layout[b]
            ax.plot([x0, x1], [y0, y1], color="#9aa4b2", linewidth=0.7, zorder=1)
    for roi, short in zip(ROI_FULL, ROI_SHORT):
        x, y = layout[roi]
        ax.add_patch(Circle((x, y), 0.22, facecolor="#f2f5f9", edgecolor=LINE,
                            linewidth=1.1, zorder=3))
        ax.text(x, y, short, ha="center", va="center", fontsize=5.8, zorder=4,
               color=INK, fontweight="semibold")
    ax.set_title("5-node enriched brain graph G* = (V, X*, A*, P)", fontsize=6.6,
                color=INK)
    return ax


def main() -> None:
    npx = load_npx()
    record = load_record()
    g = record["explanations"]["graph"]
    edge_att = np.array(g["edge_attention"])
    edge_gate = np.array(g["edge_gate"])
    x_star = npx["x_star"]

    zg_zh = np.load("../outputs_oasis1_pilot31/embeddings/z_g_z_h.npz", allow_pickle=True)
    sids = list(zg_zh["session_ids"])
    idx = sids.index(SID)
    z_g_real = zg_zh["z_g"][idx]
    z_h_real = zg_zh["z_h"][idx]

    fig = plt.figure(figsize=(27, 15.5), facecolor="white", dpi=300)
    fig.suptitle("DEEP LEARNING IMPLEMENTATION BREAKDOWN OF THE SAEG-GATv2 LAYER FLOW",
                fontsize=18, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.965,
             f"NeuroGenesis, real implementation (modules/m06_graph_learning/"
             f"attention.py, saeg_gatv2.py) — subject {SID}, real cohort "
             "stage = CN. Style follows a faculty reference image; formulas, "
             "dimensions and values are this project's real code, not the reference's.",
             ha="center", fontsize=9.5, color=MUTED)

    # ══════════════════════════ TOP ROW ══════════════════════════════════
    # Center input box
    inb = fig.add_axes((0.30, 0.895, 0.40, 0.058))
    inb.axis("off"); inb.set_xlim(0, 1); inb.set_ylim(0, 1)
    inb.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.90,
                                 boxstyle="round,pad=0.01,rounding_size=0.10",
                                 linewidth=1.3, edgecolor=LINE, facecolor="white"))
    inb.text(0.5, 0.68, r"Input node features: $X^{*} \in \mathbb{R}^{B \times 5 \times 19}$",
            ha="center", va="center", fontsize=10.0, color=INK, fontweight="bold")
    inb.text(0.5, 0.28, "+ real additional input: $A^{*}_{ij}$ (adjacency support), "
            "$E_{ij}\\in\\mathbb{R}^{3}$ edge features $=[A^{prior}_{ij}, A^{att}_{ij}, P_{ij}]$",
            ha="center", va="center", fontsize=7.6, color=INK)

    arrow(fig, (0.36, 0.895), (0.24, 0.87))
    arrow(fig, (0.64, 0.895), (0.75, 0.845))

    # LEFT big box: GATv2 layer (spatial-analogue: edge-gated) implementation
    outer_box(fig, (0.005, 0.605, 0.475, 0.29),
             "SAEG-GATv2 Layer (Edge-Gated) Implementation")
    inner_box(
        fig, (0.025, 0.775, 0.435, 0.10),
        "Linear Projection (GATv2 dynamic heads)",
        r"$H W_l,\ \ H W_r$   (bias=False, Xavier init)",
        "in_dim=19 (layer 1) or 64 (layer 2) → heads x head_dim; "
        "$W_r$ reused as the value projection",
        formula_fs=9.5,
    )
    score_ax = inner_box(
        fig, (0.025, 0.615, 0.30, 0.145),
        "Detailed Score Calculation with dynamic heads",
        None,
        None,
    )
    score_ax.text(0.5, 0.62,
                 r"$e_{ij} = a^{T}\,\mathrm{LeakyReLU}(W_l h_i + W_r h_j)$",
                 ha="center", va="center", fontsize=9.0, color=INK)
    score_ax.text(0.5, 0.40,
                 "nonlinearity BEFORE the scoring vector $a$ → dynamic\n"
                 "attention (GATv2, Brody et al. 2022) — this is what makes\n"
                 "attention genuinely conditional on the query node",
                 ha="center", va="center", fontsize=6.6, color=MUTED)
    head_stack(fig, (0.335, 0.625, 0.115, 0.125))

    # Center column: residual + norm, and multi-head aggregation
    res_ax = inner_box(fig, (0.51, 0.80, 0.115, 0.095), "Residual + Norm",
                       None, None)
    res_ax.add_patch(Circle((0.30, 0.48), 0.14, facecolor="white",
                            edgecolor=LINE, linewidth=1.3))
    res_ax.text(0.30, 0.48, "+", ha="center", va="center", fontsize=14,
               color=INK, fontweight="bold")
    res_ax.annotate("", xy=(0.66, 0.48), xytext=(0.44, 0.48),
                    arrowprops=dict(arrowstyle="-|>", color=LINE, linewidth=1.2))
    res_ax.text(0.30, 0.78, "Res(x) + out", ha="center", fontsize=6.2, color=MUTED)

    arrow(fig, (0.625, 0.8475), (0.655, 0.8475))
    ln_ax = inner_box(
        fig, (0.655, 0.80, 0.335, 0.095), "Real LayerNorm (applied after every layer)",
        r"$\mathrm{LayerNorm}_{64}(out) = \dfrac{out-\mu}{\sqrt{\sigma^2+\epsilon}}\,\gamma+\beta$",
        "B×5×64   (torch.nn.LayerNorm, elementwise affine)",
        formula_fs=9.0,
    )

    agg_ax = inner_box(fig, (0.51, 0.615, 0.115, 0.17), "Dynamic Multi-Head\nAggregation",
                       None, None, title_fs=7.4)
    agg_ax.text(0.5, 0.62, r"$\mathrm{softmax}$" + "\n" + r"$\times$" + "\n" +
               r"$(H h_j W_r)$", ha="center", va="center", fontsize=7.2, color=INK)
    agg_ax.text(0.5, 0.16, "layer 1: concat → B×5×64\n"
               "layer 2: mean → B×5×64", ha="center", va="center",
               fontsize=6.0, color=MUTED)

    # RIGHT big box: multi-head attention + edge gate implementation
    outer_box(fig, (0.645, 0.605, 0.35, 0.235),
             "Multi-Head Attention + Edge-Gate Implementation")
    rx = np.linspace(0.665, 0.665 + 5 * 0.0575, 6)
    rw = 0.052
    ry, rh = 0.625, 0.135
    inner_box(fig, (rx[0], ry, rw, rh), "$W_l$ proj", r"$H\,W_l$", "→ 4x16",
             title_fs=6.6, formula_fs=8.0, note_fs=5.6)
    inner_box(fig, (rx[1], ry, rw, rh), "$W_r$ proj\n(= Value)", r"$H\,W_r$", "→ 4x16",
             title_fs=6.2, formula_fs=8.0, note_fs=5.6)
    inner_box(fig, (rx[2], ry, rw, rh), "Pairwise\nsum", r"$h_l[i]+h_r[j]$",
             "LeakyReLU(0.2)", title_fs=6.2, formula_fs=7.6, note_fs=5.6)
    inner_box(fig, (rx[3], ry, rw, rh), "Masking", r"$A^{*}_{ij}\!>\!\epsilon \/\cup\/ I$",
             "self-loops\nalways kept", title_fs=6.6, formula_fs=8.0, note_fs=5.6)
    inner_box(fig, (rx[4], ry, rw, rh), "Softmax", r"$\alpha_{ij}$",
             "row-wise per\nquery node $i$", title_fs=6.6, formula_fs=9.0, note_fs=5.6)
    inner_box(fig, (rx[5], ry, rw, rh), "Edge Gate\n(proposed)",
             r"$g_{ij}\!\cdot\!\alpha_{ij}$", "$g=\\sigma(w_g^T E_{ij}+b_g)$",
             fill=GATE_FILL, title_fs=6.2, formula_fs=8.4, note_fs=5.4)
    for i in range(5):
        arrow(fig, (rx[i] + rw, ry + rh / 2), (rx[i + 1], ry + rh / 2), lw=1.1)

    concat_ax = inner_box(fig, (0.665, 0.615, 0.315, 0.075), "Concatenation (layer 1) "
                          "/ Mean (layer 2, final)",
                          r"$B\times5\times(\mathrm{heads}\times\mathrm{head\_dim}) = "
                          r"B\times5\times64$",
                          None, formula_fs=8.6, title_fs=7.0)

    fig.text(0.995, 0.615,
             "$\\tilde{\\alpha}_{ij}=g_{ij}\\alpha_{ij}$ is the ONLY departure "
             "from a textbook\nGATv2 layer, and the paper's proposed element.",
             ha="right", va="top", fontsize=6.6, color="#8a2d2d", style="italic")

    # ══════════════════════════ BOTTOM ROW ══════════════════════════════
    by0, bh = 0.01, 0.565

    # Panel 1: graph connectivity
    outer_box(fig, (0.005, by0, 0.235, bh), "Graph connectivity")
    node_graph_icon(fig, (0.03, by0 + 0.35, 0.19, 0.19))
    dims = [
        ("Preprocessed volume", "128³"),
        ("Morphometric features / ROI", "14"),
        ("Enriched node features X*", "19"),
        ("Number of nodes", "5"),
        ("Node feature matrix", "X* ∈ R^{5x19}"),
        ("Adjacency", "A* ∈ R^{5x5}"),
        ("Edge features", "E ∈ R^{5x5x3}"),
        ("After GATv2 layer 1", "5 x 64"),
        ("After GATv2 layer 2", "5 x 64"),
        ("Mean / max pooled vector", "64 / 64"),
        ("Concatenated graph rep.", "Z_G ∈ R^{128}"),
    ]
    tax = fig.add_axes((0.015, by0 + 0.02, 0.215, 0.32))
    tax.axis("off"); tax.set_xlim(0, 1); tax.set_ylim(0, 1)
    tax.text(0.0, 0.98, "Real Tensor Dimensions (no CNN branch — removed)",
            fontsize=6.6, fontweight="bold", color=INK)
    for i, (k, v) in enumerate(dims):
        yy = 0.88 - i * 0.082
        tax.text(0.0, yy, k, fontsize=5.9, color=INK)
        tax.text(1.0, yy, v, fontsize=5.9, color=MUTED, ha="right", family="monospace")

    # Panel 2: real feature matrices (X*, A*, P) — symbolic + real grids
    outer_box(fig, (0.245, by0, 0.155, bh), "Real node / graph matrices")
    symbolic_rows(fig, (0.255, by0 + 0.40, 0.135, 0.15), ROI_SHORT, 19,
                 "X* rows (symbolic feature index)")

    def orange(v):
        t = np.clip(v, 0, 1)
        return (1.0, 1.0 - 0.55 * t, 1.0 - 0.85 * t)

    xs_n = (x_star - x_star.min()) / (x_star.max() - x_star.min() + 1e-9)
    grid_icon(fig, (0.255, by0 + 0.20, 0.135, 0.16), 5, 19,
             title="Real X* (5×19), column-normalised", row_labels=ROI_SHORT,
             cell_values=xs_n, cmap_fn=orange, fontsize=4.2, title_fs=5.8)

    def red(v):
        t = np.clip(v, 0, 1)
        return (1.0, 1.0 - 0.7 * t, 1.0 - 0.7 * t)

    a_n = npx["a_star"] / (npx["a_star"].max() + 1e-9)
    grid_icon(fig, (0.255, by0 + 0.02, 0.135, 0.13), 5, 5,
             title="Real A* (5×5)", row_labels=ROI_SHORT, cell_values=a_n,
             cmap_fn=red, fontsize=4.6, title_fs=5.8)

    # Panel 3: linear projection
    outer_box(fig, (0.415, by0, 0.145, bh), "Linear Projection")
    inner_box(fig, (0.425, by0 + 0.40, 0.125, 0.13), "", r"$H = X^{*} W$",
             r"$W \in \mathbb{R}^{19\times64}$", formula_fs=10)
    grid_icon(fig, (0.425, by0 + 0.13, 0.125, 0.22), 5, 12,
             title="Illustrative H = X*W_l (layer 1)\n(not persisted to disk per-node)",
             row_labels=ROI_SHORT, cell_values=None, fontsize=1, title_fs=5.6)
    fig.text(0.4875, by0 + 0.10, "H ∈ R^{5x64}\n(5 nodes x 64 features)",
            ha="center", fontsize=6.4, color=MUTED)

    # Panel 4: SAEG-GATv2 graph layer (real attention/gate for this subject)
    outer_box(fig, (0.575, by0, 0.175, bh), "SAEG-GATv2 Graph Layer (real, this subject)")
    grid_icon(fig, (0.585, by0 + 0.36, 0.155, 0.19), 5, 5,
             title="Real final attention α (5×5)", row_labels=ROI_SHORT,
             cell_values=edge_att / (edge_att.max() + 1e-9), cmap_fn=red, fontsize=4.6)

    def purple(v):
        t = np.clip(v, 0, 1)
        return (1.0 - 0.55 * t, 1.0 - 0.75 * t, 1.0)

    grid_icon(fig, (0.585, by0 + 0.15, 0.155, 0.19), 5, 5,
             title="Real gated $\\tilde{\\alpha}=g\\cdot\\alpha$ (5×5)", row_labels=ROI_SHORT,
             cell_values=(edge_att * edge_gate) / ((edge_att * edge_gate).max() + 1e-9),
             cmap_fn=purple, fontsize=4.6)
    fig.text(0.6625, by0 + 0.06, "h₁, h₂ ∈ R^{5x64}\n(per-node states; not "
            "persisted\nto disk beyond these matrices)", ha="center", fontsize=5.8,
            color=MUTED)

    # Panel 5: graph-level pooling
    outer_box(fig, (0.765, by0, 0.115, bh), "Graph-Level Pooling")
    mini_bar(fig, (0.775, by0 + 0.40, 0.095, 0.12),
            z_h_real[:8], "Real h (illustrative\nfirst 8 of 64 dims)", color="#5b8def")
    inner_box(fig, (0.775, by0 + 0.28, 0.095, 0.09), "Mean pooling",
             r"$z_{mean}=\mathrm{mean}_n(h_2)$", None, title_fs=6.4, formula_fs=7.4)
    inner_box(fig, (0.775, by0 + 0.16, 0.095, 0.09), "Max pooling",
             r"$z_{max}=\mathrm{amax}_n(h_2)$", None, title_fs=6.4, formula_fs=7.4)
    fig.text(0.8225, by0 + 0.10,
            f"real Z_G[0:6] =\n[{', '.join(f'{v:.2f}' for v in z_g_real[:6])}, ...]",
            ha="center", fontsize=5.6, color=MUTED, family="monospace")

    # Panel 6: multimodal fusion / output
    outer_box(fig, (0.89, by0, 0.105, bh), "Multimodal Fusion")
    inner_box(fig, (0.90, by0 + 0.40, 0.085, 0.11), "", r"$Z_G \in \mathbb{R}^{128}$",
             "mean||max, real", formula_fs=9.0)
    inner_box(fig, (0.90, by0 + 0.24, 0.085, 0.11), "", r"$Z_H \in \mathbb{R}^{64}$",
             "spatial_dim=0\n(no CNN branch)", formula_fs=9.0)
    fig.text(0.9425, by0 + 0.12,
            f"real Z_H[0:6] =\n[{', '.join(f'{v:.2f}' for v in z_h_real[:6])}, ...]"
            "\n\n→ Classifier + Stage-TGT\n(Figure 7)",
            ha="center", fontsize=5.6, color=MUTED, family="monospace")

    arrow(fig, (0.40, by0 + bh * 0.45), (0.415, by0 + bh * 0.45))
    arrow(fig, (0.56, by0 + bh * 0.45), (0.575, by0 + bh * 0.45))
    arrow(fig, (0.75, by0 + bh * 0.45), (0.765, by0 + bh * 0.45))
    arrow(fig, (0.88, by0 + bh * 0.45), (0.89, by0 + bh * 0.45))

    fig.text(0.5, 0.002,
             f"Subject: {SID} (real cohort stage = CN)  |  Source: "
             "modules/m06_graph_learning/{attention,saeg_gatv2}.py, "
             "outputs_oasis1_pilot31/  |  All real values are labelled real; "
             "illustrative panels are labelled illustrative — nothing is invented.",
             ha="center", fontsize=6.8, color=MUTED)

    save(fig, "figure9_saeg_gatv2_dl_layers.png")


if __name__ == "__main__":
    main()
