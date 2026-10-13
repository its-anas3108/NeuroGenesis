"""Figure 8 — Explainability: SHAP attribution + composite ROI ranking, real data only."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    BORDER, COLOR, MUTED, ROI_SHORT, SID, TEXT,
    bar_panel, box, footer, heatmap_panel, load_record, save, text_panel,
)


def table_panel(fig, xywh, col_labels, rows, title, header_color):
    ax = fig.add_axes(xywh)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(header_color); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fbfcfe" if r % 2 else "#f4f2fb")
    ax.set_title(title, fontsize=8.0, color=TEXT, pad=8)
    return ax


def main() -> None:
    record = load_record()
    attr = record["explanations"]["attribution"]
    roi_rank = record["roi_ranking"]

    fig = plt.figure(figsize=(24, 22), facecolor="white", dpi=300)
    fig.suptitle("Figure 8 — Explainability: SHAP Attribution + Composite ROI Ranking",
                fontsize=18, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.5, 0.977,
             f"Subject {SID}  —  method: {attr['method']} "
             f"(signed={attr['signed']}, n_background={attr['n_background']}, "
             f"target_stage={attr['target_stage']})",
             ha="center", fontsize=10, color=MUTED)

    box(fig, (0.30, 0.925, 0.40, 0.04),
       "M16 Explainability — SHAP attribution + SAEG attention + SRVE vulnerability "
       "→ composite ROI ranking", COLOR["xai"], fontsize=8.5)

    values = np.array(attr["values"])
    ax = fig.add_axes((0.155, 0.68, 0.505, 0.20))
    vmax = float(np.abs(values).max()) or 1.0
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(5), ROI_SHORT, fontsize=8)
    ax.set_xticks(range(len(attr["feature_names"])), attr["feature_names"],
                  fontsize=6.0, rotation=55, ha="right")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if abs(v) > 1e-9:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if abs(v) > vmax * 0.5 else TEXT)
    ax.set_title(f"Real signed SHAP values (kernel SHAP, target class = "
                f"{attr['target_stage']}), 5 ROIs × 14 features", fontsize=8.5,
                color=TEXT, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=5.5)

    bar_panel(fig, (0.70, 0.68, 0.13, 0.20), ROI_SHORT,
             list(attr["per_roi"].values()), "Real |SHAP| mass\nper ROI",
             color="#8e44ad", fmt="{:.3f}")

    feat_items = sorted(attr["per_feature"].items(), key=lambda kv: -kv[1])[:8]
    bar_panel(fig, (0.85, 0.68, 0.13, 0.20), [k for k, _ in feat_items],
             [v for _, v in feat_items], "Real |SHAP| mass\nper feature (top 8)",
             color="#2f6feb", fmt="{:.3f}")

    tf_rows = [[f"{t['roi_short']}", t["feature"], f"{t['value']:.4f}"]
              for t in attr["top_features"]]
    table_panel(fig, (0.05, 0.50, 0.42, 0.14),
               ["ROI", "Feature", "Real SHAP value"], tf_rows,
               "Real top attributed (ROI, feature) pairs", COLOR["xai"])

    rr_rows = [[f"#{r['rank']}", r["roi_short"], f"{r['score']:.3f}",
               f"{r['vulnerability_normalized']:.3f}", f"{r['attention_normalized']:.3f}",
               f"{r['attribution_normalized']:.3f}"] for r in roi_rank]
    table_panel(fig, (0.52, 0.50, 0.44, 0.14),
               ["Rank", "ROI", "Composite\nscore", "SRVE vuln.\n(norm.)",
                "SAEG attn.\n(norm.)", "SHAP attrib.\n(norm.)"],
               rr_rows, "Real composite ROI ranking (M16, from per_subject_records.json)",
               COLOR["xai"])

    composite = {r["roi_short"]: r["score"] for r in roi_rank}
    bar_panel(fig, (0.05, 0.27, 0.28, 0.18), list(composite.keys()),
             list(composite.values()), "Real composite ROI score\n(vulnerability + "
             "attention + attribution)", color="#c0392b", fmt="{:.3f}")

    x = np.arange(5)
    ax2 = fig.add_axes((0.40, 0.27, 0.55, 0.18))
    w = 0.25
    ax2.bar(x - w, [r["vulnerability_normalized"] for r in roi_rank], width=w,
           label="SRVE vulnerability", color="#8e44ad")
    ax2.bar(x, [r["attention_normalized"] for r in roi_rank], width=w,
           label="SAEG attention", color="#c0392b")
    ax2.bar(x + w, [r["attribution_normalized"] for r in roi_rank], width=w,
           label="SHAP attribution", color="#2f6feb")
    ax2.set_xticks(x, [r["roi_short"] for r in roi_rank], fontsize=8)
    ax2.set_title("Real normalised component scores feeding the composite ROI ranking",
                  fontsize=8.5, color=TEXT)
    ax2.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    text_panel(
        fig, (0.05, 0.03, 0.90, 0.20), "How Figure 8 connects to the rest of the pipeline",
        ["SHAP (kernel SHAP, model-agnostic, real perturbation-based estimation with n_background=10 samples) "
         "explains the classifier's CN prediction in terms of the real 5x14 morphometric feature matrix from Figure 4.",
         "SAEG attention comes from the trained SAEG-GATv2 model (Figure 6, real node_importance).",
         "SRVE vulnerability comes from NeuroProp-X (Figure 5, real regional_vulnerability).",
         "The composite ROI ranking (table above, right) combines all three into a single per-subject, per-ROI "
         "importance score — this is the same computation that produces the dashboard's stage-wise ROI ranking table (M15).",
         "All values on this figure are read directly from outputs_oasis1_pilot31/roi_ranking/per_subject_records.json "
         "for this subject — nothing here is recomputed or approximated for display purposes."],
        fontsize=7.6, color=COLOR["xai"],
    )

    footer(fig, "SHAP: kernel method, signed, n_background=10, target_stage=CN "
          "(real reference stage for this subject)")
    save(fig, "figure8_explainability.png")


if __name__ == "__main__":
    main()
