# NeuroGenesis Architecture Figures — Reference Guide

This directory contains 8 publication-quality figures (300 DPI) documenting the
**complete, currently-implemented** NeuroGenesis pipeline, built entirely from
real project outputs for representative subject **OAS1_0001_MR1** (OASIS-1,
real cohort stage = CN), plus real cohort-wide statistics where noted.

**Scope decision (resolves a real conflict):** an earlier version of this task's
brief assumed a 3D-CNN branch producing 5×128 spatial features concatenated
with morphometric features into a 5×142 node representation. That branch was
**deliberately deleted from this codebase** earlier in this project's
development, at explicit user instruction ("drop the 3D cnn completely").
These figures diagram the **current, real, morphometry-only architecture** —
confirmed with the user before building — not the earlier CNN-inclusive spec.
No figure below shows a CNN branch, a Z_3D embedding, or 5×142-dim features,
because none of those exist in the running code.

**No-hallucination rule:** every number, matrix, image, and table cell in
these 8 figures is read directly from files under `outputs_oasis1_pilot31/`,
or is copied verbatim from the pipeline's own saved JSON records. Where a
real artifact does not exist on disk, the figure shows a labelled
**"ACTUAL OUTPUT NOT FOUND"** panel with the specific reason, rather than a
fabricated placeholder. The only genuinely-missing artifacts across all 8
figures are: (1) the raw pre-M1 OASIS-1 `.hdr`/`.img` source volume, no
longer present on this machine after the pipeline consumed it, and (2) M3's
five intensity-processing sub-steps as individual images — the pipeline
persists their numeric trace (shape/intensity/timing) but not per-step image
files. Both are called out explicitly in Figure 2.

---

## Section 1 — Artifact Inventory (representative subject: OAS1_0001_MR1)

| Pipeline stage | Actual output | File path (under `outputs_oasis1_pilot31/`) | Shape / dimension | Visualized in |
|---|---|---|---|---|
| M1 Load & header | Source NIfTI (removed post-run) | *not present on this machine* | 176×208×176, 1mm iso | Fig 1, 2 (labelled missing) |
| M1 metadata | `preprocessing/<sid>/<sid>_preprocessing.json` | real | shape/intensity/QC record | Fig 2 |
| M2 QC | QC report image | `preprocessing/<sid>/<sid>_qc_report.png` | — | Fig 1, 2 |
| M3 intensity processing | Numeric trace only (6 sub-steps, no images) | `preprocessing/<sid>/<sid>_preprocessing.json` → `stages[]` | 176×208×176 per step | Fig 2 |
| M4 skull stripping | Brain mask + panel | `preprocessing/<sid>/<sid>_brain_mask.nii.gz`, `_skull_stripping.png` | 176×208×176 | Fig 1, 2 |
| M4 skull-stripped volume | NIfTI | `preprocessing/<sid>/<sid>_skull_stripped.nii.gz` | 176×208×176 | — |
| M5 spatial standardization | Resampled volume + panel | `preprocessing/<sid>/<sid>_resampled_128iso.nii.gz`, `_resampling.png` | 128×128×128 | Fig 1, 2 |
| M6 ROI segmentation | Segmentation record | `roi/<sid>/<sid>_segmentation.json` | 5 ROIs (voxel count, volume, bbox, centroid) | Fig 3 |
| M6 ROI masks | Per-ROI NIfTI masks + overlays | `roi/masks/<sid>/<sid>_<roi>_mask.nii.gz`, `_overlay.png` | 128×128×128 each | Fig 1, 3 |
| M7 ROI patch extraction | Patch tensor | `roi/patches/<sid>/<sid>_roi_tensor.npy` | **(5, 48, 48, 48)** | Fig 1, 3 |
| M8 morphometric features | Feature table | `features/morphometric_features.csv` | **X: (5, 14)** real units | Fig 1, 4 |
| NeuroProp-X: SRVE | Regional vulnerability | `neuropropx/vulnerability/<sid>/<sid>_regional_vulnerability.{npy,csv}` | (5,) | Fig 1, 5, 8 |
| NeuroProp-X: AP-LAF | Anatomical prior / learned attention / fused adjacency | `neuropropx/adaptive_adjacency/<sid>/<sid>_A_{prior,att,star}.{npy,csv}` | (5,5) each | Fig 1, 5 |
| NeuroProp-X: ANP | Propagation matrix | `neuropropx/propagation/<sid>/<sid>_propagation_score.{npy,csv}` | (5,5) | Fig 1, 5 |
| NeuroProp-X: SAGR | Enriched node features + edge features + graph record | `neuropropx/enriched_graph/<sid>/<sid>_X_star.npy`, `_edge_features.npy`, `_G_star.json` | **X\*: (5, 19)**, E: (5,5,3) | Fig 1, 5 |
| SAEG-GATv2 | Attention / edge gate / node importance / per-layer config | `roi_ranking/per_subject_records.json` → `explanations.graph` | attention/gate: (5,5); 2 layers × 4 heads | Fig 1, 6 |
| Fusion + Classifier | Class probabilities | `roi_ranking/per_subject_records.json` → `class_probabilities` | 3-way (CN/MCI/AD) | Fig 1, 7 |
| Stage-TGT | Full propensity record | `roi_ranking/per_subject_records.json` → `stage_propensity_detail` | scalar + 3-way alignment | Fig 7 |
| Explainability: SHAP | Signed per-(ROI,feature) attribution | `roi_ranking/per_subject_records.json` → `explanations.attribution` | (5, 14) | Fig 8 |
| Explainability: composite ranking | Combined SRVE+attention+SHAP ranking | `roi_ranking/per_subject_records.json` → `roi_ranking` | 5 ROIs, ranked | Fig 8 |
| Cohort embeddings | Z_G / Z_H for all 30 subjects | `embeddings/z_g_z_h.npz` | Z_G: (30,128), Z_H: (30,64) | Fig 7 (via fig19) |
| Checkpoint | Trained model weights | `checkpoints/A7.pt` | 132,136 parameters | — |

---

## Section 2 — Verified Dimensions (from code, not assumed)

All dimensions below were cross-checked against the running code
(`modules/model.py`, `modules/m06_neuropropx/neuropropx_engine.py`,
`modules/m06_graph_learning/{attention,fusion}.py`) during this session —
code took precedence over any prior spec wherever the two disagreed.

| Quantity | Value | Verified against |
|---|---|---|
| Morphometric features (X) | 5 × 14 | `feature_spec.py`, `morphometric_features.csv` columns |
| NeuroProp-X enriched features (X\*) | 5 × **19** (not 14) | SAGR appends RV (1) + centrality (4) to X (14) |
| Adjacency / propagation matrices | 5 × 5 | `A_prior`, `A_att`, `A_star`, `P` |
| Edge features (E) | 5 × 5 × 3 | `neuropropx_engine.py` |
| SAEG-GATv2 layers | 2, 4 heads each | saved explanation record `layers[]` |
| SAEG-GATv2 layer 1 | in=19 → out=64, head_dim=16 | saved explanation record |
| SAEG-GATv2 layer 2 | in=64 → out=64, head_dim=64 | saved explanation record |
| Edge gate | `g_ij = σ(w_g^T E_ij + b_g)`, bias init **+1.0**, no renorm, self-loops always on | `modules/m06_graph_learning/attention.py` |
| Graph embedding (Z_G) | **128** (not 64) | `readout='mean_max'` concatenates mean+max pooling, doubling hidden_dim=64 |
| Fusion spatial_dim | **0** | 3D-CNN branch removed; `MultimodalFusion` handles this natively |
| Fused embedding (Z_H) | 64 | `fusion.out_dim` |
| Classifier | 3-way softmax (CN/MCI/AD) | — |
| Stage-TGT d_model | 64 | — |
| Total model parameters | 132,136 | loaded checkpoint |
| SHAP method | kernel SHAP, signed, n_background=10 | saved explanation record |

---

## Section 3 — Figure-by-Figure Summary

1. **`figure1_complete_architecture.png`** — full pipeline skeleton, M1→M19,
   with a curated subset of real thumbnails/matrices at each stage for
   orientation. Ends in a closing summary strip (real classifier probabilities
   + Stage-TGT output).
2. **`figure2_preprocessing_detail.png`** — M1–M5 in full detail: real header
   metadata, real QC report + metrics, the real M3 numeric trace (6 sub-steps,
   no images — labelled honestly), real skull-stripping and resampling panels
   with real voxel counts.
3. **`figure3_roi_localization.png`** — M6–M7 in full detail: real combined
   ROI overlay, real per-ROI statistics panel, all 5 real ROI overlays, all 5
   real 48³ patches (from `roi_tensor.npy`), and the real segmentation
   statistics table (voxel count, volume, bounding box, centroid per ROI).
4. **`figure4_morphometric_features.png`** — the full real 5×14 feature
   matrix in real units, 6 real per-feature bar charts across ROIs, and an
   explicit note that `atrophy_index` is real-but-NaN for this subject (no
   normative baseline for a cross-sectional scan — not fabricated).
5. **`figure5_neuropropx.png`** — all 4 NeuroProp-X sub-stages (SRVE, AP-LAF,
   ANP, SAGR) with real vulnerability, real A_prior/A_att/A_star/P matrices,
   real AP-LAF fusion coefficients, the real enriched X* matrix with its
   real component boundaries (morphometry/vulnerability/centrality), and a
   real edge-weighted graph rendering.
6. **`figure6_saeg_gatv2.png`** — real per-layer configuration and gate
   ranges, real final-layer attention and edge-gate matrices, real node
   importance, and the real top-5 attended pathways matched against
   anatomical tract names where the atlas dictionary has one.
7. **`figure7_fusion_classifier_stagetgt.png`** — real classifier
   probabilities, the real cohort-wide Z_G/Z_H PCA projection (reused from
   `fig19_embedding_projection.png`), the full real Stage-TGT output
   (stage-alignment distribution, advanced_stage_alignment,
   ad_associated_propensity, ad_axis_projection, stage_transition_propensity,
   transition_target_stage), and the Stage-TGT disclaimer reproduced
   **verbatim** from the saved record.
8. **`figure8_explainability.png`** — the real signed SHAP value matrix
   (5×14), real per-ROI and per-feature attribution mass, the real top
   attributed (ROI, feature) pairs, and the real composite ROI ranking that
   combines SRVE vulnerability + SAEG attention + SHAP attribution into the
   same ranking the dashboard's M15 page shows.
9. **`figure9_saeg_gatv2_dl_layers.png`** (bonus deep-dive) — a
   nested-box, deep-learning-style nomenclature diagram of the SAEG-GATv2
   layer, laid out to match a faculty-supplied reference image's visual
   language (grouped sub-panels, colored multi-head stacks, matrix-grid
   icons, math-rendered formulas), but with every formula, dimension and
   value drawn from the real code
   (`modules/m06_graph_learning/{attention,saeg_gatv2}.py`, real
   `GraphLearningConfig` defaults) instead of the reference's own content:
   real GATv2 dynamic score `e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)`, real
   adjacency+self-loop masking, real edge gate
   `g_ij = σ(w_g^T E_ij + b_g)` (highlighted as the one proposed departure
   from a textbook GATv2 layer), real residual+LayerNorm, and the real
   mean+max readout to Z_G (128-d). The bottom row walks the real data
   (X*, A*, this subject's real attention/gated-attention matrices, real
   Z_G/Z_H values) through Graph connectivity → Linear Projection →
   SAEG-GATv2 Graph Layer → Graph-Level Pooling → Multimodal Fusion, mirroring
   the reference's bottom-row structure. Panels showing values the pipeline
   does not persist to disk (e.g. per-node hidden states) are explicitly
   labelled "illustrative," never "real." The reference's GAT-v1-style score
   formula, its fabricated spatial-distance term `d_ij`, its CNN-embedding
   row, and its 5×142 node features do not appear anywhere, because none of
   them exist in this codebase.

---

## Section 4 — Validation Checklist

- [x] Every figure built from files under `outputs_oasis1_pilot31/`, read at
      build time — no hand-typed data values.
- [x] Every genuinely-missing artifact shown as a labelled
      "ACTUAL OUTPUT NOT FOUND" panel with a stated reason, never fabricated.
- [x] No 3D-CNN branch, Z_3D embedding, or 5×142 dimension appears anywhere —
      confirmed against the actual (post-deletion) codebase.
- [x] All dimensions cross-checked against code, not assumed from a prior spec.
- [x] Stage-TGT's own disclaimer is reproduced verbatim, not paraphrased.
- [x] Each figure visually inspected at full resolution after generation;
      layout bugs (title/subtitle collisions, text overflowing panel
      boundaries, axes-overlap, clipped tick labels) were found and fixed
      iteratively before acceptance.
- [x] One representative subject (OAS1_0001_MR1) traced consistently through
      all 8 figures; cohort-level statistics (embedding PCA) explicitly
      labelled as cohort-wide, not per-subject, where used.

## Reproducing these figures

```
cd architecture_figures
python build_figure1.py   # ... through build_figure8.py
```

Each script is self-contained (via `common.py`) and reads directly from
`../outputs_oasis1_pilot31/`. Re-running against a different checkpoint or
subject requires only re-running the OASIS-1 `--mode xai`/`--mode report`
pipeline first, then re-running these scripts — no code in this directory
recomputes model outputs itself.
