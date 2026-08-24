# NeuroGenesis — Migration Plan

**Status:** authoritative refactor plan. Produced after full inspection of the pre-refactor
repository (Section 36 requirement). Records what was found, what is kept, what moves to
legacy, and what is newly built.

---

## 1. Pre-refactor inventory (measured, not assumed)

### 1.1 Two overlapping implementations

The repository contained two parallel stacks:

| Stack | Files | Verdict |
|---|---|---|
| Top-level packages: `preprocessing/`, `segmentation/`, `features/`, `graph/`, `visualization/`, `dataset/` | ~200 KB | **Real, detailed, working implementations.** Driven by `main.py`. **KEEP.** |
| `backend/modules/**` | ~90 KB | Thin duplicates plus heuristic stubs. **Supersede.** |

`backend/modules/06_neuropropx/engine.py` and `backend/modules/neuropropx/engine.py` were two
divergent copies of the same `NeuroPropXEngine` class.

### 1.2 Defects found

| Location | Defect |
|---|---|
| `dashboard/app.py:91` | `SyntaxError` — page dispatch begins with `elif`, no preceding `if`. Dashboard could not launch. |
| `dashboard/app.py` | Two conflicting page-name sets; pages "8/9/10/11" defined twice, second set unreachable. |
| `backend/modules/dataset/__init__.py:8` | `SyntaxError` — `from backend.modules["01_dataset"]...`. Numeric-prefixed dirs are not importable. |
| `main.py:891` | `importlib.import_module("backend.modules.06_neuropropx.engine")` string-loading hack, forced by the above. |

### 1.3 Fabricated outputs found (removed from the active system)

These violate Section 28 and are **not** carried forward into the active experiment:

| Location | Problem |
|---|---|
| `backend/modules/08_xai/explainable_ai.py:60` | Comment reads *"Synthetic SHAP value generation"*. Invents features `"Naming Accuracy Index"`, `"Speech Fluency Score"` that do not exist in OASIS-1. |
| `main.py:922` | `SpeechAssessmentScores(88.0, 86.0, 84.0, 87.0, 86.25)` — hard-coded fake speech assessment scores fed into NeuroProp-X. |
| `backend/modules/06_neuropropx/temporal_transformer.py` | Closed-form atrophy extrapolation named "Temporal Graph Transformer". No attention is learned; no transformer present. |
| `backend/modules/06_neuropropx/engine.py` | `compute_drve` / `compute_anpe` are hand-coded arithmetic with magic constants (`NORMATIVE_VOLUMES`, `0.6/0.4`, `50/(dist+10)`). **Zero learnable parameters.** |

Consequence: SRVE / AP-LAF / ANP are **genuinely rewritten** as parameterised PyTorch modules.
The old engine is not adapted, because there was nothing learnable to adapt.

### 1.4 Assets worth preserving

- `graph/graph_builder.py:114` — `ANATOMICAL_EDGES`: 13 directed edges over the 5 speech ROIs with
  named white-matter tracts (superior longitudinal fasciculus, short arcuate fibres, extreme
  capsule system, inferior fronto-occipital fasciculus). This becomes `A_prior`.
  **Provenance caveat:** the source comment claims *"normalised DTI fractional anisotropy values
  from literature"* but supplies no citation. It is therefore documented as **expert-assigned
  ordinal tract weights** reflecting established speech-network anatomy, not as measured DTI FA.
  See `modules/m05_graph_construction/anatomical_prior.py` for the full provenance note.
- `features/feature_extractor.py` — 13 real per-ROI morphometric/textural features. Kept.
  `fractal_dimension` was a `NaN` placeholder and is dropped rather than faked.
- The full `preprocessing/` chain (QC, N4, WM-KDE, CLAHE, anisotropic diffusion, skull strip,
  resample, ROI crop) and `segmentation/roi_extraction.py` (Harvard-Oxford). Kept unchanged,
  wrapped behind clean interfaces.

### 1.5 Environment and data reality

Measured on this machine:

- `torch 2.8.0+cpu` **available**. `torch_geometric` absent — and not needed (decision D3).
- **Absent:** `nibabel`, `nilearn`, `SimpleITK`, `scikit-image`, `shap`, `statsmodels`.
  The imaging half of the pipeline cannot execute until these are installed.
- `streamlit` present but **broken** by a `starlette` incompatibility
  (`DEFAULT_EXCLUDED_CONTENT_TYPES` import failure).
- **`dataset/OASIS/` is empty — zero MRI files.** PNGs under `output/` are residue from a run
  on another machine.
- `dataset/oasis_cross-sectional.csv` **is** present and is the real OASIS-1 cross-sectional
  table: 436 rows, 416 unique subjects (20 `MR2` rescans).

Measured OASIS-1 label distribution:

| CDR | Stage | Sessions |
|---|---|---|
| 0.0 | CN | 135 |
| 0.5 | MCI | 70 |
| 1.0 | AD | 28 |
| 2.0 | AD | 2 |
| NaN | unassessed (young cohort) | 201 |

Labeled cohort: **235 sessions, CN 135 / MCI 70 / AD 30.** AD n=30 is the binding constraint on
every claim the study can make.

---

## 2. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| D1 | Build the complete system now; every metric surface renders `Not available / requires training data` until real MRI is present. | Section 28 forbids inventing outputs. Structure and code can be fully validated on synthetic phantoms with real CSV labels. |
| D2 | Exclude the 201 CDR-unassessed sessions. Cohort = 235 labeled sessions. An optional configurable age filter (`min_age`) supports a stricter age-matched variant. | Those subjects are 18-40 and were never clinically assessed. Labeling them CN would let the model separate stages by age rather than pathology. |
| D3 | Graph learning in **pure dense PyTorch**, no `torch_geometric`. | Graphs are fixed 5-node dense. PyG's sparse machinery adds a fragile Windows dependency and buys nothing, while dense tensors make attention/edge-gate matrices directly inspectable for the Section 53-56 dashboard panels. GATv2 attention implemented explicitly per Brody et al., 2022. |
| D4 | Directories use importable names carrying the Section 22 numbers: `m01_dataset`, `m06_neuropropx`, `m07_stage_tgt`, ... | Python cannot import an identifier starting with a digit. This is the exact defect that broke `backend/modules/dataset/__init__.py` and forced `importlib` hacks in `main.py`. Numbering is preserved for faculty readability. |

---

## 3. Target layout

```
modules/
  common/            config, seeds, logging, paths, ROI constants, run-state tracker
  m01_dataset/       label mapping, cohort build, subject-wise splits
  m02_preprocessing/ wrapper over existing preprocessing/  (KEEP)
  m03_segmentation/  wrapper over existing segmentation/    (KEEP)
  m04_feature_extraction/  reuse features/ + atrophy index + train-only scaler
  m05_graph_construction/  A_prior + subject graph
  m06_spatial_encoder/     cnn3d.py, patch_dataset.py                    (NEW)
  m06_neuropropx/          srve, ap_laf, anp, sagr, engine, types        (REWRITTEN)
  m06_graph_learning/      gat_baseline, gatv2_baseline, saeg_gatv2,
                           fusion, classifier, model                     (NEW)
  m07_stage_tgt/           stage_prototypes, stage_transformer,
                           propensity_head                               (NEW)
  m08_xai/                 SHAP, attention, vulnerability, ROI ranking   (REWRITTEN)
  m09_statistics/          stage-wise tests, FDR, effect sizes           (NEW)
  m10_results/             Tables 1-9, figures                           (NEW)
  m11_report/              report generator                              (REWRITTEN)
  m12_future_extensions/
      temporal_longitudinal/   <- old temporal_transformer.py  (LEGACY, quarantined)
      digital_twin/            <- old digital_twin.py          (LEGACY, quarantined)
  training/                losses, trainer, ablation, evaluate, baselines
  inference/               single-subject pipeline
run.py                     --mode {preprocess,train_cnn,train_graph,train_full,
                                   ablation,evaluate,xai,report,dashboard}
dashboard/                 M1-M19 pipeline inspection app
```

## 4. Disposition of every existing file

| Existing path | Action |
|---|---|
| `preprocessing/*` (6 modules) | **KEEP unchanged.** Wrapped by `modules/m02_preprocessing`. |
| `segmentation/roi_extraction.py` | **KEEP unchanged.** Wrapped by `modules/m03_segmentation`. |
| `features/feature_extractor.py` | **KEEP**, extended with a documented atrophy index. |
| `graph/graph_builder.py` | **KEEP.** `ANATOMICAL_EDGES` / `NODE_METADATA` become the canonical prior. |
| `visualization/visualize.py` | **KEEP.** Retitled figures: "stage-wise", never "longitudinal atrophy". |
| `dataset/oasis1_loader.py` | **KEEP.** Now feeds `m01_dataset` label mapping. |
| `dataset/generate_synthetic_oasis.py` | **KEEP** — explicitly marked smoke-test only. |
| `main.py` | **SUPERSEDED** by `run.py`. Preprocessing stage logic ported; `stage_neuropropx` deleted (fabricated inputs). |
| `backend/modules/06_neuropropx/engine.py` | **SUPERSEDED** by `modules/m06_neuropropx/*`. Archived under `m12_future_extensions/`. |
| `backend/modules/neuropropx/engine.py` | **DELETE** — duplicate of the above. |
| `backend/modules/dataset/__init__.py` | **DELETE** — syntactically invalid. |
| `backend/modules/06_neuropropx/temporal_transformer.py` | **MOVE** to `m12_future_extensions/temporal_longitudinal/`, quarantined behind an explicit "not part of the validated experiment" guard. |
| `backend/modules/07_digital_twin/digital_twin.py` | **MOVE** to `m12_future_extensions/digital_twin/`, same quarantine. |
| `backend/modules/08_xai/explainable_ai.py` | **DELETE** — synthetic SHAP. Replaced by real `modules/m08_xai`. |
| `backend/modules/09_report/clinical_report.py` | **REWRITTEN** as `modules/m11_report`. |
| `backend/modules/01_dataset/oasis3_manager.py` | **MOVE** to future extensions. OASIS-3 is not assumed available. |
| `backend/modules/0{1..5,7}_*` remainder | **DELETE** — duplicates of superior top-level equivalents. |
| `dashboard/app.py` | **REWRITTEN** — currently a `SyntaxError`. New M1-M19 inspection app. |
| `output/*.png` | **DELETE** — stale artifacts from another machine, would masquerade as current results. |

## 5. Execution order with validation gates

Each step is verified before the next begins (Section 36).

1. `common/` scaffolding — config, seeds, paths, ROI constants, run-state tracker.
2. `m01_dataset` — real label mapping over the real CSV; assert CN/MCI/AD = 135/70/30.
3. Legacy quarantine — move digital twin / longitudinal TGT / OASIS-3 out of the active path.
4. `m04` scaler + atrophy index; `m05` `A_prior`.
5. `m06_spatial_encoder` — CNN3D. **Checkpoint 3**: five 128-d embeddings from a `(5,48,48,48)` tensor.
6. `m06_neuropropx` — SRVE, AP-LAF, ANP, SAGR. **Checkpoint 2**: `G -> G*`.
7. `m06_graph_learning` — SAEG-GATv2, fusion, classifier. **Checkpoints 4, 5**.
8. `m07_stage_tgt` — prototypes, transformer, propensity. **Checkpoint 6**.
9. `training/` — losses, subject-wise trainer, evaluation, baselines, ablation. **Checkpoint 9**.
10. `m08_xai` — real SHAP + real attention. **Checkpoints 7, 8**.
11. `m09_statistics`, `m10_results`, `m11_report`.
12. `run.py` modes; `inference/` single-subject JSON.
13. Dashboard M1-M19. **Checkpoint 10**.
14. README rewrite.
15. `modules/m02`, `m03` imaging wrappers + **Checkpoint 1** (blocked on `nibabel`/`nilearn` install and on MRI data).

## 6. Known blockers carried forward

| Blocker | Effect |
|---|---|
| No MRI files in `dataset/OASIS/` | Checkpoint 1 cannot run on real data. No real accuracy/F1/AUC can be produced. All metric surfaces gate to "requires training data". |
| `nibabel`, `nilearn`, `SimpleITK`, `scikit-image` not installed | Imaging modules importable but not executable here. Listed in `requirements.txt`. |
| `streamlit` / `starlette` conflict | Dashboard code is written and syntax-verified, but cannot be launched here until the conflict is resolved. Pin recorded in `requirements.txt`. |
| `shap`, `statsmodels` not installed | XAI/statistics degrade gracefully with an explicit "dependency missing" surface rather than a fake number. |
| AD n=30 | Every ablation and comparison must report repeated stratified evaluation with CIs; single-split point estimates are not reportable. |

---

## 7. Completion status

Refactor complete. Recorded here so the plan documents what was actually
achieved rather than only what was intended.

### Validation checkpoints (Section 35)

Run with `python tools/validate_checkpoints.py --outputs outputs_smoke`.

| # | Checkpoint | Result |
|---|---|---|
| 1 | MRI -> preprocessing -> 5 ROIs -> features -> graph | **BLOCKED** — `nibabel`, `SimpleITK`, `nilearn`, `scikit-image` absent; `dataset/OASIS/` empty |
| 2 | graph -> NeuroProp-X -> G* | **PASS** — `X*=(B,5,147)`, `A*=(B,5,5)`, `P=(B,5,5)` |
| 3 | 3D CNN -> five ROI embeddings | **PASS** — `(1,5,128)`, 78,736 params, 6 traced stages, deterministic |
| 4 | G* -> SAEG-GATv2 -> CN/MCI/AD | **PASS** — `Z_G=(B,128)`, edge gate active |
| 5 | Fusion of CNN + graph | **PASS** — `Z_3D(128) + Z_G(128) -> Z_F(256) -> Z_H(64)` |
| 6 | Stage-TGT propensity, no longitudinal data | **PASS** — 4 tokens; AD reference correctly reports `null` |
| 7 | ROI ranking and stability | **PASS** — weights renormalise when a signal is absent |
| 8 | Attribution and attention explanations | **PASS** — permutation backend correctly *not* labelled SHAP |
| 9 | Ablation ladder A0-A7 | **PASS** — 8 variants; paired Wilcoxon comparison |
| 10 | Dashboard renders all M1-M19 | **BLOCKED** — `streamlit`/`starlette` conflict. All 22 pages verified against a stubbed Streamlit on both a populated and an empty outputs tree; the data layer and all 23 module rows pass |

**8 PASS, 2 BLOCKED, 0 FAIL.** Both blocks are absent environment packages, not
code defects. A blocked checkpoint is never reported as a pass.

### Invariant tests

`python tests/test_invariants.py` — **24/24 pass**, covering subject-wise split
integrity with two sessions per subject, train-only scaler fitting, train-only
class weights, undefined-vs-zero metric handling, CI clamping, report language
discipline, figure-title discipline, attribution labelling, FDR monotonicity,
ROI-order and checkpoint guards, ablation shape invariance, edge-gate/ANP
coupling, gradient reachability, propensity bounds and anchoring, legacy-import
quarantine, config validation, run-state integrity, and the real OASIS-1 label
counts (135/70/30).

### Defects found and fixed during the refactor

Each was found by running the code, not by inspection:

1. **`SAEGGATv2.out_dim` disagreed with the actual output width.** The
   head-combination bookkeeping assumed an averaging layer emits
   `hidden // heads`; it emits `hidden`. Fixed in both the proposed encoder and
   the baselines.
2. **The prototype temperature was a dead parameter.** No loss term consumed the
   prototype similarities, so a trainable temperature had no gradient path. Made
   a buffer, with the reason documented.
3. **The AD-associated propensity was inverted.** The CN-to-AD axis projection is
   sensitive to a common-mode offset of the representation cloud; on a trained
   model it returned 0.87 for CN and 0.69 for AD. Replaced with the
   offset-invariant ratio `d_CN / (d_CN + d_AD)`; the projection is retained as a
   geometry diagnostic.
4. **The prototype losses were scale-suppressed.** `L_proto` and `L_order` divide
   by `d_model`, so the initial weights (0.10 / 0.05) left prototypes ~8.5 from
   their own class centroids while being only ~5.1 apart. Raised to 1.00 / 0.50.
5. **Model selection could pick an epoch with untrained stage geometry.**
   Validation balanced accuracy saturated at epoch 4 while `L_order` was still
   1.88. Added a tie-break that prefers better geometry only when the monitored
   metric is *equal*, so nothing is traded away. Best epoch moved 4 -> 24 and the
   CN < MCI < AD ordering became satisfied.
6. **Permutation attribution tried to allocate 24 GB.** It re-ran the 3D CNN for
   all 700 perturbed rows. The CNN embedding is constant during morphometric
   attribution, so it is now computed once per subject and the forward passes are
   chunked.
7. **The report language guard rejected its own disclaimers.** A substring ban on
   "conversion probability" also caught "not a validated conversion probability".
   Made negation-aware, with sentence-boundary scoping so a negation in a prior
   sentence cannot license the next assertion.
8. **Confidence intervals exceeded 1.0.** A normal approximation on a bounded
   metric produced a macro-F1 upper bound of 1.047. Now clamped, with
   `ci_clamped` recorded and surfaced in Table 9.
9. **`LogisticRegression(multi_class=...)`** was removed in scikit-learn 1.8.
10. **M9-M11.4 reported `NOT_STARTED` despite executing.** They run inside one
    forward pass; their state is now recorded from the produced tensors.

### Blockers still outstanding

| Blocker | Effect | Resolution |
|---|---|---|
| No MRI under `dataset/OASIS/` | Checkpoint 1 blocked; no real metric can be produced | Place OASIS-1 T1 volumes there |
| `nibabel`, `SimpleITK`, `nilearn`, `scikit-image` absent | M1-M8 importable but not executable | `pip install -r requirements.txt` |
| `streamlit` / `starlette` conflict | Dashboard verified but not launchable here | `pip install --upgrade 'streamlit>=1.40' 'starlette>=0.40'` |
| `shap` absent | Attribution uses the permutation backend, labelled as such | `pip install shap` |
| `xgboost` absent | Gradient-boosting baseline substituted, labelled as such | `pip install xgboost` |
| AD n=30 | Binds every claim; single-split estimates not reportable | Inherent to OASIS-1 |
