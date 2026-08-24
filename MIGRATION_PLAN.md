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

Refactor complete, and the outstanding environment blockers have since been
cleared. Recorded here so the plan documents what was actually achieved.

### Validation checkpoints (Section 35)

    python tools/validate_checkpoints.py --outputs outputs_imaging         --mri-dir dataset/OASIS_synthetic

| # | Checkpoint | Result |
|---|---|---|
| 1 | MRI -> preprocessing -> 5 ROIs -> features -> graph | **PASS** — full imaging chain on phantom volumes |
| 2 | graph -> NeuroProp-X -> G* | **PASS** — `X*=(B,5,147)`, `A*`, `P` |
| 3 | 3D CNN -> five ROI embeddings | **PASS** — `(1,5,128)`, 78,736 params, deterministic |
| 4 | G* -> SAEG-GATv2 -> CN/MCI/AD | **PASS** — edge gate active |
| 5 | Fusion of CNN + graph | **PASS** — `Z_3D(128)+Z_G(128) -> Z_F(256) -> Z_H(64)` |
| 6 | Stage-TGT propensity, no longitudinal data | **PASS** — AD reference correctly `null` |
| 7 | ROI ranking and stability | **PASS** |
| 8 | Attribution and attention explanations | **PASS** — genuine Kernel SHAP |
| 9 | Ablation ladder A0-A7 | **PASS** |
| 10 | Dashboard renders all M1-M19 | **PASS** — Streamlit serves, health endpoint 200 |

**10 PASS, 0 BLOCKED, 0 FAIL.**

On the smoke-patch tree (`--outputs outputs_smoke`) checkpoint 1 correctly
reports BLOCKED, because that tree is generated from the ROI-patch stage onward
and has no imaging of its own.

### Invariant tests

`python tests/test_invariants.py` — **27/27 pass**.

### Environment blockers: cleared

| Blocker | Resolution |
|---|---|
| `nibabel`, `SimpleITK`, `nilearn`, `scikit-image` absent | Installed. M1-M8 now execute. |
| `streamlit` / `starlette` conflict | Resolved by upgrading `starlette` to >= 1.6 and `anyio` to >= 4. Streamlit 1.60 imports and serves. |
| `shap` absent | Installed. Attribution now uses genuine Kernel SHAP. |
| `xgboost` absent | Installed. |
| No MRI in `dataset/OASIS/` | **Still outstanding.** Phantom volumes in `dataset/OASIS_synthetic/` exercise the code path; real OASIS-1 T1s are still required for any result. |

> **Environment side effect.** Upgrading `starlette` broke the pin of a
> pre-existing standalone `fastapi 0.104.1` (it requires `starlette < 0.28`).
> `fastapi` still imports and nothing depends on it (`pip show fastapi` lists no
> `Required-by`), but `pip check` reports the conflict. Restore with
> `pip install "starlette>=0.27,<0.28"` if that fastapi install matters, at the
> cost of the dashboard.

### Defects found and fixed

Each was found by running the code, not by inspection.

**Found during the original refactor:**

1. `SAEGGATv2.out_dim` disagreed with the actual output width.
2. The prototype temperature was a dead parameter.
3. The AD-associated propensity was inverted by a common-mode offset; replaced
   with the offset-invariant `d_CN / (d_CN + d_AD)`.
4. `L_proto` and `L_order` were scale-suppressed by the `d_model` normalisation.
5. Model selection could pick an epoch with untrained stage geometry; added a
   tie-break that never trades classification for geometry.
6. Permutation attribution tried to allocate 24 GB by re-running the 3D CNN per
   perturbed row.
7. The report language guard rejected its own disclaimers.
8. Confidence intervals exceeded 1.0 on bounded metrics.
9. `LogisticRegression(multi_class=)` was removed in scikit-learn 1.8.
10. M9-M11.4 reported `NOT_STARTED` despite executing.

**Found once the imaging stack could actually run:**

11. **Six wrong call signatures in the M1-M7 wrappers.** They were written
    against assumed APIs and had never been executable. `SkullStripper.strip`
    needs the affine; `MRIResizer.resample` needs the affine and returns
    `(volume, new_affine)`; `normalize_wm_peak` returns `(volume, wm_peak)`;
    `ROIExtractor` takes `data_dir`, not `atlas_dir`; `ArtifactDetector` takes
    the QC threshold.
12. **The resampled affine was discarded.** M5 replaces the voxel-to-world
    mapping, and M6 was being handed the original. The atlas would have been
    registered to the wrong grid while every shape and intensity check still
    looked correct.
13. **`ROICropper.extract_all` stacks patches in dict-insertion order and
    zero-fills a failed ROI.** Both are unacceptable: the tensor axis must
    follow `ROI_ORDER`, and a blank patch labelled with a subject's stage
    corrupts training invisibly. The wrapper now crops each ROI individually in
    canonical order and records failures instead of filling them.
14. **The 201 unassessed OASIS-1 sessions were reported as "unmapped CDR
    values".** `Series.map` does not preserve `None` — pandas converts it to
    `float('nan')` — so the identity check missed every one and emitted a
    misleading data-quality warning. Labels were correct; the diagnostic was not.
15. **Training split the full 235-session cohort rather than the processed
    subset**, so splits contained sessions with no features. Added
    `Context.trainable_cohort()`.
16. **A phantom-MRI outputs tree carried no synthetic marker.** Its artifacts
    come out of the real imaging pipeline, so nothing about them reveals the
    source. Added `modules/common/provenance.py`, stamped at preprocessing time
    and read by the dashboard, reports and figures.
17. **The report named the wrong synthetic source**, describing phantom-MRI runs
    as fabricated patches from the other generator.

### Outstanding

Only one, and it is not fixable from here: **`dataset/OASIS/` contains no real
MRI.** Every metric produced so far describes a synthetic generator. Place real
OASIS-1 T1 volumes there and re-run `--mode preprocess` to obtain results.
