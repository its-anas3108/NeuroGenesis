# Data-Source Migration Report

**Required by Section 23 before any code is modified.** Records every synthetic,
mock or generated data path found in the repository, the OASIS-1 integration
points, which files change, which do not, and the schema adaptations the real
data forces.

**Scope discipline (Sections 3, 22, 25): only the data-ingestion layer changes.**
NeuroProp-X, SRVE, AP-LAF, ANP, SAGR, the 3D CNN, SAEG-GATv2, Stage-TGT, ROI
ranking, XAI, statistics, ablation, the dashboard architecture and the report
structure are untouched. Their mathematics, interfaces and responsibilities stay
exactly as implemented.

---

## 1. The real dataset, as inspected

Located at `C:\Users\avisy\Downloads\dataset`.

| Property | Value |
|---|---|
| Contents | 12 files, `oasis_cross-sectional_disc1..12.tar.gz` |
| Total size | 16 GB compressed |
| Dataset | OASIS-1 cross-sectional |
| Source | Washington University / OASIS — <https://sites.wustl.edu/oasisbrains/home/oasis-1/> |
| State on arrival | **Not extracted** |

### Archive layout (verified by listing `disc1`)

```
disc1/OAS1_0001_MR1/
├── RAW/                                   up to 4 un-averaged acquisitions
│   └── OAS1_0001_MR1_mpr-{1..4}_anon.{img,hdr}   + preview .gif
├── PROCESSED/MPRAGE/
│   ├── T88_111/                           Talairach-88 atlas space, 1 mm
│   │   ├── *_mpr_n4_anon_111_t88_gfc.{img,hdr}          skull present
│   │   ├── *_mpr_n4_anon_111_t88_masked_gfc.{img,hdr}   OASIS brain mask applied
│   │   └── t4_files/                      registration transforms
│   └── SUBJ_111/                          subject space, 1 mm
└── FSL_SEG/                               FSL tissue segmentation
```

### Volume format (verified by loading `OAS1_0001_MR1`)

| Property | Value | Consequence |
|---|---|---|
| nibabel class | `Spm2AnalyzeImage` | Analyze `.img`/`.hdr` pair, not NIfTI |
| Shape | **`(176, 208, 176, 1)`** | **4-D with a singleton 4th axis — must be squeezed** |
| dtype | **`>i2`** (big-endian int16) | Must be cast to float32 |
| Voxel size | 1.0 × 1.0 × 1.0 mm | No resampling needed before M5 |
| Orientation | `('L', 'A', 'S')` | LAS, not RAS |
| Affine translation | `(87.5, -103.5, -87.5)` | **Atlas-centred** — Harvard-Oxford registers correctly |
| Intensity range | 0 – 3910 | |
| Non-zero voxels | 73.6 % | Skull and neck present, as expected for `gfc` |

### Which volume the pipeline consumes, and why

**`PROCESSED/MPRAGE/T88_111/*_t88_gfc.{img,hdr}`** is the default.

- It is **atlas-registered**, so the MNI-space Harvard-Oxford atlas lands on the
  correct anatomy. `RAW/` is native scanner space with an arbitrary origin and
  would need registration first; `SUBJ_111/` is 1 mm but still subject-space.
- It is **averaged across the subject's 3–4 acquisitions** and gain-field
  corrected, which is the volume OASIS itself designates for analysis.
- The **skull is still present**, so the pipeline's own skull-stripping stage
  (M4) runs as designed rather than being bypassed.

`*_t88_masked_gfc` is extracted alongside and selectable by configuration, but
is **not** the default: it is already brain-extracted, which would make M4 a
no-op and silently change what the pipeline does.

> **Disclosed redundancy.** The filename component `n4` indicates OASIS already
> applied N4 bias-field correction, and `gfc` indicates gain-field correction.
> The pipeline's M3 stage applies N4 again. This is harmless — N4 on an
> already-corrected volume converges near-identically — but it is a real
> duplication and is recorded in each subject's provenance rather than hidden.

### Extraction

`tools/extract_oasis1.py` extracts **only** the `T88_111` volumes. A full
extraction would be far larger (the `RAW/` tree holds up to four uncompressed
acquisitions per subject plus preview GIFs and FSL segmentations), and none of
it is used. This is selective extraction of real data; nothing is generated.

---

## 2. Synthetic / mock / generated data paths found

Every file in the repository containing `np.random`, `default_rng`, `randn` or
`synth_`, and what each one actually is:

| # | Location | What it is | Verdict |
|---|---|---|---|
| 1 | `tools/make_synthetic_mri.py` | Generates phantom T1 NIfTI volumes (parametric ellipsoids) named after real OASIS session IDs | **SYNTHETIC SOURCE — disable for research runs** |
| 2 | `tools/make_smoke_artifacts.py` | Generates fabricated ROI patches + feature tables, skipping imaging | **SYNTHETIC SOURCE — disable for research runs** |
| 3 | `dataset/generate_synthetic_oasis.py` | Pre-refactor synthetic MRI generator, already inactive | **SYNTHETIC SOURCE — disable for research runs** |
| 4 | `modules/common/seeds.py` | Seeds `random` / `numpy` / `torch` RNGs for reproducibility | **KEEP** — controls randomness, does not create data |
| 5 | `modules/m01_dataset/splits.py` | `np.random.default_rng(seed)` shuffles **subject IDs** when partitioning | **KEEP** — permutes real subjects, creates none |
| 6 | `modules/m10_results/figures.py` | Jitter offsets for scatter-plot readability | **KEEP** — cosmetic plotting only, never a reported value |
| 7 | `tools/validate_checkpoints.py` | Random tensors to check module wiring and shapes | **KEEP, quarantine** — a unit-test harness, must never feed training |
| 8 | `tests/test_invariants.py` | Synthetic fixtures for the invariant test-suite | **KEEP, quarantine** — tests only |

**Finding: no synthetic data path exists inside the model, training, feature,
graph or evaluation code.** Items 4–6 use randomness but generate no samples;
items 7–8 are test harnesses. The only true synthetic *sources* are the three
standalone generator scripts, none of which is imported by `run.py`'s research
modes.

### Existing artifact trees built from synthetic data

| Tree | Built from | Action |
|---|---|---|
| `outputs_smoke/` | `make_smoke_artifacts.py` (fabricated patches) | Excluded from research results; already stamped `synthetic_patches` |
| `outputs_imaging/` | `make_synthetic_mri.py` (phantom volumes) | Excluded from research results; already stamped `synthetic_mri` |
| `dataset/OASIS_synthetic/` | `make_synthetic_mri.py` | Excluded; not the OASIS-1 root |

All checkpoints in those trees were trained on synthetic input and are
**discarded** per Section 15. The real experiment retrains from scratch.

> The provenance machinery added earlier (`modules/common/provenance.py`) already
> stamps and detects both synthetic kinds, and every consumer renders a banner.
> The migration extends it with a positive assertion of OASIS-1 rather than
> replacing it.

---

## 3. OASIS-1 integration points

| Point | Current behaviour | After migration |
|---|---|---|
| MRI discovery | `discover_mri_files()` globs `dataset/OASIS` for `.nii/.nii.gz/.hdr/.img/.mgz` | Delegates to `OASIS1DataManager`, which walks the extracted OASIS-1 tree and selects the configured T88 volume per session |
| Volume loading | `MRILoader.load_single()` via nibabel | Unchanged, plus a squeeze of the singleton 4th axis and a float32 cast |
| Metadata | `dataset/oasis_cross-sectional.csv` (already the real OASIS-1 table) | Unchanged — already real |
| Labels | CDR → CN/MCI/AD, already validated at 135/70/30 | Unchanged |
| Splits | `SplitManifest` JSON | Unchanged, **plus** the three CSVs Section 9 requires |
| Provenance | Detects synthetic markers | **Plus** a positive `dataset_source == "OASIS-1"` assertion and a pre-training integrity guard |

---

## 4. Files that will be modified

Data-ingestion layer only.

| File | Change | Why |
|---|---|---|
| `modules/m01_dataset/oasis1_manager.py` | **NEW** — `OASIS1DataManager`: recursive discovery, session/subject ID parsing, volume selection, integrity validation, metadata join | Section 5 |
| `modules/m01_dataset/validation.py` | **NEW** — dataset validation report + summary CSV | Section 6 |
| `modules/m01_dataset/integrity.py` | **NEW** — pre-training integrity guard | Section 8 |
| `modules/m01_dataset/cohort.py` | Route discovery through `OASIS1DataManager` | Sections 2, 5 |
| `modules/m01_dataset/splits.py` | Additionally emit `oasis1_{train,val,test}.csv` | Section 9 |
| `modules/common/config.py` | Add `oasis1_root`, `oasis1_volume_kind`, `allow_synthetic_data=False`, augmentation settings | Sections 14, 19 |
| `modules/common/provenance.py` | Add positive OASIS-1 assertion alongside synthetic detection | Sections 8, 20 |
| `modules/m02_preprocessing/pipeline.py` | **Minimal**: squeeze a 4-D singleton axis, cast big-endian int16 to float32, record provenance | Section 4 — a genuine OASIS compatibility need |
| `run.py` | Wire the manager, validation and guard into the modes; refuse to train without OASIS-1 | Sections 8, 17 |
| `dashboard/app.py`, `dashboard/state.py` | Add the "Dataset Integrity" page and the dataset banner | Sections 7, 21 |
| `modules/m11_report/report.py` | State OASIS-1 as the dataset in every report | Section 24 |

## 5. Files that will NOT be modified

The entire scientific pipeline:

```
modules/m04_feature_extraction/      feature spec, extraction, scaler
modules/m05_graph_construction/      anatomical prior
modules/m06_spatial_encoder/         3D CNN, patch dataset
modules/m06_neuropropx/              SRVE, AP-LAF, ANP, SAGR, engine, types
modules/m06_graph_learning/          SAEG-GATv2, GAT/GATv2 baselines, fusion, classifier
modules/m07_stage_tgt/               prototypes, transformer, propensity head
modules/m08_xai/                     attribution, attention, ROI ranking
modules/m09_statistics/              stage-wise tests
modules/m10_results/                 figures
modules/model.py                     model assembly, ablation ladder
modules/training/                    losses, metrics, trainer, ablation, baselines
modules/inference/                   single-subject pipeline
modules/m03_segmentation/            Harvard-Oxford ROI extraction  (Section 11)
preprocessing/  segmentation/  features/  graph/  visualization/   preserved originals
```

No mathematical formulation, module responsibility or interface changes.

---

## 6. Schema adaptations the real data requires

Four, all confined to the ingestion layer, each forced by a measured property of
the real files:

1. **Squeeze the singleton 4th axis.** OASIS T88 volumes load as
   `(176, 208, 176, 1)`. Every downstream stage expects 3-D. Squeezing only a
   trailing axis of extent 1 is lossless; a 4-D volume with a real 4th dimension
   would be rejected rather than silently reduced.

2. **Cast big-endian int16 to float32.** The data is `>i2`. Several SimpleITK
   operations and all torch tensors require native-endian float.

3. **Analyze `.img`/`.hdr` pairs, not NIfTI.** Discovery must key on the `.hdr`
   (nibabel reads the pair from it) and must not treat `.img` and `.hdr` as two
   separate volumes. The existing `MRI_SUFFIXES` already lists both, which would
   double-count; the manager selects one canonical file per session.

4. **Session vs subject identity.** OASIS-1 session IDs are `OAS1_0001_MR1`. The
   existing `extract_subject_id()` already strips `_MRn`, and splitting is
   already subject-wise — no change, but the manager asserts it.

---

## 7. What the migration does not do

- Does not download anything. The official URL is documentation only; the
  experiment uses the locally supplied files.
- Does not hard-code a path. `oasis1_root` is configuration.
- Does not delete the synthetic generators — they remain for unit-testing the
  code path — but they are disabled for research execution and cannot reach
  training past the integrity guard.
- Does not reuse any checkpoint trained on synthetic data.

---

## 8. Honest statement of what the results will be

Once migration completes, the reported study size is whatever the uploaded
OASIS-1 actually yields after validation and QC — not the nominal 416. The
labelled cohort is **235 sessions (CN 135 / MCI 70 / AD 30)**, since 201 OASIS-1
subjects have no CDR assessment. Any subject whose volume fails validation or QC
is excluded and counted in the validation report.

**AD n=30 remains the binding constraint on every claim**, exactly as before.
Real data does not change that arithmetic.
