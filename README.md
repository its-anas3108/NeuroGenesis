# NeuroGenesis

**Stage-Aware Speech-Network NeuroAI Framework for CN/MCI/AD Classification, Regional Vulnerability Analysis, and Stage-Transition Propensity Estimation**

---

## Project description

NeuroGenesis is a multimodal NeuroAI framework for stage-wise analysis of Alzheimer's disease using structural MRI of speech-related brain networks. The system localizes five speech-related regions using the Harvard-Oxford atlas, learns local three-dimensional representations using a lightweight 3D CNN, constructs a subject-specific brain graph, and applies the proposed NeuroProp-X framework to estimate stage-relevant regional vulnerability, fuse anatomical priors with learned graph attention, and derive adaptive propagation representations. The resulting enriched disease graph is processed by an edge-gated GATv2 model for CN/MCI/AD classification. A Stage-Temporal Graph Transformer learns the ordered CN→MCI→AD representation to estimate model-derived stage-transition propensity. Explainable AI, ROI ranking, statistical analysis, and ablation studies provide interpretable and rigorous evaluation.

> **Research system, not a medical device.** Every output is a model-derived association or stage-propensity estimate from a single cross-sectional scan. Nothing here is a clinical diagnosis, a validated prediction of future disease conversion, or a basis for a clinical decision. The system has not been clinically validated.

---

## Research problem and gap

**Problem.** Language impairment is an early and functionally consequential feature of Alzheimer's disease, yet most structural-MRI staging models treat the brain either as a single global volume or as a few hundred anonymous parcels. Neither view says *which speech-network region* carries the discriminative signal, nor *which inter-regional pathway* the model relies on.

**Gap.** Three specific gaps motivate this framework:

1. **Region-level attribution is usually post-hoc.** Regional importance is typically read off a saliency map after the fact. Here, per-region vulnerability is an *explicit, learned quantity inside the model* (SRVE), so it is optimised by the classification objective rather than inferred afterwards.
2. **Anatomical priors are either trusted absolutely or ignored entirely.** Graph models over brain networks tend to fix the adjacency from an atlas or learn it from scratch. AP-LAF instead fuses the two under a **learnable** mixing weight, and ablations A1 versus A3 measure how much the prior actually contributes rather than assuming it helps.
3. **Stage ordering is discarded by flat classification.** CN, MCI and AD form an ordered severity scale, but a three-way softmax treats them as unrelated labels. The Stage-TGT models the ordering explicitly and derives a propensity score from it — **without** needing longitudinal data, and without claiming to predict conversion.

---

## Architecture

```
T1 MRI
  ↓
MRI preprocessing  (N4 → WM-KDE → CLAHE → anisotropic diffusion → min-max)
  ↓
Skull stripping → 128×128×128 standardization
  ↓
Harvard-Oxford atlas speech ROI localization
  ↓
Five speech-related ROIs → ROI-centric 48×48×48 patches → (5, 48, 48, 48)
  ↓
        ┌──────────────────────── two representation branches ───────────────────────┐
        │                                                                            │
   3D CNN spatial encoder                                     morphometric features
   (per-ROI 128-d E_i_3D)                                     (14 per ROI, X_i_morph)
        │                                                                            │
        │                              H_i = [X_i_morph ‖ E_i_3D]                    │
        │                                          ↓                                 │
        │                              Brain graph  G = (V, E)                       │
        │                                          ↓                                 │
        │                        NeuroProp-X:  SRVE → AP-LAF → ANP → SAGR            │
        │                                          ↓                                 │
        │                       Enhanced graph  G* = (V, X*, A*, P)                   │
        │                                          ↓                                 │
        │                                    SAEG-GATv2                              │
        │                                          ↓  Z_G                             │
        └──────────────────────┬───────────────────┘
                               ↓
                    Multimodal fusion  →  Z_H
                               │
                ┌──────────────┴───────────────┐
                ↓                              ↓
     Current-stage classifier            Stage-TGT
        P(CN) P(MCI) P(AD)         stage-transition propensity
                └──────────────┬───────────────┘
                               ↓
                    ROI ranking + XAI
                               ↓
                    Clinical-style report → dashboard
```

### Component responsibilities

| Component | Role | Novelty status |
|---|---|---|
| **Harvard-Oxford atlas** | Anatomical localization of the five speech ROIs | Established; not a contribution |
| **3D CNN** | Local 3-D spatial structure per ROI patch that hand-crafted features cannot express | Established architecture |
| **NeuroProp-X** | Turns the ordinary graph `G` into a disease-aware enhanced graph `G*` | **Proposed framework** |
| **SAEG-GATv2** | Graph representation and CN/MCI/AD classification with NeuroProp-X edge gating | GATv2 established; the edge gate and enriched input are proposed |
| **Stage-TGT** | Ordered stage geometry and stage-transition propensity | **Proposed formulation** |
| **XAI / ranking / statistics / ablation** | Interpretation and rigorous evaluation | Established methods, applied |

---

## NeuroProp-X

**Stage-Aware Regional Vulnerability and Adaptive Graph Propagation.** The framework's primary algorithmic contribution, in four stages.

### M11.1 — SRVE: Stage-Aware Regional Vulnerability Estimation

$$RV_i = \sigma\!\left(w_v^\top \hat{x}_i + b_v\right)$$

A shared vulnerability direction `w_v` with a per-ROI bias, learned by the classification objective. Sharing `w_v` is what makes the resulting scores comparable *between* regions — the comparison the ROI ranking depends on.

`RV_i` is a **model-derived discriminative vulnerability score**, not a biological probability of future degeneration.

### M11.2 — AP-LAF: Anatomical-Prior and Learned-Attention Fusion

$$A^{*} = \alpha\,A_{\text{prior}} + (1-\alpha)\,A_{\text{att}}, \qquad \alpha = \sigma(a)$$

`A_att` uses GATv2 *dynamic* attention, `e_ij = a^T LeakyReLU(W[h_i ‖ h_j])`. Both operands are row-stochastic before mixing, so `alpha` reads directly as the share of adjacency mass contributed by anatomy. Learned attention is **not** masked to the prior's support by default — masking it would make the learned branch incapable of discovering a pathway the prior omits.

### M11.3 — ANP: Adaptive Neurodegeneration Propagation

$$P_{ij} = \sigma\!\left(\beta_1 RV_i + \beta_2 RV_j + \beta_3 A^{*}_{ij} + \beta_4 T_{ij} + b\right)$$

Separate source and target coefficients preserve directionality. `T_ij` is a per-subject-normalised hub-to-hub topology term.

`P` is a **learned edge-level representation**, not a validated biological disease-spread probability. The variable is named `propagation_score` throughout the codebase for exactly this reason.

### M11.4 — SAGR: Stage-Aware Graph Representation

$$X_i^{*} = \left[\,X_i^{\text{morph}} \;\|\; E_i^{3D} \;\|\; RV_i \;\|\; \text{centrality}\,\right]$$

Yielding `G* = (V, X*, A*, P)`. Component column ranges are recorded so XAI can attribute an importance value back to *which kind* of evidence produced it.

---

## SAEG-GATv2

**Stage-Aware Edge-Gated Graph Attention Network v2.** Each directed edge carries `E_ij = [A_prior_ij, A_att_ij, P_ij]`, from which a gate is applied to the GATv2 attention:

$$g_{ij} = \sigma\!\left(w_g^\top E_{ij} + b_g\right), \qquad \tilde{\alpha}_{ij} = g_{ij}\,\alpha_{ij}$$

The gated attention is **not** renormalised, so the gate can attenuate a node's total incoming message rather than merely redistributing it — which is the point of a gate rather than a reweighting.

GATv2 (Brody et al., 2022) is established prior work and is not claimed as novel.

---

## Stage-TGT

**Stage-Temporal Graph Transformer.** OASIS-1 is cross-sectional, so this module **does not** forecast future MRI, simulate future scans, or estimate a clinically validated MCI→AD conversion probability.

It treats the ordered stage vocabulary as the sequence axis:

```
tokens:  [ c_CN , c_MCI , c_AD , Z_H ]  + stage positional embeddings
             └─── ordered stage prototypes ───┘   └ subject
```

Multi-head self-attention over these four tokens produces `Z_T`. The "temporal" in the name refers to the ordered *severity* axis, not observed time. Positional embeddings over the stage tokens are what make the axis ordered rather than a bag of three labels.

### Reported quantities

| Quantity | Definition |
|---|---|
| **Stage alignment** `q̃` | Learned softmax over the ordered stages from `Z_T` |
| **Advanced-stage alignment** | Expected ordinal position under `q̃`; 0 = fully CN-aligned, 1 = fully AD-aligned |
| **AD-associated propensity** | `d_CN / (d_CN + d_AD)` — distance to the CN prototype relative to the AD prototype. Purely geometric, no parameters |
| **Stage-transition propensity** | Alignment mass at stages strictly beyond the predicted stage, normalised against it. **Undefined** (reported as `null`) when the predicted stage is AD |

These are **model-derived propensity scores**, never conversion probabilities, guaranteed diagnoses, or biological estimates of onset.

---

## Loss

$$L_{\text{total}} = L_{\text{cls}} + \lambda_{\text{proto}} L_{\text{proto}} + \lambda_{\text{order}} L_{\text{order}} + \lambda_{\text{align}} L_{\text{align}}$$

| Term | Definition | Default λ |
|---|---|---|
| `L_cls` | Class-weighted cross-entropy on the current-stage logits | 1.0 |
| `L_proto` | Mean squared distance to the subject's own stage prototype, normalised by `d_model` | 1.00 |
| `L_order` | Margin hinges enforcing CN < MCI < AD in the prototype geometry and in per-sample distance ranking for the extreme classes | 0.50 |
| `L_align` | Cross-entropy on the Stage-TGT alignment head | 0.30 |

`L_align` is an addition to the design's three-term loss and it is **required, not optional**: the alignment head is the only consumer of the transformer output, so without it the Stage-TGT would receive no gradient and every reported propensity would come from random weights.

`L_order` deliberately imposes **no** ranking constraint on MCI subjects: MCI lies between the extremes, and there is no defensible ordering of `d_CN` versus `d_AD` for them.

---

## Dataset

**OASIS-1 cross-sectional.** Labels are derived from CDR, measured from the shipped metadata table:

| CDR | Stage | Sessions |
|---|---|---|
| 0.0 | CN | 135 |
| 0.5 | MCI | 70 |
| 1.0 / 2.0 | AD | 28 + 2 = **30** |
| *missing* | *excluded by default* | 201 |

436 rows over **416 unique subjects** (20 are `MR2` rescans). The labelled cohort is **235 sessions**.

Two label decisions that materially affect every claim:

- **CDR 0.5 → MCI.** OASIS-1 documents CDR 0.5 as "very mild dementia". It is the conventional cross-sectional stand-in for MCI, **not** an independent clinical MCI diagnosis. Reported as "MCI / very mild dementia".
- **The 201 missing-CDR sessions are excluded.** They are young subjects (18–40) who were never clinically assessed, not unlabelled AD-risk cases. Labelling them CN would roughly triple the CN class while making age almost perfectly predictive of stage. `missing_cdr_policy="cn"` exists only for a deliberate sensitivity analysis and emits a warning.

### Pointing the pipeline at a real OASIS-1 release

Set `paths.oasis1_root` (or pass `--oasis1-root`) to a directory holding the extracted
discs. Discovery, validation and indexing then run through
`modules/m01_dataset/oasis1_manager.py`, and the legacy `dataset/OASIS/` scan is bypassed.
No path is hard-coded, and nothing is downloaded automatically — the
[official OASIS-1 page](https://sites.wustl.edu/oasisbrains/home/oasis-1/) is
documentation, not a fetch target.

The release ships as 12 `.tar.gz` discs. Only the atlas-registered T1 series is needed,
so extract selectively — 11 GB instead of roughly 45 GB:

```bash
python tools/extract_oasis1.py --archives <archive-dir> --out <archive-dir>/extracted
python run.py --mode validate_dataset --config config_oasis1.json
```

Volumes are selected from `PROCESSED/MPRAGE/T88_111/*_t88_gfc.{img,hdr}` — Talairach-88
space, 1 mm isotropic. The **unmasked** series is deliberate: T88 registration puts the
Harvard-Oxford atlas on correct anatomy, and retaining the skull means M4 skull stripping
still does work rather than silently becoming a no-op.

`--mode validate_dataset` reports volumes found, unique subjects, unreadable files, shape
and orientation anomalies, sessions without metadata, metadata without a volume, missing
clinical variables, and the class distribution — then writes
`outputs/dataset_validation/oasis1_validation_report.json`. Nothing is repaired, imputed
or substituted: a session that fails is excluded and the reason is recorded, so the final
study size is explainable rather than asserted.

Before any model sees data, `check_dataset_integrity()` asserts that every training
session ID appears in the validated index. It is a **positive assertion**, not a synthetic
sniff test — a stray sample is rejected by name whatever its origin (a leftover cached
artifact, a hand-edited feature table, a file from an earlier run).

Without `oasis1_root`, volumes are read from `dataset/OASIS/`; flat, BIDS and
FreeSurfer-disc layouts are all discovered by matching the session ID anywhere in the path.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

The framework is layered: without the imaging packages (`nibabel`, `SimpleITK`, `nilearn`, `scikit-image`) the cohort, model, training, XAI, statistics and dashboard paths still run on cached artifacts, and `run.py --mode status` reports exactly what is missing.

---

## Usage

```bash
python run.py --mode status            # environment, artifacts, per-module state
python run.py --mode validate_dataset  # OASIS-1 discovery, validation, exclusion report
python run.py --mode preprocess        # M1-M8: imaging → patches → features
python run.py --mode train_cnn         # M9 only: cache 3D CNN embeddings
python run.py --mode train_graph       # graph branch without the CNN
python run.py --mode train_full        # end-to-end training of the full model
python run.py --mode evaluate          # metrics for an existing checkpoint
python run.py --mode ablation          # A0-A7 over repeated splits + baselines
python run.py --mode statistics        # stage-wise tests with FDR correction
python run.py --mode xai               # ROI ranking and explanations
python run.py --mode report            # subject reports
python run.py --mode figures           # research figures
python run.py --mode dashboard         # Streamlit pipeline inspection app
```

### Resuming an interrupted preprocessing pass

M3 (N4 bias-field correction) is roughly 90% of the imaging cost, so a full-cohort pass
runs for hours, and the feature table is only assembled after the last subject. Without
`--resume`, an interruption discards every subject already finished:

```bash
python run.py --mode preprocess --config config_oasis1.json --resume
```

A subject's standardised volume is reused only when its manifest records success, the file
it names is still on disk, **and** it was standardised to the grid the current config asks
for — so a changed `preprocess.target_shape` recomputes rather than silently leaving half
the cohort on the old grid. M6-M8 re-run from the cached volume, so the values produced are
the ones an uninterrupted pass would have produced; only the repeated work changes.

Single-subject inference:

```bash
python run.py --mode report --patient_id OAS1_0028_MR1
```

### Validating the code path without MRI data

If `dataset/OASIS/` is empty, the pipeline can still be exercised end-to-end on clearly-labelled synthetic data. Two generators exist, covering different halves of the pipeline.

**Phantom MRI — exercises the imaging chain (M1–M8):**

```bash
python tools/make_synthetic_mri.py --out dataset/OASIS_synthetic --n 9
python run.py --mode preprocess --config config_imaging.json   # M1-M8 for real
python run.py --mode train_full --config config_imaging.json
python run.py --mode report     --config config_imaging.json --patient_id OAS1_0028_MR1
```

Writes NIfTI volumes with a proper MNI-centred affine, named after **real** OASIS-1 session IDs so the labels and cohort machinery are genuine. The real preprocessing, Harvard-Oxford registration and feature extraction then run on them.

**Fabricated ROI patches — skips imaging, exercises the model (M9–M19):**

```bash
python tools/make_smoke_artifacts.py --out outputs_smoke --n-subjects 60
python tools/smoke_train.py --variant A7 --epochs 30
python run.py --mode statistics --outputs outputs_smoke
python run.py --mode ablation   --outputs outputs_smoke --repeats 3 --epochs 8
python run.py --mode dashboard  --outputs outputs_smoke
```

Faster, since it needs no imaging stack.

> **Provenance is tracked and cannot be lost.** `modules/common/provenance.py` detects both generators — one marks the outputs tree, the other marks the dataset — and `--mode preprocess` stamps the provenance into the outputs tree. Every consumer (dashboard, report, figures, tables) reads it and renders the correct banner: **"SYNTHETIC PHANTOM MRI"** or **"SYNTHETIC SMOKE-TEST DATA"**, never generic. The phantom-MRI case matters most: its artifacts come out of the real imaging pipeline, so nothing about them reveals the source. Metrics from either describe a generator, not Alzheimer's disease.

### Validation

```bash
python tools/validate_checkpoints.py --outputs outputs_imaging --mri-dir dataset/OASIS_synthetic
python tests/test_invariants.py
```

The Section 35 checkpoints report **PASS / FAIL / BLOCKED**; a checkpoint blocked by a missing package or absent data is never reported as a pass.

---

## Evaluation

### Data-leakage control

Enforced structurally, not by convention:

- **Splits partition subjects, never sessions.** OASIS-1 contains 20 `MR2` rescans; splitting rows would put two scans of the same brain on both sides of the boundary, and test accuracy would measure scan reproducibility rather than generalisation.
- **The feature scaler is fitted on the training split only,** and re-fitted per repeat in the ablation. The atrophy index is a *cohort-referenced* feature, so its reference median is a fitted quantity too.
- **Class weights come from the training split only.** A weight vector derived from all labels encodes the test set's composition.
- **Attribution background sets come from the training split.** Explaining a test subject never consults test-set statistics.
- **The test split is touched once,** after training, by `Trainer.evaluate`.
- Every split is written to a manifest recording the seed, fractions, counts and exact subject lists.

### Metrics

Accuracy, macro-F1, balanced accuracy, per-class precision / recall / specificity / F1, one-vs-rest ROC-AUC, confusion matrix. **Balanced accuracy is the default model-selection metric** — with CN 88 / MCI 46 / AD 20 training sessions, plain accuracy would select a CN-biased model.

Undefined metrics are reported as `null`, never as `0.0`: a class the model never predicted has *undefined* precision, which is a different statement from "the model was wrong about it".

### Ablation

| | Configuration |
|---|---|
| A0 | Morphometry only (graph-free MLP) |
| A1 | + anatomical prior (fixed propagation, no attention) |
| A2 | + standard GAT (static attention) |
| A3 | + learned attention (AP-LAF) |
| A4 | + SRVE |
| A5 | + ANP |
| A6 | 3D CNN + graph baseline (GATv2, no NeuroProp-X) |
| A7 | Full NeuroProp-X + SAEG-GATv2 + 3D CNN |
| A7_no_tgt | Full model without Stage-TGT |

Two deliberate properties:

- **Disabled components are replaced by shape-preserving stand-ins,** not removed. Every rung keeps identical downstream tensor shapes, so a measured difference is attributable to the component's *information* rather than to a change in capacity.
- **The edge gate turns on exactly when ANP does** (A5 onward). Gating before ANP exists would gate on a duplicate of `A*`; running ANP without the gate would compute `P` and discard it. Either would make the A4→A5 increment uninterpretable.

Every comparison runs over repeated stratified subject-wise splits and is reported as mean ± SD with a confidence interval, paired across splits with the Wilcoxon signed-rank test. With AD n=30, a single split's difference between two rungs is mostly noise.

### Statistics

For every ROI × feature, CN vs MCI / MCI vs AD / CN vs AD. Test choice is **data-driven per cell**: Shapiro-Wilk (D'Agostino above n=50) screens normality, Levene screens variance, then Student's *t*, Welch's *t* or Mann-Whitney *U* is selected — and the chosen test is recorded in the output. Effect size matches the test (Hedges' *g* or rank biserial). Benjamini-Hochberg FDR is applied across all tests as **one family**, not per contrast.

### Explainable AI

- **Feature attribution** — Kernel SHAP when `shap` is installed; otherwise **exact single-feature replacement**, reported as `permutation` and never labelled SHAP.
- **Graph attention** — the attention and edge-gate matrices recorded during the actual forward pass, not re-derived.
- **NeuroProp-X** — regional vulnerability and propagation values taken directly from the module outputs, with an exact per-feature decomposition of any ROI's `RV`.
- **3D CNN** — whole-ROI occlusion attribution. Voxel-level saliency is deliberately not produced: at this patch size and sample size such maps are dominated by noise.

---

## Repository layout

```
modules/
  common/                 config, seeds, logging, paths, ROI constants, run state
  m01_dataset/            cohort, CDR label mapping, subject-wise splits
  m02_preprocessing/      M1-M5 wrapper over the preserved preprocessing/ package
  m03_segmentation/       M6-M7 Harvard-Oxford ROI localization and patches
  m04_feature_extraction/ feature spec, extraction, leakage-safe scaler
  m05_graph_construction/ documented anatomical prior
  m06_spatial_encoder/    3D CNN, ROI patch dataset
  m06_neuropropx/         srve, ap_laf, anp, sagr, engine, types
  m06_graph_learning/     saeg_gatv2, GAT/GATv2 baselines, fusion, classifier
  m07_stage_tgt/          stage prototypes, transformer, propensity head
  m08_xai/                attribution, attention, vulnerability, ROI ranking
  m09_statistics/         stage-wise tests with FDR correction
  m10_results/            research figures
  m11_report/             report generator with enforced language discipline
  m12_future_extensions/  quarantined legacy longitudinal / Digital Twin code
  training/               losses, metrics, trainer, ablation, baselines
  inference/              single-subject pipeline
  model.py                assembled model and the ablation ladder
run.py                    pipeline orchestrator (all modes)
dashboard/                M1-M19 inspection app
tools/                    synthetic smoke-test generator and smoke training
preprocessing/  segmentation/  features/  graph/  visualization/  dataset/
                          preserved Phase-1 implementations, wrapped not rewritten
MIGRATION_PLAN.md         what was found, kept, quarantined and rebuilt
```

### Preserved versus rebuilt

**Preserved unchanged** and wrapped behind clean interfaces: the whole `preprocessing/` chain (QC, N4, WM-KDE, CLAHE, anisotropic diffusion, skull strip, resample, ROI crop), `segmentation/roi_extraction.py`, `features/feature_extractor.py`, the `ANATOMICAL_EDGES` prior in `graph/graph_builder.py`, `visualization/`, and `dataset/oasis1_loader.py`.

**Genuinely rebuilt:** NeuroProp-X. The pre-refactor engine computed vulnerability from hard-coded normative volumes and magic constants and had **zero trainable parameters**, so there was nothing to adapt — SRVE, AP-LAF and ANP are new parameterised PyTorch modules.

**Removed:** the pre-refactor XAI module, whose own comment read *"Synthetic SHAP value generation"* and which invented features (`"Naming Accuracy Index"`, `"Speech Fluency Score"`) that OASIS-1 does not measure; and the hard-coded `SpeechAssessmentScores(88.0, 86.0, 84.0, 87.0, 86.25)` that fed fabricated inputs into NeuroProp-X.

**Quarantined** under `modules/m12_future_extensions/`: the Digital Twin, the legacy longitudinal atrophy extrapolation (previously mislabelled a "Temporal Graph Transformer"), and the OASIS-3 manager. They require an explicit `acknowledge_not_validated=True` to load and emit a warning, so a legacy result cannot enter the pipeline by an accidental import.

---

## Dashboard

The dashboard is a **research pipeline inspection system**, not a results viewer. Every module (M1 → M19) exposes its input, processing status, output, numerical values, visualisation, downloadable artifact and an explanation of what it did.

Module status is read from recorded execution state (`outputs/state/`), written by the context manager that wraps each pipeline stage. A module that never ran shows `NOT_STARTED`; there is no hard-coded "completed" anywhere in the dashboard.

```bash
python run.py --mode dashboard --outputs outputs
```

---

## Reproducibility

- Seeds set for `random`, `numpy`, `torch` (CPU and all CUDA devices) and `PYTHONHASHSEED`; deterministic algorithms requested in warn-only mode, with what was actually achieved recorded in a `SeedReport`.
- Config snapshot and environment report written to `outputs/experiments/<id>/` on every run.
- Split manifests, fitted scalers and model checkpoints all persisted; checkpoints embed the config, the ROI order and the stage order, and **refuse to load** into a mismatched ordering.
- Determinism holds for a fixed device, library version and thread count. Cross-hardware bit-identical results are not guaranteed, and the `SeedReport` says so.

---

## Limitations

- **The study is cross-sectional.** Each subject contributes one scan, so no within-subject change is measured and no statement about the future can be supported. Figures use "stage-wise differences", never "progression".
- **AD n = 30.** Single-split point estimates are not reportable; every comparison uses repeated splits with confidence intervals. A non-significant difference is not evidence of equivalence.
- **CDR 0.5 as MCI** is a conventional cross-sectional proxy, not an independent clinical diagnosis.
- **The anatomical prior uses expert-assigned ordinal tract weights, not measured diffusion tractography.** The pre-refactor source claimed DTI fractional-anisotropy provenance without citation; that claim is not carried forward. The prior's contribution is tested by ablation rather than assumed.
- **Grey-matter volume is an intensity-threshold proxy**, and cortical thickness is a volume-to-surface proxy. Neither is a tissue-class segmentation or a FreeSurfer measurement.
- **Confidence indicators come from softmax outputs** and are not calibrated probabilities of correctness.
- **Confidence intervals describe variability across splits of one fixed cohort**, not sampling variability of the wider population — the same subjects recur in every repeat.
- **Baseline hyper-parameters are library defaults and were not tuned.** A tuned baseline could do better; the comparison is against reasonable off-the-shelf baselines, not the best achievable ones.
- **The system has not been clinically validated.**

---

## Optional future extension: longitudinal analysis

The Stage-TGT operates on ordered stage representations and needs no longitudinal data, which is why it works on OASIS-1. Should true longitudinal sequences become available, the natural extension is to add an observed-time axis alongside the stage axis and to validate stage-transition propensity against measured conversion outcomes. Until such data and validation exist, propensity remains a model-derived alignment score. The quarantined code under `modules/m12_future_extensions/` is a starting point, not a validated capability.

---

## Claim discipline

NeuroProp-X is presented as the **proposed** framework. Its components build on established ideas — logistic scoring, attention-based adjacency learning, learned edge features, feature concatenation — and none of those is claimed as new. What is proposed is:

- the specific SRVE + AP-LAF + ANP + SAGR formulation,
- the integration of local 3D ROI representation with a disease-aware graph representation,
- the NeuroProp-X + edge-gated GATv2 architecture,
- simultaneous current-stage classification, regional vulnerability and stage-transition propensity from one shared representation,
- ablation-based validation of each component.

GATv2 (Brody et al., 2022), GAT (Veličković et al., 2018), the Harvard-Oxford atlas, N4 bias correction, CLAHE and Perona-Malik diffusion are all established prior work used as foundations.

---

## License

Research use only.
