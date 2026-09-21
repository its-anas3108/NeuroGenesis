# NeuroGenesis subject report

Stage-aware speech-network analysis of structural MRI: CN / MCI / AD classification, regional vulnerability, and stage-transition propensity.

| Field | Value |
|---|---|
| Subject ID | `OAS1_0003_MR1` |
| Generated | 2026-09-21 23:26:03 |
| Experiment | `oasis1_pilot31` |
| Dataset | OASIS-1 |
| Dataset source | Washington University / OASIS — https://sites.wustl.edu/oasisbrains/home/oasis-1/ |
| Data provenance | OASIS-1 (real) |

## 1. Current stage classification

**Model-assigned stage: CN** (Cognitively normal)

| Stage | Probability |
|---|---|
| CN | 0.4486  **<-- assigned** |
| MCI | 0.3025 |
| AD | 0.2489 |

Reference CDR-derived stage for this session: **MCI**. The model's assignment differs from it.

### Model confidence

- Top-class probability: 0.4486
- Margin over runner-up: 0.1462
- Normalised predictive entropy: 0.9716 (0 = fully confident, 1 = uniform)

_Confidence indicators are derived from softmax outputs and are not calibrated probabilities of correctness. A low-entropy prediction indicates that the model separated the classes confidently in its own representation, not that the classification is verified._

## 2. NeuroProp-X regional vulnerability

Model-derived vulnerability `RV_i = sigmoid(w_v^T xhat_i + b_v)`, measuring how informative each speech-related region is for the current CN/MCI/AD discrimination.

| Rank | Region | Vulnerability | Function |
|---|---|---|---|
| 1 | Wernicke | 0.8433 | Speech comprehension, auditory word recognition |
| 2 | STG | 0.5449 | Auditory-verbal processing, spectrotemporal analysis |
| 3 | Broca | 0.5110 | Speech production, phonological encoding |
| 4 | IFG | 0.4273 | Syntactic processing, verbal working memory |
| 5 | Insula | 0.4042 | Articulatory planning, phonological awareness |

> Vulnerability is a model-derived discriminative score. It is not a probability that the region will degenerate.

AP-LAF anatomical-prior weight `alpha` = 0.3319 — the share of the adaptive adjacency's mass contributed by the anatomical prior, with the remainder from learned attention.

## 3. Inter-regional pathways

### Most attended pathways (SAEG-GATv2 gated attention)

| Pathway | Attention | Edge gate | Anatomical tract |
|---|---|---|---|
| Broca -> Insula | 0.1826 | 0.8039 | Short_Arcuate_Fibres |
| IFG -> Insula | 0.1802 | 0.8027 | Inferior_Fronto_Occipital_Fasciculus |
| Wernicke -> Insula | 0.1792 | 0.8064 | Extreme_Capsule_System |
| STG -> Insula | 0.1786 | 0.8046 | Extreme_Capsule_System |
| Wernicke -> Broca | 0.1781 | 0.8102 | Superior_Longitudinal_Fasciculus |

### Highest propagation-representation pathways (NeuroProp-X ANP)

| Pathway | Propagation score |
|---|---|
| STG -> Wernicke | 0.6995 |
| Wernicke -> STG | 0.6974 |
| Broca -> Wernicke | 0.6791 |
| Wernicke -> Broca | 0.6770 |
| Insula -> Wernicke | 0.6714 |

> Propagation scores are a learned edge-level representation combining endpoint vulnerability with adaptive connectivity. They are not biological disease-spread probabilities.

## 4. Stage-transition propensity

| Quantity | Value |
|---|---|
| Reference stage | CN |
| Advanced-stage alignment | 0.5758 |
| AD-associated propensity | 0.4893 |
| Stage-transition propensity | 0.6057 |
| Transition target | MCI |

_Propensity of the current representation toward the MCI-associated representation._

Stage alignment distribution:

- CN: 0.3943
- MCI: 0.0598
- AD: 0.5459

### Stage geometry quality

- CN < MCI < AD prototype ordering respected: **True**

> Stage-transition propensity is a model-derived measure of how closely the subject's current structural representation aligns with a more advanced disease-stage representation. It is derived from a single cross-sectional scan. It is not a clinically validated conversion probability, not a prediction of future diagnosis, and not a biological estimate of disease onset.

## 5. Unified ROI ranking

Signal weights actually applied: attention = 0.30, attribution = 0.30, vulnerability = 0.40.

| Rank | Region | Combined score | vulnerability | attention | attribution |
|---|---|---|---|---|---|
| 1 | Wernicke | 0.7000 | 1.0000 | 0.0000 | 1.0000 |
| 2 | STG | 0.3902 | 0.3205 | 0.5001 | 0.3731 |
| 3 | Broca | 0.3726 | 0.2431 | 0.7835 | 0.1343 |
| 4 | IFG | 0.3461 | 0.0526 | 0.5497 | 0.5338 |
| 5 | Insula | 0.3000 | 0.0000 | 1.0000 | 0.0000 |

## 6. Explainable AI

### Feature attribution — method: **SHAP (Kernel)**

| Region | Feature | Attribution | Direction |
|---|---|---|---|
| Wernicke | surface_area_vox | 0.0568 | increases |
| Wernicke | surface_to_volume_ratio | -0.0458 | decreases |
| IFG | surface_to_volume_ratio | 0.0277 | increases |
| STG | surface_to_volume_ratio | -0.0271 | decreases |
| IFG | std_intensity | 0.0220 | increases |
| Wernicke | std_intensity | 0.0176 | increases |
| Broca | surface_to_volume_ratio | 0.0161 | increases |
| IFG | skewness | -0.0145 | decreases |
| STG | normalized_volume | -0.0124 | decreases |
| STG | gm_volume_mm3 | -0.0053 | decreases |

- Kernel SHAP with a k-means-summarised background drawn from the training split. Shapley values assume feature independence, which correlated morphometric features violate to some degree.

### Node importance (incoming attention mass)

| Region | Incoming attention |
|---|---|
| Insula | 0.9169 |
| Broca | 0.8612 |
| IFG | 0.8011 |
| STG | 0.7884 |
| Wernicke | 0.6598 |

## 7. Stage-wise statistical context (cohort level)

- Group sizes: {'CN': 14, 'MCI': 10, 'AD': 6}
- Tests run: 195 (skipped: 15)
- FDR-significant: 31
- Multiple-comparison correction: benjamini_hochberg, applied across all ROI x feature x contrast tests as one family

Largest FDR-significant stage differences:

| Region | Feature | Contrast | FDR p | Effect size | Trend |
|---|---|---|---|---|---|
| Broca | voxel_count | MCI vs AD | 2.46e-02 | -1.998 (hedges_g) | decrease |
| Broca | voxel_count | CN vs AD | 2.46e-02 | -1.805 (hedges_g) | decrease |
| Broca | brain_volume_mm3 | MCI vs AD | 2.46e-02 | -1.998 (hedges_g) | decrease |
| Broca | brain_volume_mm3 | CN vs AD | 2.46e-02 | -1.805 (hedges_g) | decrease |
| Broca | gm_volume_mm3 | MCI vs AD | 4.85e-02 | -0.800 (rank_biserial) | decrease |
| Broca | gm_volume_mm3 | CN vs AD | 2.46e-02 | -0.857 (rank_biserial) | decrease |
| Broca | skewness | CN vs AD | 3.87e-02 | 1.496 (hedges_g) | increase |
| Broca | surface_area_vox | CN vs MCI | 3.67e-02 | -0.700 (rank_biserial) | decrease |

> Benjamini-Hochberg FDR was applied across all 195 tests in one family (every ROI x feature x contrast), not per contrast.

> 15 cell(s) were not tested because a group fell below the minimum of 5 or the feature was constant. Those cells carry no p-value rather than a fabricated one.

> The smallest stage group has 6 subject(s). Effect-size estimates at this sample size have wide confidence intervals, and a non-significant result should not be read as evidence of no difference.

## 8. Ablation results (cohort level)

| Configuration                                                      | Variant   | Accuracy          | Macro-F1          | Balanced Accuracy   | ROC-AUC           |
|:-------------------------------------------------------------------|:----------|:------------------|:------------------|:--------------------|:------------------|
| Baseline (morphometry only)                                        | A0        | 0.2857 +/- 0.1650 | 0.2714 +/- 0.1787 | 0.3333 +/- 0.2187   | 0.5120 +/- 0.1467 |
| + Anatomical Prior                                                 | A1        | not run           | not run           | not run             | not run           |
| + Standard GAT                                                     | A2        | not run           | not run           | not run             | not run           |
| + Learned Attention (AP-LAF)                                       | A3        | not run           | not run           | not run             | not run           |
| + SRVE                                                             | A4        | not run           | not run           | not run             | not run           |
| + ANP                                                              | A5        | 0.2419 +/- 0.1167 | 0.1946 +/- 0.1129 | 0.2444 +/- 0.1694   | 0.5369 +/- 0.2373 |
| Full: NeuroProp-X + SAEG-GATv2 + structural covariance + Stage-TGT | A7        | 0.3657 +/- 0.1263 | 0.3022 +/- 0.1574 | 0.3889 +/- 0.1884   | 0.5513 +/- 0.2199 |

> Confidence intervals and p-values describe variability across repeated splits of one fixed cohort. The same subjects appear in every repeat, so these are not confidence intervals over the population of Alzheimer's patients.

> Comparisons against the reference variant are paired across identical splits and use the Wilcoxon signed-rank test, which does not assume normality of the paired differences.

> With a small AD class, a non-significant difference is not evidence that two variants perform equally.

## 9. Model version and configuration

| Field | Value |
|---|---|
| Variant | `A7` — Full: NeuroProp-X + SAEG-GATv2 + structural covariance + Stage-TGT |
| Graph encoder | saeg_gatv2 |
| NeuroProp-X | True (SRVE=True, AP-LAF attention=True, ANP=True) |
| Edge gate | True |
| Stage-TGT | True |
| Trainable parameters | 132,136 |
| Node feature width | 19 |
| Shared representation width | 64 |
| Checkpoint | `outputs_oasis1_pilot31/checkpoints/A7.pt` |

Parameters by component:

- spatial_encoder: 0
- spatial_projection: 0
- neuropropx: 1,946
- graph_encoder: 37,128
- fusion: 25,024
- classifier: 195
- stage_tgt: 67,648
- propensity_head: 195

## 10. Dataset

| Field | Value |
|---|---|
| Dataset | **OASIS-1** |
| Source | Washington University / OASIS — https://sites.wustl.edu/oasisbrains/home/oasis-1/ |
| Volume | PROCESSED/MPRAGE/T88_111 — Talairach-88 atlas space, 1 mm isotropic, averaged across acquisitions, N4 and gain-field corrected, skull present |
| Synthetic data | DISABLED |

> OASIS applied N4 bias-field and gain-field correction before distribution (filename components 'n4' and 'gfc'). The pipeline's M3 stage applies N4 again; this duplication is harmless but is recorded rather than hidden.

## 10. Limitations

- The dataset is **OASIS-1 cross-sectional** (Washington University / OASIS). Findings apply to that cohort and have not been replicated on any independent dataset.
- The study is **cross-sectional**. Every subject contributes one scan, so no within-subject change is measured and no statement about the future can be supported.
- The MCI class is derived from CDR = 0.5, documented in OASIS-1 as 'very mild dementia'. It is a conventional cross-sectional stand-in for MCI, not an independent clinical MCI diagnosis.
- The labelled OASIS-1 cohort contains 30 AD sessions. Point estimates from a single split are unreliable at this size, which is why every comparison is reported over repeated splits with confidence intervals.
- The anatomical prior uses expert-assigned ordinal tract weights, not measured diffusion tractography. Its contribution is tested empirically by ablations A1 and A3 rather than assumed.
- Grey-matter volume is an intensity-threshold proxy, and cortical thickness is a volume-to-surface proxy. Neither is a tissue-class segmentation or a FreeSurfer measurement.
- Confidence indicators come from softmax outputs and are not calibrated probabilities of correctness.
- The system has not been clinically validated and is not a medical device.

## Disclaimer

> These results are model-derived associations and stage-propensity estimates produced by a research system from a single cross-sectional structural MRI scan. They are not a confirmed clinical diagnosis, not a validated prediction of future disease conversion, and not a basis for any clinical decision. Regional vulnerability, propagation scores and stage-transition propensity describe how informative each region and pathway is for the model's current discrimination between CN, MCI and AD; they are not measurements of biological degeneration or disease spread. This system has not been clinically validated.
