# NeuroGenesis subject report

Stage-aware speech-network analysis of structural MRI: CN / MCI / AD classification, regional vulnerability, and stage-transition propensity.

| Field | Value |
|---|---|
| Subject ID | `OAS1_0001_MR1` |
| Generated | 2026-09-21 23:26:00 |
| Experiment | `oasis1_pilot31` |
| Dataset | OASIS-1 |
| Dataset source | Washington University / OASIS — https://sites.wustl.edu/oasisbrains/home/oasis-1/ |
| Data provenance | OASIS-1 (real) |

## 1. Current stage classification

**Model-assigned stage: CN** (Cognitively normal)

| Stage | Probability |
|---|---|
| CN | 0.5515  **<-- assigned** |
| MCI | 0.1061 |
| AD | 0.3423 |

Reference CDR-derived stage for this session: **CN**. The model's assignment agrees with it.

### Model confidence

- Top-class probability: 0.5515
- Margin over runner-up: 0.2092
- Normalised predictive entropy: 0.8495 (0 = fully confident, 1 = uniform)

_Confidence indicators are derived from softmax outputs and are not calibrated probabilities of correctness. A low-entropy prediction indicates that the model separated the classes confidently in its own representation, not that the classification is verified._

## 2. NeuroProp-X regional vulnerability

Model-derived vulnerability `RV_i = sigmoid(w_v^T xhat_i + b_v)`, measuring how informative each speech-related region is for the current CN/MCI/AD discrimination.

| Rank | Region | Vulnerability | Function |
|---|---|---|---|
| 1 | Wernicke | 0.9362 | Speech comprehension, auditory word recognition |
| 2 | Broca | 0.8519 | Speech production, phonological encoding |
| 3 | IFG | 0.5232 | Syntactic processing, verbal working memory |
| 4 | STG | 0.4443 | Auditory-verbal processing, spectrotemporal analysis |
| 5 | Insula | 0.2173 | Articulatory planning, phonological awareness |

> Vulnerability is a model-derived discriminative score. It is not a probability that the region will degenerate.

AP-LAF anatomical-prior weight `alpha` = 0.3319 — the share of the adaptive adjacency's mass contributed by the anatomical prior, with the remainder from learned attention.

## 3. Inter-regional pathways

### Most attended pathways (SAEG-GATv2 gated attention)

| Pathway | Attention | Edge gate | Anatomical tract |
|---|---|---|---|
| Insula -> Wernicke | 0.1946 | 0.8085 | Extreme_Capsule_System |
| Broca -> STG | 0.1909 | 0.8005 | no prior edge |
| Wernicke -> STG | 0.1896 | 0.8100 | Intra_STG_Fibres |
| STG -> Wernicke | 0.1892 | 0.8139 | Intra_STG_Fibres |
| IFG -> STG | 0.1883 | 0.8029 | Inferior_Fronto_Occipital_Fasciculus |

### Highest propagation-representation pathways (NeuroProp-X ANP)

| Pathway | Propagation score |
|---|---|
| Wernicke -> Broca | 0.7347 |
| Broca -> Wernicke | 0.7241 |
| IFG -> Broca | 0.7057 |
| Broca -> IFG | 0.7009 |
| Wernicke -> IFG | 0.6903 |

> Propagation scores are a learned edge-level representation combining endpoint vulnerability with adaptive connectivity. They are not biological disease-spread probabilities.

## 4. Stage-transition propensity

| Quantity | Value |
|---|---|
| Reference stage | CN |
| Advanced-stage alignment | 0.5043 |
| AD-associated propensity | 0.4799 |
| Stage-transition propensity | 0.5092 |
| Transition target | MCI |

_Propensity of the current representation toward the MCI-associated representation._

Stage alignment distribution:

- CN: 0.4908
- MCI: 0.0098
- AD: 0.4994

### Stage geometry quality

- CN < MCI < AD prototype ordering respected: **True**

> Stage-transition propensity is a model-derived measure of how closely the subject's current structural representation aligns with a more advanced disease-stage representation. It is derived from a single cross-sectional scan. It is not a clinically validated conversion probability, not a prediction of future diagnosis, and not a biological estimate of disease onset.

## 5. Unified ROI ranking

Signal weights actually applied: attention = 0.30, attribution = 0.30, vulnerability = 0.40.

| Rank | Region | Combined score | vulnerability | attention | attribution |
|---|---|---|---|---|---|
| 1 | Wernicke | 0.7528 | 1.0000 | 0.9241 | 0.2518 |
| 2 | Broca | 0.6656 | 0.8827 | 0.0973 | 0.9445 |
| 3 | IFG | 0.4702 | 0.4255 | 0.0000 | 1.0000 |
| 4 | STG | 0.4684 | 0.3158 | 1.0000 | 0.1403 |
| 5 | Insula | 0.1759 | 0.0000 | 0.5863 | 0.0000 |

## 6. Explainable AI

### Feature attribution — method: **SHAP (Kernel)**

| Region | Feature | Attribution | Direction |
|---|---|---|---|
| IFG | surface_area_vox | 0.1134 | increases |
| Broca | surface_area_vox | 0.0905 | increases |
| Broca | surface_to_volume_ratio | -0.0739 | decreases |
| IFG | surface_to_volume_ratio | -0.0368 | decreases |
| STG | gm_fraction | 0.0250 | increases |
| IFG | mean_intensity | -0.0238 | decreases |
| Wernicke | std_intensity | 0.0221 | increases |
| Wernicke | surface_area_vox | 0.0158 | increases |
| Wernicke | kurtosis | 0.0064 | increases |
| Insula | cortical_thickness_mm | -0.0007 | decreases |

- Kernel SHAP with a k-means-summarised background drawn from the training split. Shapley values assume feature independence, which correlated morphometric features violate to some degree.

### Node importance (incoming attention mass)

| Region | Incoming attention |
|---|---|
| STG | 0.9431 |
| Wernicke | 0.9215 |
| Insula | 0.8252 |
| Broca | 0.6857 |
| IFG | 0.6580 |

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
