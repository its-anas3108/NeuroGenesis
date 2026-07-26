# NeuroGenesis — Phase 1: Self-Evolving Digital Twin Framework

> **Project Title**: NeuroGenesis: A Self-Evolving Digital Twin Framework with NeuroProp-X for Predictive Modeling of Progressive Atrophy in Speech-Related Brain Networks

> **Phase**: 1 of 3 — Preprocessing Pipeline (first 30%)

---

## Overview

NeuroGenesis is a research-grade neuroimaging pipeline designed for Alzheimer's disease research, specifically targeting **progressive atrophy in speech-related brain networks**. This Phase 1 implementation provides a complete preprocessing, segmentation, feature extraction, and connectivity graph foundation that will feed into the NeuroProp-X prediction algorithm in Phase 2.

### Key Novel Contributions (Phase 1)

| Novel Feature | Description |
|---|---|
| **MRI Artifact Detector** | Automated QC scoring (0–100) detecting motion, dropout, Gibbs ringing, SNR, saturation |
| **N4 Bias Field Correction** | Scanner RF coil intensity gradient removal before normalization |
| **White Matter Peak Normalization** | KDE-based WM-peak as intensity reference (inter-subject comparable) |
| **CLAHE (3D slice-wise)** | Adaptive histogram equalization per axial slice for local contrast |
| **Anisotropic Diffusion** | Edge-preserving denoising (Perona-Malik) vs. blurring Gaussian |
| **ROI-Centric Pipeline** | Extract speech ROI patches (48×48×48 each) → `(5, 48, 48, 48)` NeuroProp-X input |

---

## Project Structure

```
NeuroGenesis/
├── dataset/
│   └── OASIS/                   ← Place your .nii/.nii.gz files here
│
├── preprocessing/
│   ├── loader.py                ← MRILoader: nibabel, metadata, tri-plane viz
│   ├── artifact_detector.py     ← ArtifactDetector: QC scoring (NOVEL)
│   ├── normalization.py         ← MRINormalizer: N4+CLAHE+Aniso+WM-norm
│   ├── skull_strip.py           ← SkullStripper: Nilearn + SimpleITK
│   ├── resize.py                ← MRIResizer: SimpleITK resampling
│   └── roi_crop.py              ← ROICropper: 3D patch extraction (NOVEL)
│
├── segmentation/
│   └── roi_extraction.py        ← ROIExtractor: Harvard-Oxford atlas, 5 ROIs
│
├── features/
│   └── feature_extractor.py     ← FeatureExtractor: 13 features per ROI
│
├── graph/
│   └── graph_builder.py         ← BrainConnectivityGraph: NetworkX + anatomy
│
├── visualization/
│   └── visualize.py             ← NeuroGenesisVisualizer: 300 DPI figures
│
├── outputs/
│   ├── original/                ← Tri-plane images, slice galleries
│   ├── processed/               ← Preprocessing stages, skull strip
│   ├── segmented/               ← ROI masks, overlays, statistics
│   ├── roi_patches/             ← Per-ROI .nii.gz patches + mosaic
│   ├── tensors/                 ← (5,48,48,48) .npy tensors → NeuroProp-X
│   ├── graphs/                  ← Connectivity graph + adjacency matrix
│   ├── features/                ← CSV, JSON, heatmap per subject
│   └── neurogenesis.log         ← Full execution log
│
├── main.py                      ← Pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/NeuroGenesis.git
cd NeuroGenesis

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Place your OASIS MRI files (`.nii` or `.nii.gz`) in the `dataset/OASIS/` directory:

```
dataset/OASIS/
├── OAS1_0001_MR1.nii.gz
├── OAS1_0002_MR1.nii.gz
└── ...
```

The pipeline will **auto-scan recursively** — any depth of nesting is supported (BIDS format, flat, etc.).

> **First run** requires an internet connection to download the Harvard-Oxford atlas (~6 MB, cached by Nilearn).

---

## Running the Pipeline

```bash
python main.py
```

### Configuring the Pipeline

All parameters are in the `CONFIG` dict at the top of `main.py`:

```python
CONFIG = {
    "dataset_dir":      Path("dataset/OASIS"),   # Your data here
    "output_dir":       Path("outputs"),
    "target_shape":     (128, 128, 128),          # Resize target
    "patch_size":       (48, 48, 48),             # ROI patch size
    "min_quality_score": 50.0,                    # QC threshold
    "skip_failed_qc":   False,                    # Skip bad scans?
    "max_subjects":     None,                     # Limit for testing
    ...
}
```

---

## Pipeline Stages

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | MRI Loading | `loader.py` | Tri-plane PNGs, metadata JSON |
| 2 | QC Check | `artifact_detector.py` | QC score, QC report PNG |
| 3 | Preprocessing | `normalization.py` | 5-stage NIfTI files, histograms |
| 4 | Skull Strip | `skull_strip.py` | Masked brain, brain mask NIfTI |
| 5 | Resize | `resize.py` | 128×128×128 resampled NIfTI |
| 6 | ROI Extraction | `roi_extraction.py` | 5 binary mask NIfTI files, overlays |
| 7 | ROI Patches 🆕 | `roi_crop.py` | 5 × 48×48×48 .nii.gz patches + .npy tensor |
| 8 | Feature Extraction | `feature_extractor.py` | CSV, JSON, heatmap |
| 9 | Graph Construction | `graph_builder.py` | Graph PNG, adjacency matrix CSV |
| 10 | Visualisation | `visualize.py` | Pipeline summary PNG (300 DPI) |

---

## ROI-Centric Pipeline (Novel Architecture)

Instead of processing full brain volumes, NeuroGenesis extracts compact 3D patches around each speech-related ROI:

```
Full MRI (182×218×182 = ~7.2M voxels)
    ↓ Skull Strip + Preprocess
Brain Volume
    ↓ ROI Mask Extraction (Harvard-Oxford Atlas)
5 × Binary Masks (Broca, Wernicke, Insula, IFG, STG)
    ↓ 3D Bounding Box + Context Padding + Resize
5 × (48, 48, 48) ROI Patches = ~553K voxels  [6× reduction]
    ↓ Stack
(5, 48, 48, 48) Tensor  ←── NeuroProp-X Phase 2 Input
```

### NeuroProp-X Integration Hooks

All Phase 2 integration points are built and documented:

```python
# Get ROI tensor for NeuroProp-X spatial encoder
tensor = roi_cropper.get_neuroprox_tensor("OAS1_0001")
# → np.ndarray shape (5, 48, 48, 48)

# Get feature vector for NeuroProp-X conditioning
vec = feature_extractor.get_feature_vector(df)
# → np.ndarray shape (N_rois × N_features,)

# Get graph for NeuroProp-X graph encoder (GCN/GAT)
graph, adj = graph_builder.get_graph_data("OAS1_0001")
# → (nx.DiGraph, np.ndarray shape (5, 5))
```

---

## Speech ROIs Extracted

| ROI | Anatomical Region | Function | Brodmann |
|---|---|---|---|
| **Broca Area** | IFG pars triangularis + opercularis | Speech production | BA 44, 45 |
| **Wernicke Area** | Posterior STG + planum temporale | Speech comprehension | BA 22 |
| **Insula** | Insular cortex | Articulatory planning | BA 13, 14 |
| **IFG** | Full inferior frontal gyrus | Syntactic processing | BA 44, 45, 47 |
| **STG** | Superior temporal gyrus | Auditory-verbal processing | BA 22, 41, 42 |

---

## Outputs Reference

After a successful run:

| Output | Location | Description |
|---|---|---|
| Original MRI images | `outputs/original/` | Tri-plane PNG, slice gallery |
| Preprocessed MRI | `outputs/processed/` | All 5 stage NIfTI + figures |
| Skull stripped MRI | `outputs/processed/` | `*_skull_stripped.nii.gz` |
| ROI masks | `outputs/segmented/` | `*_{roi_name}_mask.nii.gz` |
| ROI overlays | `outputs/segmented/` | Per-ROI + combined PNG |
| ROI patches | `outputs/roi_patches/` | `*_{roi_name}_patch.nii.gz` |
| NeuroProp-X tensor | `outputs/tensors/` | `*_roi_tensor.npy` (5,48,48,48) |
| Feature CSV | `outputs/features/` | Per-subject + aggregate |
| Feature JSON | `outputs/features/` | Per-subject JSON |
| Connectivity graph | `outputs/graphs/` | Graph PNG + adjacency CSV |
| QC report | `outputs/processed/` | QC dashboard PNG |
| Execution log | `outputs/neurogenesis.log` | Full stage-by-stage log |
| Pipeline summary | `outputs/pipeline_summary.json` | Complete run metadata |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| nibabel | ≥5.0 | NIfTI I/O |
| SimpleITK | ≥2.3 | N4 bias correction, anisotropic diffusion, resampling |
| nilearn | ≥0.10 | Atlas download, ROI extraction, brain masking |
| numpy | ≥1.24 | Array operations |
| pandas | ≥2.0 | Feature tables |
| scipy | ≥1.11 | Gaussian smoothing, KDE, statistical features |
| scikit-image | ≥0.21 | CLAHE, marching cubes (surface area) |
| networkx | ≥3.1 | Connectivity graph |
| matplotlib | ≥3.7 | All visualisations (300 DPI) |
| tqdm | ≥4.66 | Progress bars |
| opencv-python | ≥4.8 | Image operations |

---

## Phase Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Preprocessing Pipeline (this repo) | ✅ Complete |
| **Phase 2** | NeuroProp-X — Spatial + Graph Encoder | 🔜 Next |
| **Phase 3** | Digital Twin + Longitudinal Prediction | 🔮 Future |

---

## Citation

If you use NeuroGenesis in your research, please cite:

```bibtex
@software{neurogenesis2024,
  title  = {NeuroGenesis: A Self-Evolving Digital Twin Framework for
             Predictive Modeling of Progressive Atrophy in Speech Networks},
  year   = {2024},
  note   = {Phase 1: Preprocessing Pipeline with ROI-Centric Processing},
}
```

---

## License

Research use only. See `LICENSE` for details.
