# FreqFuseNet: Resolving Feature-Scale Mismatch in Dual-Frequency Fusion for Thin-Wall Head-and-Neck OAR Segmentation

Official implementation accompanying the manuscript submitted to **Computer Methods and Programs in Biomedicine (CMPB)**.

**Preprint (medRxiv):** https://www.medrxiv.org/content/10.64898/2026.07.09.26357642v2
**DOI:** 10.64898/2026.07.09.26357642

**Authors:** Shu-Yen Wan¹, Wen-Yu Chen¹ᐟ³, Guan-Yu Lin²

¹ Department of Information Management, Chang Gung University, Taoyuan 333323, Taiwan
² Department of Otolaryngology, Head, and Neck Surgery, Chang Gung Memorial Hospital, Taoyuan 333423, Taiwan
³ Department of Computer Science and Information Engineering, Chang Gung University, Taoyuan 333323, Taiwan

---

## Overview

Thin-wall head-and-neck organs-at-risk (OARs) — the cochlea, vestibular semicircular canals, internal auditory canal, tympanic cavity, and middle ear — are small, boundary-dominated structures that are notoriously difficult to segment automatically, despite their importance for cochlear- and vestibular-sparing radiotherapy planning.

This project investigates dual-frequency (FFT + FcaNet) feature fusion for boundary-sensitive segmentation, and identifies a large activation-scale mismatch (~863×) between the two branches under FP16 mixed-precision training. Left uncorrected, this mismatch causes a nominal 5% residual fusion coefficient to behave as a roughly 43× dominant term, effectively inverting the intended FFT-dominant design.

**FreqFuseNet** fixes this by rescaling the FcaNet branch to the FFT branch's activation statistics *before* residual fusion, restoring the fusion coefficient to its intended low-amplitude role. On the SegRap2023 benchmark (10 thin-wall OARs, 180 binary per-OAR test samples), FreqFuseNet reaches a mean Dice of 0.849 and HD95 of 0.824 mm in the primary run (consistent results in a second independent seed), with statistically significant case-level improvements over 3D U-Net and MedNeXt-S, using only 29.7M parameters — a ~93% reduction versus the full wavelet-based baseline.

## Repository Structure

```
files/segrap_research/
├── configs/                        # Configuration files
├── data/                           # SegRap2023 dataset loading utilities
├── models/                         # Core model definitions and loss functions
├── external_baselines/             # 3D U-Net / MedNeXt-S / SegResNet baseline comparisons
├── stage1_focal_freq_loss/         # Stage 1: DWT baseline + focal frequency loss
├── stage2_fcanet_plugin/           # Stage 2: FcaNet frequency-channel attention
├── stage3_fft_branch/              # Stage 3: FFT branch
├── stage4_mamba_fusion/            # Stage 4: Mamba-based fusion (ablation only)
├── stage5_moe_router/              # Stage 5: FixedFusion (5B) and FreqFuseNet scale-normalized residual fusion (5C, final architecture)
├── stage6_domain_generalization/   # Stage 6: Domain generalization experiments
├── utils/                          # Metrics and training utilities
├── requirements.txt                # Python dependencies
└── generate_paper_figures.py / visualize_*.py   # Figure/visualization scripts for the paper

paper_viz/                          # Representative qualitative result figures used in the paper
```

## Dataset

Experiments use the **SegRap2023** head-and-neck CT benchmark (120 cases; 84 train / 18 val / 18 test), publicly available at:
https://segrap2023.grand-challenge.org

The raw dataset is **not included** in this repository. To reproduce the experiments:

1. Download the dataset from the official SegRap2023 challenge page (subject to the challenge's own data-use terms).
2. Place it under:
   ```
   data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases/
   ```
3. Adjust `--data_root` in the relevant scripts (or `configs/config.py`) if using a different path.

This work uses a controlled binary per-OAR ROI protocol: for each of 10 clinically prioritized thin-wall OARs, a bounding box is derived from the ground-truth annotation, expanded by a fixed margin, and resampled to 128×128×128.

## Requirements

```bash
pip install -r files/segrap_research/requirements.txt
```

## Usage

Each stage directory contains the training script for that stage of the architecture's development. To train the final FreqFuseNet model (Stage 5C, scale-normalized residual fusion):

```bash
cd files/segrap_research/stage5_moe_router
python train_fft_residual.py --data_root ../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases
```

Other stages (ablations, external baselines, domain-generalization experiments) follow the same pattern — see each folder for its specific script(s).

## Citation

If you use this code, please cite:

```bibtex
@article{wan2026freqfusenet,
  title   = {FreqFuseNet: Resolving Feature-Scale Mismatch in Dual-Frequency Fusion for Thin-Wall Head-and-Neck OAR Segmentation},
  author  = {Wan, Shu-Yen and Chen, Wen-Yu and Lin, Guan-Yu},
  year    = {2026},
  journal = {medRxiv},
  doi     = {10.64898/2026.07.09.26357642}
}
```

## Data Availability

All data analyzed in this study derive from the publicly available SegRap2023 head-and-neck CT dataset (https://segrap2023.grand-challenge.org). Processed data and trained models are available from the corresponding author upon reasonable request.

## Contact

For questions regarding this repository, please contact the corresponding authors via the information listed in the manuscript.
