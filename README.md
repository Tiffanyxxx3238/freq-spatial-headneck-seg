# FreqFuseNet: Resolving Feature-Scale Mismatch in Dual-Frequency Fusion for Thin-Wall Head-and-Neck OAR Segmentation

This repository contains the official implementation of **FreqFuseNet**, a scale-normalized dual-frequency residual fusion architecture for segmenting thin-wall organs-at-risk (OARs) in head-and-neck CT scans.

A preprint version of this work is available on medRxiv:
> https://www.medrxiv.org/content/10.64898/2026.07.09.26357642v2

---

## Overview

FreqFuseNet addresses a critical feature-scale mismatch between FcaNet and FFT-based frequency branches in dual-frequency fusion networks, and proposes a scale-normalized residual fusion strategy to resolve it. The method is developed and evaluated on the **SegRap2023** head-and-neck CT dataset.

The repository is organized as a staged research pipeline, reflecting the progressive development of the architecture from an initial focal-frequency loss baseline through to the final domain-generalization experiments.

## Repository Structure

```
files/segrap_research/
├── configs/                        # Configuration files
├── data/                           # Dataset loading utilities (SegRap2023 dataset class)
├── models/                         # Core model definitions and loss functions
├── external_baselines/             # Baseline model comparisons (nnU-Net, etc.)
├── stage1_focal_freq_loss/         # Stage 1: Focal frequency loss
├── stage2_fcanet_plugin/           # Stage 2: FcaNet frequency-channel attention plugin
├── stage3_fft_branch/              # Stage 3: FFT branch integration
├── stage4_mamba_fusion/            # Stage 4: Mamba-based fusion module
├── stage5_moe_router/              # Stage 5: Scale-normalized dual-frequency residual fusion (final architecture)
├── stage6_domain_generalization/   # Stage 6: Domain generalization experiments
├── utils/                          # Metrics and training utilities
├── requirements.txt                # Python dependencies
└── generate_paper_figures.py / visualize_*.py   # Figure/visualization scripts for the paper

paper_viz/                          # Representative qualitative result figures used in the paper
```

## Dataset

This project uses the **SegRap2023** head-and-neck CT dataset. The dataset is **not included** in this repository due to size and usage-license restrictions.

1. Request/download the dataset from the official SegRap2023 challenge page.
2. Place it under:
   ```
   data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases/
   ```
3. Update `--data_root` in the relevant training scripts (or `configs/config.py`) if you use a different location.

## Requirements

```bash
pip install -r files/segrap_research/requirements.txt
```

## Usage

Each stage directory contains its own `train.py` reflecting that stage of the architecture's development. For example, to train the final scale-normalized dual-frequency residual fusion model (Stage 5):

```bash
cd files/segrap_research/stage5_moe_router
python train_fft_residual.py --data_root ../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases
```

See each stage folder for stage-specific scripts (ablation studies, statistical analysis, etc.).

## Citation

If you use this code, please cite:

```bibtex
@article{freqfusenet2026,
  title   = {FreqFuseNet: Resolving Feature-Scale Mismatch in Dual-Frequency Fusion for Thin-Wall Head-and-Neck OAR Segmentation},
  author  = {Wan, Shu-Yen and Chen, Wen-Yu},
  year    = {2026},
  journal = {medRxiv preprint},
  doi     = {10.64898/2026.07.09.26357642}
}
```

## Contact

For questions regarding this repository, please contact the corresponding authors via the information listed in the manuscript.
