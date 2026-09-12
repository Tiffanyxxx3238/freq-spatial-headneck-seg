"""
Stage 2: FcaFrequencyBranch3D — AFS-DSN bottleneck plug-in.

Drop-in replacement for FrequencyBranchV4 (DWT) and FFTFrequencyBranch3D.
Identical interface:
  __init__(channels)
  forward(x) → (output, band_energies)
    output       : (B, C, D, H, W)  same shape as input
    band_energies: (B, 8)           compatible with AdaptiveRouter

Purpose (Stage 2 comparison):
  Test "simple DCT-based channel attention" against:
    • Stage 1 DWT baseline   (FrequencyBranchV4,    ~414 M params)
    • Stage 3 FFT branch     (FFTFrequencyBranch3D,  ~29.4 M params)
  Expected: lighter than both; validates whether explicit spectral reconstruction
  (Stage 3) outperforms a pure channel-attention descriptor (Stage 2).

Architecture:
  1. MultiSpectralAttention3D (DCT coefficient at k_idx=(1,1,1) as descriptor)
       → channel-reweighted feature, same shape as input
  2. Single 1×1×1 conv block (C → C, IN3d+affine, LeakyReLU)
       → lightweight feature mixing after attention reweighting
  3. band_energies from raw input x at 8 DCT frequencies → (B, 8)
       DCT coefficients computed on raw x (before attention) so they reflect
       the spectral content of the incoming bottleneck, not the post-attention output.

AdaptiveRouter compatibility:
  band_energies: (B, 8)
  → view(B, -1)[:, :8] = (B, 8)  ✓  (same as Stage 3 FFT branch)
"""
import torch
import torch.nn as nn

from .fca_module import MultiSpectralAttention3D


class FcaFrequencyBranch3D(nn.Module):
    """
    FcaNet-style frequency branch for AFS-DSN bottleneck.

    Parameters
    ----------
    channels : int — number of bottleneck feature channels (= base_features × 16)
    """

    def __init__(self, channels: int):
        super().__init__()

        # DCT channel attention: k_idx=(1,1,1) = first non-trivial 3-D diagonal mode
        self.attention = MultiSpectralAttention3D(
            channels, reduction=16, k_idx=(1, 1, 1)
        )

        # Lightweight post-attention conv: 1×1×1 preserves spatial size and keeps params low
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        x            : (B, C, D, H, W)   bottleneck feature (GPU tensor)
        output       : (B, C, D, H, W)   DCT-attention-refined feature
        band_energies: (B, 8)            for AdaptiveRouter — view(B,-1)[:,:8] ✓
        """
        # 1. Spectral energy profile from raw input (before modification)
        band_energies = self.attention.compute_band_energies(x)   # (B, 8)

        # 2. Channel-attention reweighting via DCT descriptor
        out = self.attention(x)    # (B, C, D, H, W)

        # 3. Lightweight 1×1×1 conv mixing
        out = self.conv(out)       # (B, C, D, H, W)

        return out, band_energies
