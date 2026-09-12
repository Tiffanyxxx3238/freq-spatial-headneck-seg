"""
Stage 2: 3-D FcaNet-style DCT-based channel attention.

Key idea (FcaNet, NeurIPS 2021):
  Replace SE-Net's Global Average Pooling with a DCT coefficient as the channel
  descriptor.  A single DCT basis function (k=1) generalises GAP: where GAP uses
  a uniform (DC) basis, FcaNet uses the first non-trivial cosine basis, letting the
  attention network distinguish which spatial frequencies each channel responds to.

3-D extension: the basis is separable along D, H, W, so the 3-D coefficient is the
product of three 1-D cosine bases.

Design for AFS-DSN plug-in (this file):
  MultiSpectralAttention3D
    - Single frequency index k_idx = (1, 1, 1)  (lowest non-trivial 3-D diagonal mode)
    - Per-channel descriptor = weighted-sum of spatial features with that cosine basis
    - FC: C → C//r → C → Sigmoid  (same bottleneck structure as SE-Net)
    - Output: channel-reweighted feature map, same shape as input

  band_energies() helper
    - Computes |DCT coefficient| at 8 pre-defined 3-D frequencies → (B, 8)
    - Used by FcaFrequencyBranch3D to satisfy AdaptiveRouter's band_energies interface
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Eight frequency indices for band_energies (AdaptiveRouter compat.)  #
# All combinations of (k_d, k_h, k_w) ∈ {0, 1}³: DC through first   #
# diagonal mode.                                                       #
# ------------------------------------------------------------------ #
_BAND_FREQS = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1),
]


def _dct3_coeff(x: torch.Tensor, k_d: int, k_h: int, k_w: int) -> torch.Tensor:
    """
    Compute one 3-D DCT-II coefficient for every (batch, channel) pair.

    The basis is separable:
        basis[d, h, w] = cos(π k_d (2d+1)/(2D))
                       × cos(π k_h (2h+1)/(2H))
                       × cos(π k_w (2w+1)/(2W))

    For k=0 along any axis the cosine is identically 1, so that axis
    contributes a plain sum — i.e., (0,0,0) reproduces Global Average Pooling.

    Args:
        x : (B, C, D, H, W) — input feature map
    Returns:
        (B, C) — one descriptor scalar per (batch, channel)
    """
    B, C, D, H, W = x.shape
    dev   = x.device
    dtype = x.dtype

    def _cos1d(k: int, N: int) -> torch.Tensor:
        if k == 0:
            return torch.ones(N, device=dev, dtype=dtype)
        n = torch.arange(N, device=dev, dtype=dtype)
        return torch.cos(torch.pi * k * (2 * n + 1) / (2 * N))

    bd = _cos1d(k_d, D)   # (D,)
    bh = _cos1d(k_h, H)   # (H,)
    bw = _cos1d(k_w, W)   # (W,)
    # Outer product: (D, H, W) basis
    basis = bd[:, None, None] * bh[None, :, None] * bw[None, None, :]
    # Weighted spatial mean: normalise by spatial volume so scale ≈ GAP
    return (x * basis[None, None]).sum(dim=[2, 3, 4]) / (D * H * W)   # (B, C)


class MultiSpectralAttention3D(nn.Module):
    """
    3-D FcaNet channel attention with k=1 (single DCT frequency).

    Replaces SE-Net's GAP with a DCT coefficient at k_idx = (1, 1, 1):
    the lowest-frequency diagonal mode that is strictly different from GAP.

    Parameters
    ----------
    channels  : number of input channels C
    reduction : SE-style bottleneck reduction factor  (default 16 → mid = C//16)
    k_idx     : 3-D DCT frequency index (k_d, k_h, k_w)
                (0,0,0) = GAP  |  (1,1,1) = lowest non-trivial diagonal mode
    """

    def __init__(self, channels: int, reduction: int = 16,
                 k_idx: tuple = (1, 1, 1)):
        super().__init__()
        self.k_idx = k_idx
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W)  →  channel-reweighted (B, C, D, H, W)"""
        k_d, k_h, k_w = self.k_idx
        fd  = _dct3_coeff(x, k_d, k_h, k_w)          # (B, C)
        att = self.fc(fd).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        return x * att

    def compute_band_energies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate |DCT coefficient| at the 8 pre-defined 3-D frequencies and
        average across channels → (B, 8).

        This gives AdaptiveRouter the spectral energy profile of the bottleneck
        feature without requiring any additional learned parameters.
        """
        energies = []
        for k_d, k_h, k_w in _BAND_FREQS:
            fd = _dct3_coeff(x, k_d, k_h, k_w)   # (B, C)
            energies.append(fd.abs().mean(dim=1))  # (B,)
        return torch.stack(energies, dim=1)        # (B, 8)


# ------------------------------------------------------------------ #
# Kept for backward-compat with nnunet_wrapper.py (deprecated path).  #
# ------------------------------------------------------------------ #
class FcaResBlock3D(nn.Module):
    """Deprecated — original nnU-Net hook-based wrapper. Use FcaFrequencyBranch3D instead."""

    def __init__(self, channels: int, reduction: int = 16, **_kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.attn = MultiSpectralAttention3D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.conv(x))
