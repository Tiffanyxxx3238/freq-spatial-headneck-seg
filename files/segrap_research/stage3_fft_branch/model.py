"""
Stage 3A: AFS-DSN with FFT frequency branch (replaces DWT FrequencyBranchV4).

CrossDomainAttention, AdaptiveRouter, encoder, and decoder are unchanged.
Only the bottleneck frequency branch is swapped via _AFS_DSN_Base's freq_cls hook.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from .fft_branch import FFTFrequencyBranch3D


def AFS_DSN_FFT(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                use_freq_branch: bool = True, use_cross_attention: bool = True,
                use_router: bool = True):
    """
    Factory — identical call signature to AFS_DSN_V4 / AFS_DSN_Lite.
    Swaps FrequencyBranchV4 (DWT, ~390 M for base_features=32) for
    FFTFrequencyBranch3D (spectral gating, ~1.6 M).
    All other sub-modules are constructed by _AFS_DSN_Base unchanged.
    """
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FFTFrequencyBranch3D,
    )
