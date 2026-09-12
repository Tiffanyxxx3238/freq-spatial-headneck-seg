"""
Stage 4: AFS-DSN with Mamba SSM cross-domain fusion (replaces CrossDomainAttention).

Two variants, pairing the Mamba fusion with each Stage 2/3 frequency branch:
  AFS_DSN_FcaNet_Mamba  — Stage 4A: FcaNet DCT channel-attention branch + Mamba fusion
  AFS_DSN_FFT_Mamba     — Stage 4B: FFT spectral branch          + Mamba fusion

Both reuse _AFS_DSN_Base unchanged (encoder/decoder/router untouched) via its
new `fusion_cls` hook (models/afs_dsn_original.py), which defaults to
CrossDomainAttention so Stage 1/2/3 model.py call sites are unaffected.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from stage2_fcanet_plugin.fca_branch import FcaFrequencyBranch3D
from stage3_fft_branch.fft_branch import FFTFrequencyBranch3D
from .mamba_fusion import FreqSpatialMambaFusion


def AFS_DSN_FcaNet_Mamba(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                         use_freq_branch: bool = True, use_cross_attention: bool = True,
                         use_router: bool = True):
    """Stage 4A: FcaNet frequency branch + Mamba cross-domain fusion."""
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FcaFrequencyBranch3D,
        fusion_cls=FreqSpatialMambaFusion,
    )


def AFS_DSN_FFT_Mamba(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                      use_freq_branch: bool = True, use_cross_attention: bool = True,
                      use_router: bool = True):
    """Stage 4B: FFT frequency branch + Mamba cross-domain fusion."""
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FFTFrequencyBranch3D,
        fusion_cls=FreqSpatialMambaFusion,
    )
