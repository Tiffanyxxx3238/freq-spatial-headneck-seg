"""
Stage 2A: AFS-DSN with FcaNet DCT channel-attention frequency branch.

CrossDomainAttention, AdaptiveRouter, encoder, and decoder are unchanged.
Only the bottleneck frequency branch is swapped via _AFS_DSN_Base's freq_cls hook.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from .fca_branch import FcaFrequencyBranch3D


def AFS_DSN_FcaNet(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                   use_freq_branch: bool = True, use_cross_attention: bool = True,
                   use_router: bool = True):
    """
    Factory — identical call signature to AFS_DSN_V4 / AFS_DSN_Lite / AFS_DSN_FFT.

    Swaps FrequencyBranchV4 (DWT, ~414 M) for FcaFrequencyBranch3D
    (DCT channel attention, ~24 M).  All other sub-modules are unchanged.

    Stage 2 purpose: compare "lightweight DCT channel attention" against
      • Stage 1 DWT baseline   (FrequencyBranchV4)
      • Stage 3 FFT branch     (FFTFrequencyBranch3D)
    in exactly the same encoder/decoder/attention/router scaffold.
    """
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FcaFrequencyBranch3D,
    )
