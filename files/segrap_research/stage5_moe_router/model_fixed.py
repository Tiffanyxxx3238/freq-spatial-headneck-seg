"""
Stage 5B: AFS-DSN with Deterministic Dual-Frequency Fusion.

CrossDomainAttention and the final AdaptiveRouter (spatial/frequency mix)
are UNCHANGED — only freq_branch is swapped via _AFS_DSN_Base's freq_cls
hook, exactly like Stage 2/3/5. FixedDualFrequencyFusion always runs both
FcaNet and FFT experts (no top-k selection, no Identity expert) and mixes
them with a channel-wise learnable alpha — see fixed_fusion.py for the
rationale (avoiding Stage 5's hard-routing collapse risk).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from .fixed_fusion import FixedDualFrequencyFusion


def AFS_DSN_FixedFusion(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                        use_freq_branch: bool = True, use_cross_attention: bool = True,
                        use_router: bool = True):
    """
    Stage 5B: Deterministic Dual-Frequency Fusion — FcaNet + FFT both always
    participate; a channel-wise learnable alpha (init 0 -> 50/50) decides
    the mix. CrossDomainAttention and the final AdaptiveRouter (spatial/freq
    mix) stay exactly as in _AFS_DSN_Base — unchanged from Stage 1-5.
    """
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FixedDualFrequencyFusion,
    )
