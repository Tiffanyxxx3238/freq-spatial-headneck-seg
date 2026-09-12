"""
Stage 5C: AFS-DSN with FFT-main Residual Dual-Frequency Fusion.

CrossDomainAttention and the final AdaptiveRouter (spatial/frequency mix)
are UNCHANGED — only freq_branch is swapped via _AFS_DSN_Base's freq_cls
hook, exactly like Stage 2/3/5/5B. FFTMainResidualFusion always runs both
FFT (primary) and FcaNet (bounded residual correction, beta <= 0.2) — see
fft_residual_fusion.py for the full rationale.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from .fft_residual_fusion import FFTMainResidualFusion


def AFS_DSN_FFTResidual(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                        use_freq_branch: bool = True, use_cross_attention: bool = True,
                        use_router: bool = True):
    """
    Stage 5C: FFT-main Residual Dual-Frequency Fusion.
    FFT branch is the primary frequency expert.
    FcaNet branch is used as a bounded small residual correction (beta <= 0.2).
    CrossDomainAttention and the final AdaptiveRouter (spatial/freq mix)
    stay exactly as in _AFS_DSN_Base — unchanged from Stage 1-5B.
    """
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=FFTMainResidualFusion,
    )
