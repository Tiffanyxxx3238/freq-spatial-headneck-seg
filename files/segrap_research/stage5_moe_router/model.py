"""
Stage 5: AFS-DSN with Anatomy-aware Multi-Expert Frequency Routing.

CrossDomainAttention and the final AdaptiveRouter (spatial/frequency mix)
are UNCHANGED — only the freq_branch is swapped via _AFS_DSN_Base's freq_cls
hook, exactly like Stage 2 (FcaNet) and Stage 3 (FFT). MultiExpertFrequencyRouter
itself contains the 3 experts (FcaNet, FFT, Identity) and its own small
routing network; it is a single drop-in freq_cls, not a second router.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.afs_dsn_original import _AFS_DSN_Base
from .moe_router import MultiExpertFrequencyRouter


def AFS_DSN_MoE(in_channels: int = 1, num_classes: int = 2, base_features: int = 32,
                use_freq_branch: bool = True, use_cross_attention: bool = True,
                use_router: bool = True, top_k: int = 2):
    """
    Stage 5: MultiExpertFrequencyRouter replaces freq_branch.
    CrossDomainAttention and the final AdaptiveRouter (spatial/freq mix)
    stay exactly as in _AFS_DSN_Base — unchanged from Stage 1-4.

    top_k defaults to 2 (NOT 1): see moe_router.py docstring — with 3 experts,
    top_k=1 makes softmax(single logit) identically 1.0, giving the routing
    network no gradient signal through the mixing weight. top_k=1 is only a
    valid choice for evaluating a top_k=2-trained model under harder sparsity,
    never as the training default.
    """
    freq_cls = lambda channels: MultiExpertFrequencyRouter(channels, top_k=top_k)
    return _AFS_DSN_Base(
        in_channels, num_classes, base_features,
        use_freq_branch, use_cross_attention, use_router,
        freq_cls=freq_cls,
    )
