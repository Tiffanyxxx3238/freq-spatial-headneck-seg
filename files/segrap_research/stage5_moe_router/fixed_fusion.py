"""
Stage 5B: Deterministic Dual-Frequency Fusion.

Replaces Stage 5's hard Top-K MoE router (stage5_moe_router/moe_router.py)
with a design that cannot collapse the way discrete routing can: FcaNet and
FFT both ALWAYS run, every forward pass, for every sample. There is no
selection mechanism and therefore no possibility of an expert being starved
of gradient the way Stage 5's top_k=1 case was (softmax of a single selected
logit is identically 1.0 -> zero gradient through the mixing weight -> the
router could in principle learn to permanently exclude an expert with no
pressure to ever reconsider). The Identity expert from Stage 5 is dropped
entirely — with both real experts mandatory, "exclude everything and fall
back to identity" is no longer an available escape valve.

Mixing weight: channel-wise learnable alpha, NOT a single global scalar.
sigmoid(alpha[c]) is channel c's FFT weight; 1-sigmoid(alpha[c]) is its
FcaNet weight. This sits between two extremes:
  - A single global scalar would force every channel to use the same
    FcaNet/FFT ratio, even though different channels likely encode different
    kinds of structure (e.g. some channels may be more boundary/edge-like —
    where FFT's explicit phase reconstruction might help more — while others
    are smoother/lower-frequency, where FcaNet's DCT attention may suffice).
  - Per-element (channel AND spatial-position) weighting would reintroduce
    much of the instability surface that made the hard-routing design risky
    (more independently-moving parameters, more ways for a subset of them to
    drift into a degenerate corner) for limited expected benefit.
Channel-wise is the smallest step up in expressiveness from a global scalar
that still lets different channels specialise, while every alpha[c] is a
smooth, differentiable function with a well-defined gradient at every
training step — there is no discrete decision boundary anywhere in this
module, unlike torch.topk's hard selection in Stage 5.

alpha initialised to exactly 0 -> sigmoid(0) = 0.5 for every channel:
training starts from an unbiased, exactly-50/50 FcaNet/FFT mix. The FINAL
learned alpha therefore reflects what training discovered to be useful, not
an artifact of whichever expert the initialisation happened to favour (the
exact problem this design is meant to avoid — see module docstring above).

IMPORTANT CLARIFICATION — what alpha is NOT:
  alpha is a single set of GLOBAL, per-CHANNEL nn.Parameter values, fixed
  for the whole model once trained. It is NOT input-dependent: it does not
  look at `x`, does not vary per sample, per case_id, or per oar_name, and
  has no routing/gating network computing it from features (unlike Stage
  5A's MultiExpertFrequencyRouter, which DOES compute a per-sample decision
  from pooled input statistics). Concretely: two different test samples
  (different OARs, different cases) passed through the same trained
  FixedDualFrequencyFusion get mixed with the EXACT SAME alpha values —
  only their FcaNet/FFT branch OUTPUTS differ (since those experts are
  themselves input-dependent), not the mixing weight applied to them.
  Therefore: any analysis of a trained model's alpha may only claim things
  like "channel 37 ended up weighted 80% toward FFT" — it must NOT claim
  "OAR X prefers FFT more than OAR Y" or "case_id Z routes differently",
  since alpha carries no information about which OAR/case produced it.

Channel count: self.alpha has shape (channels,), where `channels` is the
SAME constructor argument passed to FcaFrequencyBranch3D(channels) and
FFTFrequencyBranch3D(channels) below — not a separately hard-coded number.
Verified empirically (not just assumed by reading the source) that both
experts preserve channel count (output channels == input channels == the
constructor's `channels` argument) and return identically-shaped (B, 8)
band_energies; see the band_energies mixing code below for the defensive
fallback this implies.

Defensive numerical-stability measures carried over from Stage 4/5 findings
(applied proactively here rather than discovered via a mid-training crash):
  - Input clamped before being handed to EITHER expert: stress-testing in
    Stage 5 found that even FcaNet's own InstanceNorm3d can produce NaN
    under fp16 autocast at extreme input magnitude (squaring during
    variance computation overflows fp16's ~65504 max), not just the FFT
    expert specifically.
  - FFT expert's rfftn/irfftn forced to fp32 with autocast disabled for
    that call: rfftn on an fp16 input produces ComplexHalf, which crashed
    CUDA outright during Stage 4 testing (not just produced NaN).
  - Expert outputs and band_energies clamped before mixing, as a hard
    backstop independent of the above.
  - _debug_check_finite probes at the same key points as Stage 4/5,
    controlled by env var FIXEDFUSION_DEBUG_NAN (default on).
"""
import os

import torch
import torch.nn as nn

from stage2_fcanet_plugin.fca_branch import FcaFrequencyBranch3D
from stage3_fft_branch.fft_branch import FFTFrequencyBranch3D

_DEBUG_NAN_CHECK = os.environ.get('FIXEDFUSION_DEBUG_NAN', '1') != '0'


def _debug_check_finite(name: str, t: torch.Tensor) -> None:
    if not _DEBUG_NAN_CHECK:
        return
    if not torch.isfinite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        finite_vals = t[torch.isfinite(t)]
        if finite_vals.numel() > 0:
            rng = f"finite range=[{finite_vals.min().item():.3e}, {finite_vals.max().item():.3e}]"
        else:
            rng = "no finite values remain"
        print(f"[FixedFusion DEBUG] non-finite at '{name}': "
              f"{n_nan} NaN + {n_inf} Inf / {t.numel()} elements  ({rng})")


class FixedDualFrequencyFusion(nn.Module):
    """
    Drop-in replacement for FcaFrequencyBranch3D / FFTFrequencyBranch3D /
    MultiExpertFrequencyRouter as _AFS_DSN_Base's freq_cls:
        __init__(channels)
        forward(x) -> (output, band_energies)
        shapes: (B, C, D, H, W) in, (B, C, D, H, W) out; band_energies (B, 8).

    Both FcaNet and FFT experts run on every forward call; there is no
    routing/selection mechanism and no third (Identity) expert.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.fcanet = FcaFrequencyBranch3D(channels)
        self.fft    = FFTFrequencyBranch3D(channels)

        # alpha=0 -> sigmoid(alpha)=0.5 for every channel: unbiased 50/50
        # mix at initialisation (see module docstring).
        self.alpha = nn.Parameter(torch.zeros(channels))

    def get_alpha_distribution(self) -> torch.Tensor:
        """
        sigmoid(alpha) per channel, detached -- the FFT weight for each of
        the `channels` bottleneck channels (FcaNet weight = 1 - this).
        Intended for post-training analysis: log this (e.g. alongside the
        oar_name of whichever batch produced a given checkpoint, or simply
        as a histogram of the trained model's final per-channel mix) to see
        whether training discovered a non-trivial, continuous FcaNet/FFT
        preference rather than the hard 0/1 splits Stage 5's top-k routing
        was limited to.

        Note: alpha is a single set of model-level parameters, not a
        per-sample or per-OAR quantity — it does not vary across OARs
        within one trained checkpoint. Any anatomy-related pattern (e.g.
        "thin-wall OARs end up associated with higher mean FFT weight")
        would have to be inferred indirectly: by training separate
        checkpoints on OAR subsets and comparing their alpha distributions,
        or by correlating each channel's typical activation pattern (which
        DOES vary per OAR, since the input x differs) against this fixed
        per-channel weighting.
        """
        return torch.sigmoid(self.alpha.detach())

    def get_alpha_stats(self) -> dict:
        """
        Aggregate statistics over the per-channel FFT weight sigmoid(alpha),
        for logging/reporting (e.g. train_fixed.py dumps this to a JSON file
        after evaluation). All values are plain Python floats.

        Reminder (see module docstring): these are GLOBAL, channel-wise
        statistics of a fixed model parameter — not a function of any
        particular input, OAR, or case. Do not report this as "how the
        model routes for OAR X."
        """
        w = torch.sigmoid(self.alpha.detach())
        return {
            'mean_fft_weight':              w.mean().item(),
            'std_fft_weight':                w.std().item(),
            'min_fft_weight':                w.min().item(),
            'max_fft_weight':                w.max().item(),
            'fraction_channels_fft_gt_0_5':  (w > 0.5).float().mean().item(),
            'fraction_channels_fft_gt_0_9':  (w > 0.9).float().mean().item(),
        }

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, D, H, W) — bottleneck feature (GPU tensor)
        Returns (output, band_energies), same shapes as either expert alone.
        """
        _debug_check_finite('FixedDualFrequencyFusion.input', x)

        # Defensive clamp before EITHER expert sees the input (Stage 5
        # finding: FcaNet's own InstanceNorm3d can overflow under fp16
        # autocast at extreme magnitude, not just the FFT expert).
        x_clamped = torch.clamp(x, -1e4, 1e4)

        fcanet_out, fcanet_be = self.fcanet(x_clamped)
        _debug_check_finite('FixedDualFrequencyFusion.fcanet_out', fcanet_out)
        _debug_check_finite('FixedDualFrequencyFusion.fcanet_be', fcanet_be)

        # FFT expert: force fp32, autocast disabled (Stage 4 finding: rfftn
        # on fp16 input produces ComplexHalf, which crashed CUDA outright).
        with torch.autocast(device_type=x.device.type, enabled=False):
            fft_out, fft_be = self.fft(x_clamped.float())
        fft_out = fft_out.to(x.dtype)
        fft_be  = fft_be.to(x.dtype)
        _debug_check_finite('FixedDualFrequencyFusion.fft_out', fft_out)
        _debug_check_finite('FixedDualFrequencyFusion.fft_be', fft_be)

        # Hard backstop regardless of which expert produced these.
        fcanet_out = torch.clamp(fcanet_out, -1e4, 1e4)
        fft_out    = torch.clamp(fft_out,    -1e4, 1e4)
        fcanet_be  = torch.clamp(fcanet_be,  -1e4, 1e4)
        fft_be     = torch.clamp(fft_be,     -1e4, 1e4)

        w_fft = torch.sigmoid(self.alpha).to(x.dtype)   # (C,) in (0, 1)
        w_fca = 1.0 - w_fft                               # (C,)

        w_fft_5d = w_fft[None, :, None, None, None]
        w_fca_5d = w_fca[None, :, None, None, None]
        output = w_fca_5d * fcanet_out + w_fft_5d * fft_out

        # band_energies is (B, 8) — indexed by frequency band, not channel,
        # so there is no natural per-channel correspondence to mix it with.
        # Use the mean channel preference as a single representative scalar
        # weight for the 8-dim energy vector (the "average tendency across
        # all channels to prefer FFT vs FcaNet" for this forward pass).
        #
        # Verified empirically (see PR discussion) that FcaFrequencyBranch3D
        # and FFTFrequencyBranch3D both return (B, 8) band_energies, so the
        # scalar-weighted sum below is shape-safe. Guard it anyway rather
        # than silently relying on that forever: if either branch's
        # band_energies shape is ever changed upstream, fail loudly here
        # instead of producing a silently-wrong broadcast.
        if fcanet_be.shape != fft_be.shape:
            raise RuntimeError(
                f"FixedDualFrequencyFusion: band_energies shape mismatch "
                f"(fcanet={tuple(fcanet_be.shape)}, fft={tuple(fft_be.shape)}). "
                f"The scalar-weighted mix below assumes identical shapes; "
                f"this assumption was true when this module was written but "
                f"no longer holds. Add an explicit projection (e.g. a small "
                f"Linear mapping the smaller shape up to the larger one) "
                f"before mixing rather than relying on broadcasting."
            )
        w_fft_scalar = w_fft.mean()
        w_fca_scalar = 1.0 - w_fft_scalar
        band_energies = w_fca_scalar * fcanet_be + w_fft_scalar * fft_be

        _debug_check_finite('FixedDualFrequencyFusion.output', output)
        _debug_check_finite('FixedDualFrequencyFusion.band_energies', band_energies)

        return output, band_energies
