"""
Stage 5C: FFT-main Residual Dual-Frequency Fusion.

Final Stage 5 variant. Stage 5A's hard Top-K MoE router collapsed
unpredictably across seeds (seed2: FFT-dominant ~94/6/0, best test
performance; seed3: Identity-dominant ~0/0.35/99.6, worse performance).
Stage 5B's channel-wise learnable alpha (free 50/50 mix) didn't reproduce
seed2's result either — alpha barely moved from its 0.5 initialisation
(mean_fft_weight=0.5009, std=0.0016 after training), suggesting the
unconstrained symmetric mixing weight gets very little gradient signal to
work with regardless of which expert "should" dominate.

Stage 5C removes the symmetry entirely: FFT is the fixed PRIMARY branch,
FcaNet is a bounded SMALL residual correction on top of it — not a
50/50-or-anything-else peer. There is no Identity expert and no Top-K
selection (per explicit instruction not to resurrect either).

    fused = fft_out + beta * fca_out
    beta  = max_beta * sigmoid(raw_beta),  max_beta = 0.2

beta is capped at 0.2 by construction (sigmoid never reaches 1), so FcaNet's
contribution can never exceed 20% of FFT's scale per channel, regardless of
training — this is a deliberate, hard architectural constraint, not just an
initialisation choice. raw_beta is initialised so that beta starts at 0.05
(one quarter of its ceiling): a small residual correction should start
small, not at the midpoint of its own range — unlike Stage 5B's alpha
(where 50/50 was the appropriate "no prior bias" start because the design
was symmetric), here the design is already asymmetric by intent (FFT main,
FcaNet residual), so the natural starting point for "how much correction" is
small, not half.

    raw_beta_init = logit(init_beta / max_beta) = logit(0.05 / 0.2) = logit(0.25)
                  = ln(0.25 / 0.75) = ln(1/3) ≈ -1.0986

This is NOT copied from Stage 5A seed2's observed FFT/FcaNet ratio
(~94/6, i.e. an implied ratio of ~0.06) — it's set from the general
residual-correction design principle (start small, let training decide how
much correction is actually useful), not hand-tuned to reproduce one seed's
post-hoc result.

CRITICAL FIX (found via the feature-scale diagnostic, before any training
was run): FFTFrequencyBranch3D's real_head/imag_head are deliberately
small-weight-initialised (std=1e-3, a Stage 3 design choice for ITS OWN
training stability, not touched here) and have no final normalisation
layer, while FcaFrequencyBranch3D ends in InstanceNorm3d (~unit variance by
construction). Measured at init: fft_out.std() ~= 0.0007 vs fca_out.std()
~= 0.59 — fca_out's raw scale is ~863x LARGER. Without correcting for this,
`beta * fca_out` at beta=0.05 would contribute ~43x MORE than fft_out's own
scale to the fused output — the opposite of "FFT primary, FcaNet small
correction". This is also the most likely explanation for why Stage 5B's
free alpha barely moved (one branch's raw scale was negligible next to the
other's, leaving almost no loss gradient to prefer one over the other).
Fix: fca_out (and fca_be) are rescaled to match fft_out's (fft_be's)
per-(sample, channel) std BEFORE beta is applied, so beta's value has a
stable, scale-free meaning ("beta=0.05" ~= "5% of FFT's own scale") instead
of being applied to two arbitrarily-different raw magnitudes. See the
scale_ratio computation in forward() below.

CrossDomainAttention and the final AdaptiveRouter (spatial/frequency mix)
are UNCHANGED — same as Stage 5/5B, this module only replaces freq_branch.
"""
import os

import torch
import torch.nn as nn

from stage2_fcanet_plugin.fca_branch import FcaFrequencyBranch3D
from stage3_fft_branch.fft_branch import FFTFrequencyBranch3D

_DEBUG_NAN_CHECK = os.environ.get('FFTRESIDUAL_DEBUG_NAN', '1') != '0'

MAX_BETA = 0.2
INIT_BETA = 0.05
# logit(INIT_BETA / MAX_BETA) = logit(0.25) = ln(0.25/0.75) = ln(1/3)
import math
RAW_BETA_INIT = math.log(INIT_BETA / (MAX_BETA - INIT_BETA))


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
        print(f"[FFTResidualFusion DEBUG] non-finite at '{name}': "
              f"{n_nan} NaN + {n_inf} Inf / {t.numel()} elements  ({rng})")


class FFTMainResidualFusion(nn.Module):
    """
    Drop-in replacement for FcaFrequencyBranch3D / FFTFrequencyBranch3D /
    MultiExpertFrequencyRouter / FixedDualFrequencyFusion as _AFS_DSN_Base's
    freq_cls:
        __init__(channels)
        forward(x) -> (output, band_energies)
        shapes: (B, C, D, H, W) in, (B, C, D, H, W) out; band_energies (B, 8).

    FFT always runs as the primary branch; FcaNet always runs too, but only
    contributes a beta-scaled (beta <= MAX_BETA=0.2) residual correction —
    both experts execute on every forward call, no selection/bypass.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.fft    = FFTFrequencyBranch3D(channels)
        self.fcanet = FcaFrequencyBranch3D(channels)

        # beta[c] = MAX_BETA * sigmoid(raw_beta[c]); raw_beta init so that
        # beta starts at INIT_BETA=0.05 for every channel (see module docstring).
        self.raw_beta = nn.Parameter(torch.full((channels,), RAW_BETA_INIT))

    def _beta(self) -> torch.Tensor:
        """(channels,) in (0, MAX_BETA), differentiable w.r.t. raw_beta."""
        return MAX_BETA * torch.sigmoid(self.raw_beta)

    def get_beta_stats(self) -> dict:
        """
        Aggregate statistics over the per-channel residual weight beta, for
        logging (train_fft_residual.py dumps this to JSON after evaluation).
        Like Stage 5B's alpha, beta is a GLOBAL per-channel model parameter,
        not a function of input/OAR/case — same caveat applies here.
        """
        b = self._beta().detach()
        return {
            'mean_beta':                     b.mean().item(),
            'std_beta':                       b.std().item(),
            'min_beta':                       b.min().item(),
            'max_beta':                       b.max().item(),
            'fraction_channels_beta_gt_0_05': (b > 0.05).float().mean().item(),
            'fraction_channels_beta_gt_0_10': (b > 0.10).float().mean().item(),
            'fraction_channels_beta_gt_0_15': (b > 0.15).float().mean().item(),
        }

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, D, H, W) — bottleneck feature (GPU tensor)
        Returns (output, band_energies), same shapes as either expert alone.
        """
        _debug_check_finite('FFTMainResidualFusion.input', x)

        # Defensive clamp before EITHER expert (Stage 5/5B finding: FcaNet's
        # own InstanceNorm3d can overflow under fp16 autocast at extreme
        # input magnitude too, not just the FFT expert).
        x_clamped = torch.clamp(x, -1e4, 1e4)

        # FFT expert (primary): force fp32, autocast disabled (Stage 4/5
        # finding: rfftn on fp16 input produces ComplexHalf, crashed CUDA).
        with torch.autocast(device_type=x.device.type, enabled=False):
            fft_out, fft_be = self.fft(x_clamped.float())
        fft_out = fft_out.to(x.dtype)
        fft_be  = fft_be.to(x.dtype)
        _debug_check_finite('FFTMainResidualFusion.fft_out', fft_out)
        _debug_check_finite('FFTMainResidualFusion.fft_be', fft_be)

        fca_out, fca_be = self.fcanet(x_clamped)
        _debug_check_finite('FFTMainResidualFusion.fca_out', fca_out)
        _debug_check_finite('FFTMainResidualFusion.fca_be', fca_be)

        # channels derived dynamically from the experts' actual output, not
        # hard-coded: verify it matches what beta's shape was built from.
        assert fft_out.shape[1] == self.channels and fca_out.shape[1] == self.channels, (
            f"FFTMainResidualFusion: expert output channel count "
            f"(fft={fft_out.shape[1]}, fca={fca_out.shape[1]}) does not match "
            f"the channels={self.channels} this module (and beta) was built with."
        )

        fft_out = torch.clamp(fft_out, -1e4, 1e4)
        fca_out = torch.clamp(fca_out, -1e4, 1e4)
        fft_be  = torch.clamp(fft_be,  -1e4, 1e4)
        fca_be  = torch.clamp(fca_be,  -1e4, 1e4)

        beta = self._beta().to(x.dtype)             # (C,) in (0, MAX_BETA)
        beta_5d = beta[None, :, None, None, None]    # (1, C, 1, 1, 1)

        # Scale-match fca_out to fft_out's per-(sample, channel) spatial std
        # before applying beta. Diagnostic finding (see module docstring):
        # FFTFrequencyBranch3D's real_head/imag_head are deliberately
        # small-weight-initialised (std=1e-3, for ITS OWN training
        # stability — Stage 3 design, not touched here), so fft_out's raw
        # scale is ~863x smaller than fca_out's at initialisation (fca_out
        # ends in InstanceNorm3d, ~unit variance by construction). Without
        # this rescaling, beta=0.05 would make FcaNet's actual contribution
        # ~43x LARGER than FFT's own scale — the opposite of "FFT primary,
        # FcaNet small correction" — and is the same root cause identified
        # for why Stage 5B's alpha barely moved (one branch's raw scale was
        # negligible next to the other's, so the loss had very little
        # gradient signal to prefer one over the other). Rescaling
        # per-(sample, channel) so beta's value has a stable, scale-free
        # meaning ("beta=0.05" ~= "5% of fft_out's own scale on this
        # channel") regardless of how each branch's absolute magnitude
        # drifts over training.
        # Forced fp32, autocast disabled: stress-testing found that under
        # fp16 (--amp), a near-zero input can make fca_out underflow to
        # EXACTLY 0.0 in fp16, and the eps=1e-8 safety margin itself
        # underflows to 0.0 in fp16 too (fp16's smallest subnormal is
        # ~5.96e-8) -- so fca_scale+eps stays exactly 0, the ratio becomes
        # inf, and 0.0 * inf = NaN. fp32 has no such underflow at this
        # magnitude, so forcing it here removes the failure mode entirely.
        eps = 1e-8
        with torch.autocast(device_type=x.device.type, enabled=False):
            fft_out_f32 = fft_out.float()
            fca_out_f32 = fca_out.float()
            fft_scale = fft_out_f32.std(dim=[2, 3, 4], keepdim=True)   # (B, C, 1, 1, 1)
            fca_scale = fca_out_f32.std(dim=[2, 3, 4], keepdim=True)   # (B, C, 1, 1, 1)
            scale_ratio = fft_scale / (fca_scale + eps)
            fca_out_scaled = (fca_out_f32 * scale_ratio).to(x.dtype)
        _debug_check_finite('FFTMainResidualFusion.fca_out_scaled', fca_out_scaled)

        fused_feature = fft_out + beta_5d * fca_out_scaled

        # band_energies fallback: same shape-check discipline established in
        # Stage 5B (fail loudly, not a silently-wrong broadcast).
        if fft_be.shape != fca_be.shape:
            raise RuntimeError(
                f"FFTMainResidualFusion: band_energies shape mismatch "
                f"(fft={tuple(fft_be.shape)}, fcanet={tuple(fca_be.shape)}). "
                f"The scalar-weighted mix below assumes identical shapes "
                f"(verified true for FFTFrequencyBranch3D/FcaFrequencyBranch3D "
                f"when this module was written, both (B, 8)); add an explicit "
                f"projection before mixing rather than relying on broadcasting."
            )
        # Same scale-matching principle as fused_feature above, applied to
        # band_energies: empirically fft_be's raw scale is ~5x LARGER than
        # fca_be's (opposite direction from the spatial-feature case, but
        # still a real mismatch) — rescale fca_be to fft_be's std first so
        # beta_mean's value means the same thing here as it does for the
        # spatial mix.
        # Forced fp32 for the same reason as fca_out_scaled above (fp16
        # underflow of both the value and the epsilon -> 0/0-style NaN).
        be_eps = 1e-8
        with torch.autocast(device_type=x.device.type, enabled=False):
            fft_be_f32 = fft_be.float()
            fca_be_f32 = fca_be.float()
            fft_be_scale = fft_be_f32.std(dim=1, keepdim=True)   # (B, 1)
            fca_be_scale = fca_be_f32.std(dim=1, keepdim=True)   # (B, 1)
            fca_be_scaled = (fca_be_f32 * (fft_be_scale / (fca_be_scale + be_eps))).to(x.dtype)

        beta_mean = beta.mean()
        mixed_band_energies = fft_be + beta_mean * fca_be_scaled

        _debug_check_finite('FFTMainResidualFusion.fused_feature', fused_feature)
        _debug_check_finite('FFTMainResidualFusion.mixed_band_energies', mixed_band_energies)

        return fused_feature, mixed_band_energies
