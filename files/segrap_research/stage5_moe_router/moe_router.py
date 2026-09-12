"""
Stage 5: Anatomy-aware Multi-Expert Frequency Routing.

Replaces _AFS_DSN_Base's freq_branch (NOT CrossDomainAttention, NOT the final
AdaptiveRouter — both stay exactly as in Stage 1-4). Interface is identical to
FcaFrequencyBranch3D / FFTFrequencyBranch3D / FFTFrequencyBranch3D:
    __init__(channels)
    forward(x) -> (output, band_energies)
    shapes: (B, C, D, H, W) in, (B, C, D, H, W) out; band_energies (B, 8).

Three experts:
  0. FcaNet expert    — stage2_fcanet_plugin.fca_branch.FcaFrequencyBranch3D (reused as-is)
  1. FFT expert       — stage3_fft_branch.fft_branch.FFTFrequencyBranch3D    (reused as-is)
  2. Identity expert  — IdentityResidualExpert (new, this file): Conv3d(C,C,1)
     + InstanceNorm3d residual transform, the lightest of the three.

Routing network: global mean+std pooling -> small FC -> 3 expert logits.
top_k experts (by logit) are selected per-sample; softmax is computed ONLY
over the selected top_k logits, then used both to mix expert outputs and to
mix band_energies.

Why top_k=2 is the training default, not top_k=1 (see model.py / train.py
for the same note — repeated here because it governs this module's design):
  softmax of a single selected logit is identically 1.0 regardless of that
  logit's value, so d(weight)/d(logit) = 0 for top_k=1 — the routing network
  gets no gradient signal from the *mixing weight* and can only ever learn
  through the indirect, non-differentiable effect of which index argmax/topk
  picks. With top_k=2, softmax([logit_a, logit_b]) genuinely depends on the
  difference logit_a - logit_b, so the router receives a real gradient and
  can learn anatomy-specific expert preferences. top_k=1 is offered only as
  an eval-time stress test of a trained top_k=2 model (harder sparsity), not
  as a training configuration.

Sparse computation ("group-by-expert", not per-sample Python loop):
  For each of the 3 experts (a small, fixed, constant-size loop — NOT a loop
  over batch size or sequence length, which is exactly the anti-pattern that
  made Stage 4's SimpleSelectiveScan loop slow), boolean-mask the batch to
  find which samples selected that expert, gather just that subset via
  fancy indexing, run ONE batched forward call for the subset, then scatter
  the weighted result back into the right batch rows via index_add_. Total
  Python-level iteration count is fixed at num_experts=3 regardless of batch
  size — see stress-test section in the project notes for a timing
  comparison confirming this is faster than computing all 3 experts densely.

Known risk carried over from Stage 3/4 (intentionally NOT patched upstream —
see Stage 4 discussion: user chose not to touch stage3_fft_branch/fft_branch.py):
  FFTFrequencyBranch3D calls rfftn directly on its input with no fp32-forcing
  and no final normalisation on its real_head/imag_head output. Under --amp
  this risks the same ComplexHalf crash and unbounded-magnitude drift found
  in Stage 4. Since this module reuses FFTFrequencyBranch3D unmodified (per
  spec), the defensive fix is applied HERE instead, at the point of use:
  the FFT expert's input subset is cast to fp32 and run with autocast
  disabled, and every expert's output subset is clamped before being mixed
  into the shared `output` tensor.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_fcanet_plugin.fca_branch import FcaFrequencyBranch3D
from stage3_fft_branch.fft_branch import FFTFrequencyBranch3D

NUM_EXPERTS = 3
EXPERT_NAMES = ['fcanet', 'fft', 'identity']
_FFT_EXPERT_IDX = EXPERT_NAMES.index('fft')


# ------------------------------------------------------------------ #
# Debug: NaN/Inf probe (same pattern as stage4_mamba_fusion.mamba_fusion,    #
# duplicated locally rather than cross-imported so stage5 has no stage4     #
# dependency — each stage stays independently runnable)                    #
# ------------------------------------------------------------------ #

_DEBUG_NAN_CHECK = os.environ.get('MOE_DEBUG_NAN', '1') != '0'


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
        print(f"[MoERouter DEBUG] non-finite at '{name}': "
              f"{n_nan} NaN + {n_inf} Inf / {t.numel()} elements  ({rng})")


# ------------------------------------------------------------------ #
# Identity / lightweight residual expert                              #
# ------------------------------------------------------------------ #

class IdentityResidualExpert(nn.Module):
    """
    Lightest of the three experts: a single 1x1x1 Conv3d + InstanceNorm3d
    residual transform. No frequency-domain processing at all — this expert
    represents "the bottleneck feature is already good enough as-is."

    band_energies is intentionally NOT all-zero. An all-zero band_energies
    would make AdaptiveRouter (downstream, unchanged) systematically read
    this expert's selection as "zero frequency-domain signal", biasing its
    spatial/frequency mixing decision against this expert regardless of
    whether the segmentation quality actually warranted picking it. Instead,
    pooled input statistics (mean/std/max over the input the expert
    received) are projected through a small FC into an 8-dim pseudo
    band-energy vector — comparable in scale/shape to the other two
    experts' genuine spectral energy profiles, so this expert competes
    fairly for AdaptiveRouter's attention.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.be_proj = nn.Sequential(
            nn.Linear(channels * 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.Softplus(),   # non-negative, matching the other experts' magnitude-derived energies
        )

    def forward(self, x: torch.Tensor):
        """x: (B, C, D, H, W) -> (output (B,C,D,H,W), band_energies (B,8))"""
        out = x + self.conv(x)   # lightweight residual transform

        stats = torch.cat([
            x.mean(dim=[2, 3, 4]),
            x.std(dim=[2, 3, 4]),
            x.amax(dim=[2, 3, 4]),
        ], dim=1)                              # (B, 3C)
        band_energies = self.be_proj(stats)    # (B, 8)

        return out, band_energies


# ------------------------------------------------------------------ #
# Multi-Expert Frequency Router                                        #
# ------------------------------------------------------------------ #

class MultiExpertFrequencyRouter(nn.Module):
    """
    Drop-in replacement for FcaFrequencyBranch3D / FFTFrequencyBranch3D as
    _AFS_DSN_Base's freq_cls:
        __init__(channels)
        forward(x) -> (output, band_energies)

    Routing diagnostics (expert_weights, selected_experts, routing_entropy)
    are NOT part of the return value (see module docstring in this package's
    README / the PR discussion for why): they are stored on
    `self.last_diagnostics` after every forward() call, read externally by
    train.py's evaluation loop via `model.freq_branch.last_diagnostics`.
    """

    def __init__(self, channels: int, top_k: int = 2):
        super().__init__()
        if not (1 <= top_k <= NUM_EXPERTS):
            raise ValueError(f"top_k must be in [1, {NUM_EXPERTS}], got {top_k}")
        self.channels = channels
        self.top_k = top_k

        self.experts = nn.ModuleList([
            FcaFrequencyBranch3D(channels),     # 0: fcanet
            FFTFrequencyBranch3D(channels),     # 1: fft
            IdentityResidualExpert(channels),   # 2: identity
        ])

        # Routing network: global mean+std pooling -> small FC -> 3 logits
        self.router_fc = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_EXPERTS),
        )

        self.last_diagnostics = None

    def _route(self, x: torch.Tensor):
        """
        x: (B, C, D, H, W) -> scores (B, NUM_EXPERTS)

        Forced fp32, autocast disabled: stress-testing found that under --amp,
        x.mean()/x.std() can reach a magnitude (e.g. ~1e6, plausible after
        many epochs of upstream drift) that overflows fp16 the instant
        router_fc's first Linear casts its input down to fp16 for the
        matmul (fp16 max ~65504) -- producing all-NaN scores from a single
        bad batch. Same root cause/fix pattern as Stage 4's freq_norm: this
        FC is tiny (channels*2 -> 64 -> 3, once per forward), so forcing
        fp32 costs nothing measurable while removing the overflow path
        entirely. Clamp is a cheap defensive backstop on top of that.
        """
        stats = torch.cat([x.mean(dim=[2, 3, 4]), x.std(dim=[2, 3, 4])], dim=1)  # (B, 2C)
        stats = torch.clamp(stats, -1e6, 1e6)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = self.router_fc(stats.float())   # (B, NUM_EXPERTS), fp32
        # Deliberately NOT cast back to x.dtype: scores/topk_weights/full_weights
        # stay fp32 throughout (negligible cost, these are (B, <=3)-sized) so the
        # routing-weight pathway never re-enters fp16 overflow territory. The
        # mixing step casts these weights to each out_sub's dtype individually.
        return scores

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, D, H, W) — bottleneck feature (GPU tensor)
        Returns (output, band_energies); diagnostics on self.last_diagnostics.
        """
        B, C, D, H, W = x.shape
        _debug_check_finite('MultiExpertFrequencyRouter.input', x)

        scores = self._route(x)                                   # (B, NUM_EXPERTS)
        _debug_check_finite('MultiExpertFrequencyRouter.scores', scores)

        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=1)   # (B, top_k)
        topk_weights = F.softmax(topk_scores, dim=1)                     # (B, top_k), sums to 1

        # Full (B, NUM_EXPERTS) weight matrix, zero for unselected experts.
        # Functional (not in-place) scatter so autograd tracks this cleanly.
        full_weights = torch.zeros(
            B, NUM_EXPERTS, device=x.device, dtype=topk_weights.dtype
        ).scatter(1, topk_idx, topk_weights)

        output = torch.zeros_like(x)
        band_energies = torch.zeros(B, 8, device=x.device, dtype=x.dtype)

        # Defensive clamp on the shared input, once, before any expert sees
        # it. Stress-testing found that FcaNet/Identity's own InstanceNorm3d
        # layers can produce NaN under fp16 autocast at extreme input
        # magnitude (same root cause as the routing network's overflow
        # above: squaring during variance computation overflows fp16's
        # ~65504 max) -- not just the FFT expert. One clamp on the full
        # tensor here is cheaper than clamping each expert's x_sub subset
        # separately, since with top_k=2 every sample appears in exactly 2
        # of the 3 subsets (i.e. would otherwise be clamped twice).
        x_clamped = torch.clamp(x, -1e4, 1e4)

        # Group-by-expert: fixed 3-iteration loop (NOT per-sample, NOT per-Nf-step).
        for e in range(NUM_EXPERTS):
            mask = full_weights[:, e] > 0          # (B,) — which samples picked expert e
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]    # (n_sub,)
            x_sub = x_clamped[idx]                    # (n_sub, C, D, H, W)
            w_sub = full_weights[idx, e]               # (n_sub,)

            if e == _FFT_EXPERT_IDX:
                # Known risk (Stage 3/4 finding, intentionally not patched
                # upstream): rfftn on a fp16 input produces ComplexHalf,
                # which previously crashed CUDA. Force fp32 + disable
                # autocast for just this expert's subset, cast back after.
                with torch.autocast(device_type=x.device.type, enabled=False):
                    out_sub, be_sub = self.experts[e](x_sub.float())
                out_sub = out_sub.to(x.dtype)
                be_sub = be_sub.to(x.dtype)
            else:
                out_sub, be_sub = self.experts[e](x_sub)

            _debug_check_finite(f'MultiExpertFrequencyRouter.expert[{EXPERT_NAMES[e]}].output', out_sub)
            _debug_check_finite(f'MultiExpertFrequencyRouter.expert[{EXPERT_NAMES[e]}].band_energies', be_sub)

            # Defensive backstop regardless of which expert produced this
            # (cheap, and protects against the FFT expert's unbounded
            # real_head/imag_head drift contaminating the mixed output).
            # Cast to x.dtype explicitly (not just relied-upon-implicitly):
            # index_add_ requires the source to match output's/band_energies'
            # dtype, and w_sub (fp32, from the fp32-forced routing network)
            # must match out_sub's dtype too before the elementwise multiply.
            out_sub = torch.clamp(out_sub, -1e4, 1e4).to(x.dtype)
            be_sub  = torch.clamp(be_sub,  -1e4, 1e4).to(x.dtype)
            w_sub_e = w_sub.to(x.dtype)

            w_sub_5d = w_sub_e.view(-1, 1, 1, 1, 1)
            output.index_add_(0, idx, out_sub * w_sub_5d)
            band_energies.index_add_(0, idx, be_sub * w_sub_e.view(-1, 1))

        _debug_check_finite('MultiExpertFrequencyRouter.output', output)
        _debug_check_finite('MultiExpertFrequencyRouter.band_energies', band_energies)

        entropy = -(topk_weights * topk_weights.clamp_min(1e-12).log()).sum(dim=1)   # (B,)

        self.last_diagnostics = {
            'expert_weights':    full_weights.detach(),
            'selected_experts':  topk_idx.detach(),
            'routing_entropy':   entropy.detach(),
        }

        return output, band_energies
