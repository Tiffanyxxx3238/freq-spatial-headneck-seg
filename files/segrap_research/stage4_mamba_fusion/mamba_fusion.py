"""
Stage 4: Mamba SSM cross-domain fusion — replaces CrossDomainAttention's O(N^2)
bidirectional attention with an O(N) selective-scan SSM over a frequency-sorted
sequence.

CrossDomainAttention computes a (B, N, N) softmax attention matrix where
N = D*H*W spatial positions (afs_dsn_original.py CrossDomainAttention.forward,
`torch.bmm(qs.transpose(1,2), kf)`). This module replaces that O(N^2) cost with
a linear-time SSM scan over Nf = D*H*(W//2+1) frequency-domain positions.

Sequence construction (why frequency-sorted, not an arbitrary flatten):
  1. rfftn the frequency-branch output `freq` -> (B, C, D, H, Wh) complex.
  2. Each frequency-domain grid position has a well-defined radial frequency
     r(d,h,w) = sqrt(fd^2 + fh^2 + fw^2). Flattening positions in *that* order
     (ascending r: DC first, highest frequency last) gives the SSM a sequence
     with a genuine "low -> high frequency" axis — exactly the kind of ordered,
     causal-ish structure SSMs are designed to exploit (unlike a raw row-major
     spatial flatten, which has no such meaning).
  3. Each sequence step's feature vector is the per-channel magnitude at that
     frequency bin -> sequence shape (B, Nf, C).

Backend selection (try the official mamba-ssm first, fall back to pure PyTorch):
  - mamba_ssm ships custom CUDA kernels; compiling them requires a properly
    configured Linux + nvcc/ninja toolchain. This commonly fails on RunPod
    images without build-essential, and essentially always fails on Windows
    without WSL. The import is wrapped in try/except so failure is silent at
    import time; MambaBlock then decides per-instance whether to use the
    official kernel (import succeeded AND a CUDA device is present) or
    SimpleSelectiveScan (pure PyTorch, correct but O(L) sequential steps,
    used for local/CPU development and as an automatic fallback on RunPod if
    the official package fails to build).

Numerical-stability fixes (2026-06 incident: train loss -> NaN around epoch
65 in Stage 4A FcaNet+Mamba, never recovered through epoch 77). Root causes
identified in SimpleSelectiveScan, both now fixed below:

  1. exp(A_log) is unbounded above — under --amp (fp16 autocast, which wraps
     this module's forward since it's called from inside the model's
     `with torch.amp.autocast('cuda'):` block in train_one_epoch), fp16
     overflows once A_log drifts past ~11 (exp(11)=59874, fp16 max=65504),
     producing inf -> nan in the very next step. Fixed: A is now
     -(softplus(A_log) + a_min); softplus grows linearly, not exponentially,
     so it does not blow up fp16's range even for large A_log.
  2. Nothing floored A_bar = exp(delta*A) away from 1. If delta (also
     unconstrained via softplus) drifts toward 0, decay -> 1 and the
     recurrence becomes an unbounded running sum over the sequence — and
     longer sequences (Nf = D*H*(W//2+1), e.g. 320 for a 8^3 bottleneck)
     give more steps for this to accumulate before overflowing. Fixed: both
     A and delta now have additive floors (a_min, delta_min) so the decay
     factor is provably bounded away from 1 regardless of what the learned
     components do or how long Nf is — this also makes explicit
     length-dependent rescaling unnecessary (the geometric-series bound
     1/(1-rho) no longer depends on Nf once rho is bounded below 1).
  3. No normalisation anywhere in the scan. Fixed: LayerNorm before and
     after the recurrence, per-step state clamping as a hard backstop, and
     the entire scan now runs in fp32 with autocast explicitly disabled
     (recurrent accumulation is exactly the kind of op that's fragile under
     reduced precision; real Mamba implementations keep scan state in fp32
     for the same reason) — this directly addresses cause #1 even if some
     other path were to reintroduce a large exponent.
  4. No NaN/Inf visibility. Fixed: _debug_check_finite() probes at every
     major intermediate tensor, printing which one first goes non-finite.
     Enabled by default; disable with env var MAMBA_DEBUG_NAN=0 once
     stability is confirmed (each check is a cheap reduction, but it's not
     completely free at every one of these call sites every forward pass).

Gradient-level protection (utils/train_utils.py train_one_epoch) was
separately hardened to skip the optimizer step (not just clip) on a
non-finite grad norm, in both the AMP and non-AMP paths — see that file's
docstring/comments for why clip_grad_norm_ alone does not prevent permanent
parameter corruption from a single non-finite batch.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba
    _MAMBA_SSM_IMPORTED = True
except ImportError:
    _OfficialMamba = None
    _MAMBA_SSM_IMPORTED = False


# ------------------------------------------------------------------ #
# Debug: NaN/Inf probe                                                 #
# ------------------------------------------------------------------ #

_DEBUG_NAN_CHECK = os.environ.get('MAMBA_DEBUG_NAN', '1') != '0'


def _debug_check_finite(name: str, t: torch.Tensor) -> None:
    """
    Lightweight NaN/Inf probe. Prints which named tensor first goes
    non-finite, with the count and the range of any remaining finite values,
    so a divergence can be localised to a specific stage of the forward pass
    instead of only showing up as 'train_loss = nan' several layers later.
    Disable via env var MAMBA_DEBUG_NAN=0 once training is confirmed stable.
    """
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
        print(f"[MambaFusion DEBUG] non-finite at '{name}': "
              f"{n_nan} NaN + {n_inf} Inf / {t.numel()} elements  ({rng})")


# ------------------------------------------------------------------ #
# Pure-PyTorch selective-scan fallback                                 #
# ------------------------------------------------------------------ #

class SimpleSelectiveScan(nn.Module):
    """
    Pure-PyTorch fallback for mamba_ssm.Mamba (used when the official CUDA
    kernel is unavailable). Implements the same linear state-space recurrence
    Mamba is built on, simplified to a plain for-loop scan:

        delta_t = softplus(Linear(x_t)) + delta_min      (B, L, d_model), >= delta_min > 0
        A       = -(softplus(A_log) + a_min)              (d_model, d_state), <= -a_min < 0
        A_bar_t = exp(delta_t * A)                        in (0, exp(-delta_min*a_min)) — bounded away from 1
        state_t = A_bar_t * state_{t-1} + B(x_t) * x_t
        y_t     = C(x_t) . state_t + clamp(D) * x_t

    delta/B/C are input-dependent (Mamba's "selective" parameterisation,
    simplified to remove the custom kernel) so the decay rate adapts per
    sequence step rather than being a fixed LTI filter.

    Stability guarantees (see module docstring for the incident this fixes):
      - A uses softplus (linear growth) instead of exp (exponential growth)
        for its magnitude, so it cannot itself overflow fp16 the way
        exp(A_log) could once A_log drifted past ~11.
      - a_min/delta_min floors guarantee A_bar <= exp(-delta_min*a_min) < 1
        for every step, REGARDLESS of what the learned A_log/delta_proj
        converge to — this bounds the geometric-series sum
        sum_t A_bar^(T-t) * B_t*x_t <= max|B*x| / (1 - A_bar_max), a constant
        independent of sequence length L (so no separate length-based
        rescaling is needed once this floor is in place).
      - Pre/post LayerNorm bound the magnitude entering and leaving the scan.
      - Per-step state clamping is a hard backstop against any residual
        pathway producing extreme values.
      - The entire scan runs in fp32 with autocast explicitly disabled, since
        recurrent accumulation is fragile under fp16/bf16 and this module is
        invoked from inside the model's `torch.amp.autocast('cuda')` block
        during AMP training.

    All operations stay on x.device — no CPU/numpy round-trips — so this is a
    correct (if much slower, O(L) sequential steps with no kernel fusion)
    GPU-compatible stand-in.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 a_min: float = 0.5, delta_min: float = 0.05):
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.a_min     = a_min
        self.delta_min = delta_min

        self.A_log      = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B_proj     = nn.Linear(d_model, d_state, bias=False)
        self.C_proj     = nn.Linear(d_model, d_state, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.D          = nn.Parameter(torch.ones(d_model))

        self.pre_norm  = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        orig_dtype = x.dtype

        # Force fp32 for the whole scan regardless of an enclosing autocast
        # context — recurrent accumulation is exactly where fp16 silently
        # overflows (see module docstring, root-cause #1/#3).
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x = self.pre_norm(x)
            _debug_check_finite('SimpleSelectiveScan.pre_norm_out', x)

            B, L, Dm = x.shape
            dev, dtype = x.device, x.dtype

            delta = F.softplus(self.delta_proj(x)) + self.delta_min   # (B, L, d_model), >= delta_min
            A     = -(F.softplus(self.A_log) + self.a_min)             # (d_model, d_state), <= -a_min
            Bx    = self.B_proj(x)                                      # (B, L, d_state)
            Cx    = self.C_proj(x)                                       # (B, L, d_state)
            _debug_check_finite('SimpleSelectiveScan.delta', delta)
            _debug_check_finite('SimpleSelectiveScan.Bx', Bx)
            _debug_check_finite('SimpleSelectiveScan.Cx', Cx)

            state = torch.zeros(B, Dm, self.d_state, device=dev, dtype=dtype)
            ys = []
            for t in range(L):
                dt    = delta[:, t, :, None]                  # (B, d_model, 1)
                A_bar = torch.exp(dt * A[None])                # (B, d_model, d_state), bounded < 1
                Bx_t  = Bx[:, t, None, :]                        # (B, 1, d_state)
                x_t   = x[:, t, :, None]                          # (B, d_model, 1)
                state = A_bar * state + Bx_t * x_t                 # (B, d_model, d_state)
                state = torch.clamp(state, -1e4, 1e4)               # hard backstop
                ys.append((state * Cx[:, t, None, :]).sum(-1))       # (B, d_model)

            y = torch.stack(ys, dim=1)                            # (B, L, d_model)
            _debug_check_finite('SimpleSelectiveScan.scan_state_final', state)

            D_clamped = torch.clamp(self.D, -10.0, 10.0)
            y = y + x * D_clamped[None, None, :]
            y = self.post_norm(y)
            _debug_check_finite('SimpleSelectiveScan.output_fp32', y)

        return y.to(orig_dtype)


class MambaBlock(nn.Module):
    """
    Selects the official mamba_ssm CUDA kernel when both (a) the package
    imports successfully and (b) a CUDA device is present (the official
    kernels are CUDA-only and will error on CPU); otherwise uses
    SimpleSelectiveScan. Both expose forward(x: (B,L,d_model)) -> same shape.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        use_official = _MAMBA_SSM_IMPORTED and torch.cuda.is_available()

        if use_official:
            self.impl = _OfficialMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            )
            self.backend = 'mamba_ssm (official CUDA kernel)'
        else:
            self.impl = SimpleSelectiveScan(d_model, d_state=d_state)
            if not _MAMBA_SSM_IMPORTED:
                reason = 'mamba_ssm not installed'
            else:
                reason = 'mamba_ssm installed but no CUDA device available'
            self.backend = f'pure PyTorch fallback ({reason})'

        print(f"[MambaBlock] SSM backend: {self.backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.impl(x)
        _debug_check_finite(f'MambaBlock.output [{self.backend}]', out)
        return out


# ------------------------------------------------------------------ #
# Frequency-energy gate                                                #
# ------------------------------------------------------------------ #

class GatedFreqMamba(nn.Module):
    """
    Per-channel gate derived from the Mamba-scanned, frequency-sorted sequence.

    Pools the scanned sequence (mean over the Nf frequency-ordered steps) into
    a per-channel summary — this is the "frequency energy" signal — and maps
    it through a small FC + sigmoid to decide how much of the SSM's output
    mixes back into spatial / freq features in FreqSpatialMambaFusion.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, mamba_seq: torch.Tensor) -> torch.Tensor:
        """mamba_seq: (B, Nf, C) -> gate: (B, C) in (0, 1)"""
        pooled = mamba_seq.mean(dim=1)   # (B, C) — energy summary across the freq-sorted sequence
        _debug_check_finite('GatedFreqMamba.pooled', pooled)
        gate = self.fc(pooled)
        _debug_check_finite('GatedFreqMamba.gate', gate)
        return gate


# ------------------------------------------------------------------ #
# Drop-in replacement for CrossDomainAttention                         #
# ------------------------------------------------------------------ #

class FreqSpatialMambaFusion(nn.Module):
    """
    Drop-in replacement for CrossDomainAttention — identical interface:
        __init__(channels)
        forward(spatial, freq) -> (spatial_refined, freq_refined)
        shapes: (B, C, D, H, W) in, (B, C, D, H, W) out, unchanged.

    Replaces CrossDomainAttention's O(N^2) bidirectional attention
    (N = D*H*W) with:
      1. rfftn(freq) -> frequency-domain grid, flattened into a sequence of
         length Nf = D*H*(W//2+1), ORDERED by radial frequency ascending
         (DC/low-frequency first) — see module docstring for rationale.
      2. in_proj (C -> d_model) -> MambaBlock scan over the Nf-length
         sequence -> out_proj (d_model -> C).  O(Nf) instead of O(N^2).
      3. GatedFreqMamba pools the scanned sequence into a per-channel gate.
      4. Residual fusion (gamma_s / gamma_f start at 0, so the module is the
         identity at initialisation — mirrors CrossDomainAttention's own
         zero-initialised residual gammas):
           freq_refined    = freq    + gamma_f * gate * freq
           spatial_refined = spatial + gamma_s * gate * freq

    Input-scale assumption and why freq is normalised before step 1:
      `spatial` (the bottleneck feature `b`) always arrives already
      InstanceNorm3d-normalised, since it's the output of DoubleConv in
      _AFS_DSN_Base. `freq` has NO such guarantee — it depends entirely on
      which freq_cls produced it. FcaFrequencyBranch3D's output ends with
      InstanceNorm3d+LeakyReLU (bounded by construction: std~0.6 at init,
      empirically ~2.3 even after 200 adversarial training steps).
      FFTFrequencyBranch3D's output is the result of an un-normalised
      irfftn over a learned complex spectrum (real_head/imag_head are bare
      Conv3d layers with no following normalisation) — empirically std~0.0003
      at init (small-weight-initialised) but drifts to std~14, max~117 after
      the same 200 adversarial steps, ~10x FcaNet's drift, with no
      architectural ceiling. Stress-testing this module directly with
      synthetic freq tensors found the SSM scan's own pre_norm LayerNorm
      starts producing NaN once freq's raw magnitude reaches ~1e19-1e20
      (squaring during LayerNorm's variance computation overflows fp32)
      — finite at 1e18, non-finite at 1e20. That magnitude is far beyond
      anything observed in the 200-step stress test, but FFTFrequencyBranch3D
      has no normalisation layer to rule it out over a full training run.
      `freq` is therefore clamped + InstanceNorm3d'd into `freq_for_seq`
      before the FFT/sequence-construction pathway (steps 1-3), making the
      module's behaviour independent of which frequency branch feeds it.
      The residual fusion (step 4) still uses the original, un-normalised
      `freq` — that's a single bounded-gate multiply-add, not a recurrence,
      so it can't independently diverge the way the SSM scan can.
    """

    def __init__(self, channels: int, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.channels = channels
        self.freq_norm = nn.InstanceNorm3d(channels, affine=True)
        self.in_proj  = nn.Linear(channels, d_model)
        self.mamba    = MambaBlock(d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, channels)
        self.gated    = GatedFreqMamba(channels)

        self.gamma_s = nn.Parameter(torch.zeros(1))
        self.gamma_f = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _radial_grid(D: int, H: int, Wh: int, device, dtype) -> torch.Tensor:
        """
        Radial frequency magnitude per (d, h, w) bin of an rfftn output.
        Same formula as FFTFrequencyBranch3D._radial_band_energies; rfftfreq's
        first argument (Wh*2-1) always yields exactly Wh values for either
        parity of the original W.
        """
        fd = torch.fft.fftfreq(D, device=device).abs()
        fh = torch.fft.fftfreq(H, device=device).abs()
        fw = torch.fft.rfftfreq(Wh * 2 - 1, device=device)
        radial = (fd[:, None, None].pow(2)
                  + fh[None, :, None].pow(2)
                  + fw[None, None, :].pow(2)).sqrt()
        return radial.to(dtype)   # (D, H, Wh)

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor):
        """
        spatial, freq : (B, C, D, H, W) — GPU tensors throughout
        Returns (spatial_refined, freq_refined), same shapes as input.
        """
        B, C, D, H, W = spatial.shape
        _debug_check_finite('FreqSpatialMambaFusion.input_spatial', spatial)
        _debug_check_finite('FreqSpatialMambaFusion.input_freq', freq)

        # 1. FFT the frequency-branch output; build a frequency-sorted sequence
        #
        # Forced fp32, autocast disabled: under --amp, `freq` arrives here as
        # fp16 (cast by the enclosing torch.amp.autocast('cuda') block in
        # train_one_epoch). rfftn on a fp16 input produces a ComplexHalf
        # tensor — confirmed during Stage 4 smoke testing to be unsafe: it
        # triggered "ComplexHalf support is experimental and many operators
        # don't support it yet" followed by a fatal CUDA memory-allocation
        # failure that corrupted the whole CUDA context (not just a NaN —
        # an actual crash). This is treated as the primary suspect for the
        # original epoch-65 incident, ahead of the SimpleSelectiveScan
        # exp() fix above. Running the FFT in fp32 avoids ComplexHalf
        # entirely; the scope is narrow (just rfftn + abs()) so the rest of
        # the module still benefits from autocast's fp16 speedup.
        #
        # freq_for_seq: clamp (hard backstop against already-extreme values)
        # then InstanceNorm3d (branch-agnostic scale normalisation — see
        # class docstring for why FFTFrequencyBranch3D in particular needs
        # this where FcaFrequencyBranch3D's own InstanceNorm3d output mostly
        # wouldn't). Only used for the sequence/gate pathway below; the
        # residual fusion at the end still uses the original `freq`.
        freq_for_seq = self.freq_norm(torch.clamp(freq, -1e4, 1e4))
        _debug_check_finite('FreqSpatialMambaFusion.freq_for_seq', freq_for_seq)

        with torch.autocast(device_type=freq.device.type, enabled=False):
            fft_freq = torch.fft.rfftn(freq_for_seq.float(), dim=(-3, -2, -1))   # (B, C, D, H, Wh) complex64
            mag = fft_freq.abs()                                                  # (B, C, D, H, Wh) fp32
        _debug_check_finite('FreqSpatialMambaFusion.fft_magnitude', mag)
        Wh  = mag.shape[-1]
        Nf  = D * H * Wh

        radial = self._radial_grid(D, H, Wh, freq.device, mag.dtype).reshape(-1)  # (Nf,)
        order  = torch.argsort(radial)                            # ascending: low -> high frequency

        seq = mag.reshape(B, C, Nf)[:, :, order]                  # (B, C, Nf)  frequency-sorted
        seq = seq.transpose(1, 2)                                 # (B, Nf, C)  sequence-major

        # 2. SSM scan: O(Nf) instead of CrossDomainAttention's O(N^2)
        h = self.in_proj(seq)             # (B, Nf, d_model)
        h = self.mamba(h)                 # (B, Nf, d_model)
        h = self.out_proj(h)              # (B, Nf, C)
        _debug_check_finite('FreqSpatialMambaFusion.post_out_proj', h)

        # 3. Frequency-energy gate from the scanned sequence
        gate    = self.gated(h)           # (B, C) in (0, 1)
        gate_5d = gate[:, :, None, None, None]

        # 4. Residual cross-domain fusion (identity at init: gamma_s = gamma_f = 0)
        freq_refined    = freq    + self.gamma_f * gate_5d * freq
        spatial_refined = spatial + self.gamma_s * gate_5d * freq
        _debug_check_finite('FreqSpatialMambaFusion.spatial_refined', spatial_refined)
        _debug_check_finite('FreqSpatialMambaFusion.freq_refined', freq_refined)

        return spatial_refined, freq_refined
