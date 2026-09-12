"""
Stage 3A: FFT-based frequency branch for AFS-DSN.

Drop-in replacement for FrequencyBranchV4 (DWT-based) — identical interface:
  __init__(channels)
  forward(x) → (output, band_energies)
    output       : (B, C, D, H, W)   same shape as input x
    band_energies: (B, 8)             8 radial shells, compatible with AdaptiveRouter

AdaptiveRouter compatibility (models/afs_dsn_original.py line 266):
  be = band_energies.view(band_energies.shape[0], -1)[:, :8]
  Needs at least 8 elements after flatten.
  Original V4 returns (B, 8, C//4) → view(B, 8*C//4)[:, :8].
  This branch returns (B, 8)        → view(B, 8)    [:, :8]  ✓  all 8 radial energies.

Architecture:
  1. rfftn(x)  →  fft_out  (B, C, D, H, W//2+1)  complex
  2. Three real representations (log-compressed magnitude + cos/sin phase):
       magnitude  = log1p(|fft_out|)     avoids huge spectral dynamic range
       cos_phase  = cos(∠fft_out)        continuous, bounded ∈ [-1, 1]
       sin_phase  = sin(∠fft_out)        continuous, bounded ∈ [-1, 1]
  3. Each through an independent 3×3×3 conv block (C → C//4, IN3d+affine, LeakyReLU).
     Three separate weight sets; all operate on the (D, H, W//2+1) frequency volume.
  4. Concat → 1×1×1 fusion conv (3·C//4 → C//2, IN3d+affine, LeakyReLU).
  5. Two independent 1×1×1 Conv3d heads: real_head, imag_head → (B, C, D, H, W//2+1).
     Small-weight init (std=1e-3, bias=0) so the network starts near an identity-like
     operation and grows the frequency transformation gradually during training.
  6. complex_spectrum = torch.complex(real_part, imag_part).
  7. output = irfftn(complex_spectrum, s=(D,H,W), dim=(-3,-2,-1)) → (B, C, D, H, W).
     s= is mandatory: ensures correct spatial size regardless of W parity.
  8. band_energies: 8 equal-width radial shells of log-magnitude → (B, 8).
"""
import torch
import torch.nn as nn


class FFTFrequencyBranch3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        C  = channels
        Cf = max(channels // 4, 1)   # per-representation intermediate channels
        Ch = max(channels // 2, 1)   # fusion hidden channels

        def _conv_block(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, k, padding=k // 2, bias=False),
                nn.InstanceNorm3d(out_c, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )

        # Independent 3×3×3 blocks for each spectral representation
        self.mag_conv = _conv_block(C, Cf)            # log-magnitude  → C//4
        self.cos_conv = _conv_block(C, Cf)            # cos(phase)     → C//4
        self.sin_conv = _conv_block(C, Cf)            # sin(phase)     → C//4

        # 1×1×1 fusion: no further spatial mixing needed after independent per-bin processing
        self.fusion   = _conv_block(3 * Cf, Ch, k=1)  # C//4*3 → C//2

        # Two heads reconstruct the full complex spectrum channel-by-channel
        self.real_head = nn.Conv3d(Ch, C, 1)
        self.imag_head = nn.Conv3d(Ch, C, 1)

        # Small-weight init: at t=0 the output spectrum ≈ 0, encouraging stable early training
        for head in (self.real_head, self.imag_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------ #
    # Radial band energies                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _radial_band_energies(magnitude: torch.Tensor, n_bands: int = 8) -> torch.Tensor:
        """
        Partition the rfftn frequency volume into n_bands equal-width radial shells
        and return the mean log-magnitude energy per shell.

        magnitude : (B, C, D, H, Wh)  log1p-compressed, Wh = W//2+1
        returns   : (B, n_bands)       AdaptiveRouter takes view(B,-1)[:,:8] = all 8 ✓
        """
        B, C, D, H, Wh = magnitude.shape
        dev = magnitude.device

        # Normalised one-sided frequency coordinates per axis
        fd = torch.fft.fftfreq(D,           device=dev).abs()   # (D,)  ∈ [0, 0.5]
        fh = torch.fft.fftfreq(H,           device=dev).abs()   # (H,)  ∈ [0, 0.5]
        fw = torch.fft.rfftfreq(Wh * 2 - 1, device=dev)         # (Wh,) ∈ [0, 0.5]
        # rfftfreq(Wh*2-1) always returns exactly Wh values for both even and odd W

        radial   = (fd[:, None, None].pow(2)
                    + fh[None, :, None].pow(2)
                    + fw[None, None, :].pow(2)).sqrt()            # (D, H, Wh)
        r_max    = radial.max().clamp(min=1e-6)
        radial_n = (radial / r_max).view(-1)                     # (D*H*Wh,)

        # Channel-mean magnitude then flatten spatial
        mag_flat = magnitude.mean(dim=1).view(B, -1)             # (B, D*H*Wh)

        bounds   = torch.linspace(0.0, 1.0 + 1e-6, n_bands + 1, device=dev)
        energies = []
        for i in range(n_bands):
            mask = (radial_n >= bounds[i]) & (radial_n < bounds[i + 1])
            e = mag_flat[:, mask].mean(dim=1) if mask.any() else mag_flat.mean(dim=1)
            energies.append(e)

        return torch.stack(energies, dim=1)   # (B, n_bands=8)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor):
        """
        x            : (B, C, D, H, W)   bottleneck feature (GPU tensor)
        output       : (B, C, D, H, W)   frequency-refined spatial feature
        band_energies: (B, 8)             for AdaptiveRouter — view(B,-1)[:,:8] ✓
        """
        B, C, D, H, W = x.shape

        # 1. Forward FFT — all tensors remain on x.device throughout
        fft_out = torch.fft.rfftn(x, dim=(-3, -2, -1))    # (B, C, D, H, W//2+1) complex

        # 2. Real spectral representations
        magnitude = torch.log1p(torch.abs(fft_out))         # log-compressed ≥ 0
        phase     = torch.angle(fft_out)                    # ∈ [-π, π]
        cos_ph    = torch.cos(phase)                        # ∈ [-1, 1]  continuous
        sin_ph    = torch.sin(phase)                        # ∈ [-1, 1]  continuous

        # 3. Independent 3×3×3 conv blocks (operate on the (D, H, W//2+1) volume)
        m_feat = self.mag_conv(magnitude)                    # (B, C//4, D, H, W//2+1)
        c_feat = self.cos_conv(cos_ph)                       # (B, C//4, D, H, W//2+1)
        s_feat = self.sin_conv(sin_ph)                       # (B, C//4, D, H, W//2+1)

        # 4. 1×1×1 fusion across the three feature maps
        hidden = self.fusion(
            torch.cat([m_feat, c_feat, s_feat], dim=1)
        )                                                    # (B, C//2, D, H, W//2+1)

        # 5. Reconstruct complex spectrum — two independent heads
        real_part = self.real_head(hidden)                   # (B, C, D, H, W//2+1)
        imag_part = self.imag_head(hidden)                   # (B, C, D, H, W//2+1)
        complex_spectrum = torch.complex(real_part, imag_part)

        # 6. Inverse FFT: s= mandatory to restore exact original spatial size
        output = torch.fft.irfftn(complex_spectrum, s=(D, H, W), dim=(-3, -2, -1))

        # 7. Radial band energies from log-magnitude
        band_energies = self._radial_band_energies(magnitude, n_bands=8)   # (B, 8)

        return output, band_energies
