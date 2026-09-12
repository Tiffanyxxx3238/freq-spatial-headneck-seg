"""
AFS-DSN original architecture, extracted verbatim from 02_model.ipynb.
Supports Full (V4, ~414M) and Lite (~27M) variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class _DWT3DLayer(nn.Module):
    """GPU-native single-level 3D DWT via separable 1D convolutions.

    Filter coefficients are read from pywt once at init and stored as
    buffers (automatically moved to GPU with .to(device)).  The forward
    pass contains no CPU/numpy calls — all computation stays on the GPU.
    """

    def __init__(self, wavelet_name: str):
        super().__init__()
        w = pywt.Wavelet(wavelet_name)
        # F.conv1d is cross-correlation; flip filters to match pywt's convolution convention.
        lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        self.register_buffer('lo', lo)
        self.register_buffer('hi', hi)

    def _apply_1d(self, x: torch.Tensor, filt: torch.Tensor, dim: int) -> torch.Tensor:
        """Stride-2 filtering along spatial dim (0=D, 1=H, 2=W) with periodic padding."""
        spatial = 2 + dim  # tensor axis: 2 for D, 3 for H, 4 for W
        K = filt.shape[0]

        # Bring the target axis to the last position so we can flatten safely.
        if dim < 2:
            perm = list(range(5))
            perm.pop(spatial)
            perm.append(spatial)
            x = x.permute(*perm).contiguous()

        L = x.shape[-1]
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, 1, L)           # (N, 1, L)

        # Symmetric periodic padding matching pywt's 'periodization' convention:
        #   prepend K//2-1 elements from the END  (left boundary)
        #   append  K//2   elements from the START (right boundary)
        # Total extra samples = K-1, giving output length ceil(L/2).
        if K > 1:
            left_pad  = K // 2 - 1
            right_pad = K // 2
            parts = []
            if left_pad > 0:
                parts.append(x_flat[..., -left_pad:])
            parts.append(x_flat)
            parts.append(x_flat[..., :right_pad])
            x_flat = torch.cat(parts, dim=-1)

        w = filt.to(x_flat.dtype).view(1, 1, -1)
        out = F.conv1d(x_flat, w, stride=2)    # (N, 1, ceil(L/2))

        out = out.reshape(*prefix, out.shape[-1])

        # Invert the permutation.
        if dim < 2:
            inv = [0] * 5
            for i, p in enumerate(perm):
                inv[p] = i
            out = out.permute(*inv).contiguous()

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, D, H, W)
        out : (B, C*8, D//2, H//2, W//2)
        Band order: aaa, aad, ada, add, daa, dad, dda, ddd  (matches pywt.dwtn)
        """
        lo, hi = self.lo, self.hi
        # Apply separable filters: D axis first (first letter), then H, then W.
        xL  = self._apply_1d(x,  lo, 0);  xH  = self._apply_1d(x,  hi, 0)
        xLL = self._apply_1d(xL, lo, 1);  xLH = self._apply_1d(xL, hi, 1)
        xHL = self._apply_1d(xH, lo, 1);  xHH = self._apply_1d(xH, hi, 1)
        bands = torch.stack([
            self._apply_1d(xLL, lo, 2),  # aaa
            self._apply_1d(xLL, hi, 2),  # aad
            self._apply_1d(xLH, lo, 2),  # ada
            self._apply_1d(xLH, hi, 2),  # add
            self._apply_1d(xHL, lo, 2),  # daa
            self._apply_1d(xHL, hi, 2),  # dad
            self._apply_1d(xHH, lo, 2),  # dda
            self._apply_1d(xHH, hi, 2),  # ddd
        ], dim=2)                          # (B, C, 8, D//2, H//2, W//2)
        B, C = x.shape[:2]
        D2, H2, W2 = bands.shape[3], bands.shape[4], bands.shape[5]
        return bands.reshape(B, C * 8, D2, H2, W2)


class MultiScaleWavelet3D(nn.Module):
    def __init__(self, wavelet_scales=None):
        super().__init__()
        self.wavelet_scales = wavelet_scales or ['db1', 'db2', 'db4']
        self.dwt_layers = nn.ModuleList([
            _DWT3DLayer(wname) for wname in self.wavelet_scales
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same interface as before: (B, C, D, H, W) → (B, C*8*n_scales, D//2, H//2, W//2)
        return torch.cat([layer(x) for layer in self.dwt_layers], dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), nn.LeakyReLU(0.01, inplace=True),
        )
        self.residual = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class FrequencyBranchV4(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wavelet_scales = ['db1', 'db2', 'db4']
        self.multiscale_wavelet = MultiScaleWavelet3D(self.wavelet_scales)
        total_bands = 24
        self.band_weights = nn.Parameter(torch.ones(total_bands) / total_bands)
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels // 4, 1, bias=False),
                nn.InstanceNorm3d(channels // 4), nn.LeakyReLU(0.01),
            )
            for _ in range(total_bands)
        ])
        self.process = nn.Sequential(
            self._dconv(channels * 6, channels * 4),
            self._dconv(channels * 4, channels * 2),
            self._dconv(channels * 2, channels),
        )

    def _dconv(self, inc, outc):
        return nn.Sequential(
            nn.Conv3d(inc, outc, 3, padding=1, bias=False),
            nn.InstanceNorm3d(outc), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(outc, outc, 3, padding=1, bias=False),
            nn.InstanceNorm3d(outc), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        wc = self.multiscale_wavelet(x)
        band_features = []
        for i in range(len(self.band_weights)):
            band = wc[:, i * C:(i + 1) * C]
            feat = self.band_convs[i](band) * self.band_weights[i]
            feat = F.interpolate(feat, size=(D, H, W), mode='trilinear', align_corners=False)
            band_features.append(feat)
        merged = torch.cat(band_features, dim=1)
        output = self.process(merged)
        band_energies = torch.stack(
            [torch.mean(torch.abs(f), dim=[2, 3, 4]) for f in band_features[:8]], dim=1
        )
        return output, band_energies


class FrequencyBranchLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wavelet_scales = ['db1', 'db2', 'db4']
        self.multiscale_wavelet = MultiScaleWavelet3D(self.wavelet_scales)
        total_bands = 24
        self.band_weights = nn.Parameter(torch.ones(total_bands) / total_bands)
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels // 4, 1, bias=False),
                nn.InstanceNorm3d(channels // 4), nn.LeakyReLU(0.01),
            )
            for _ in range(total_bands)
        ])
        self.pw_reduce = nn.Sequential(
            nn.Conv3d(channels * 6, channels, 1, bias=False),
            nn.InstanceNorm3d(channels), nn.LeakyReLU(0.01, inplace=True),
        )
        self.dw1 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels), nn.LeakyReLU(0.01, inplace=True),
        )
        self.dw2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.InstanceNorm3d(channels), nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        wc = self.multiscale_wavelet(x)
        band_features = []
        for i in range(len(self.band_weights)):
            band = wc[:, i * C:(i + 1) * C]
            feat = self.band_convs[i](band) * self.band_weights[i]
            feat = F.interpolate(feat, size=(D, H, W), mode='trilinear', align_corners=False)
            band_features.append(feat)
        merged = torch.cat(band_features, dim=1)
        out = self.pw_reduce(merged)
        out = self.dw1(out) + out
        out = self.dw2(out) + out
        band_energies = torch.stack(
            [torch.mean(torch.abs(f), dim=[2, 3, 4]) for f in band_features[:8]], dim=1
        )
        return out, band_energies


class CrossDomainAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_spatial = nn.Conv3d(channels, channels // 8, 1)
        self.k_freq = nn.Conv3d(channels, channels // 8, 1)
        self.v_freq = nn.Conv3d(channels, channels, 1)
        self.q_freq = nn.Conv3d(channels, channels // 8, 1)
        self.k_spatial = nn.Conv3d(channels, channels // 8, 1)
        self.v_spatial = nn.Conv3d(channels, channels, 1)
        self.gamma_s = nn.Parameter(torch.zeros(1))
        self.gamma_f = nn.Parameter(torch.zeros(1))

    def forward(self, spatial, freq):
        B, C, D, H, W = spatial.shape
        N = D * H * W
        qs = self.q_spatial(spatial).view(B, -1, N)
        kf = self.k_freq(freq).view(B, -1, N)
        vf = self.v_freq(freq).view(B, -1, N)
        attn_s = F.softmax(torch.bmm(qs.transpose(1, 2), kf), dim=-1)
        out_s = torch.bmm(vf, attn_s.transpose(1, 2)).view(B, C, D, H, W)
        spatial_refined = spatial + self.gamma_s * out_s

        qf = self.q_freq(freq).view(B, -1, N)
        ks = self.k_spatial(spatial).view(B, -1, N)
        vs = self.v_spatial(spatial).view(B, -1, N)
        attn_f = F.softmax(torch.bmm(qf.transpose(1, 2), ks), dim=-1)
        out_f = torch.bmm(vs, attn_f.transpose(1, 2)).view(B, C, D, H, W)
        return spatial_refined, freq + self.gamma_f * out_f


class AdaptiveRouter(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * 2 + 8, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2), nn.Softmax(dim=1),
        )

    def forward(self, spatial, freq, band_energies):
        stats = torch.cat(
            [spatial.mean(dim=[2, 3, 4]), spatial.std(dim=[2, 3, 4])], dim=1
        )
        be = band_energies.view(band_energies.shape[0], -1)[:, :8]
        combined = torch.cat([stats, be], dim=1)
        return self.fc(combined)


class _AFS_DSN_Base(nn.Module):
    def __init__(self, in_ch, num_classes, base_features,
                 use_freq_branch, use_cross_attention, use_router, freq_cls,
                 fusion_cls=CrossDomainAttention):
        super().__init__()
        f = base_features
        self.use_freq_branch = use_freq_branch
        self.use_cross_attention = use_cross_attention and use_freq_branch
        self.use_router = use_router and use_freq_branch

        self.encoder1 = DoubleConv(in_ch, f)
        self.encoder2 = DoubleConv(f, f * 2)
        self.encoder3 = DoubleConv(f * 2, f * 4)
        self.encoder4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)

        if self.use_freq_branch:
            self.freq_branch = freq_cls(f * 16)
        if self.use_cross_attention:
            self.cross_attention = fusion_cls(f * 16)
        if self.use_router:
            self.router = AdaptiveRouter(channels=f * 16)

        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, 2, stride=2)
        self.decoder4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.decoder3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.decoder2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.decoder1 = DoubleConv(f * 2, f)
        self.final = nn.Conv3d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        routing_weights = band_energies = None
        if self.use_freq_branch:
            freq_feat, band_energies = self.freq_branch(b)
            if self.use_cross_attention:
                spatial_refined, freq_refined = self.cross_attention(b, freq_feat)
            else:
                spatial_refined, freq_refined = b, freq_feat
            if self.use_router:
                routing_weights = self.router(spatial_refined, freq_refined, band_energies)
                w_s = routing_weights[:, 0:1, None, None, None]
                w_f = routing_weights[:, 1:2, None, None, None]
                b = spatial_refined * w_s + freq_refined * w_f
            else:
                b = (spatial_refined + freq_refined) / 2

        d4 = self.decoder4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return {
            'output': self.final(d1),
            'routing_weights': routing_weights,
            'band_energies': band_energies,
        }


class AFS_DSN_V4(_AFS_DSN_Base):
    """Full model (~414M params, paper original)."""
    def __init__(self, in_channels=1, num_classes=2, base_features=32,
                 use_freq_branch=True, use_cross_attention=True, use_router=True):
        super().__init__(in_channels, num_classes, base_features,
                         use_freq_branch, use_cross_attention, use_router, FrequencyBranchV4)


class AFS_DSN_Lite(_AFS_DSN_Base):
    """Lite model with depthwise-separable freq branch (~27M params)."""
    def __init__(self, in_channels=1, num_classes=2, base_features=32,
                 use_freq_branch=True, use_cross_attention=True, use_router=True):
        super().__init__(in_channels, num_classes, base_features,
                         use_freq_branch, use_cross_attention, use_router, FrequencyBranchLite)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
