"""
Stage 6: Frequency Domain Generalization.

Spectral Domain Modulator (SDM) separates FFT features into:
  - domain_invariant: low-frequency structure (stable across scanners)
  - domain_specific:  high-frequency texture/noise (scanner-dependent)

Training:
  domain_invariant → main segmentation task (Dice + FFL)
  domain_specific  → Gradient Reversal Layer → Domain Classifier
                     (adversarial: model learns these features are not discriminative)

Reference concept: WaveRNet Spectral-guided Domain Modulator (arXiv 2025).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Gradient Reversal Layer                                              #
# ------------------------------------------------------------------ #

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ------------------------------------------------------------------ #
# Domain Classifier                                                    #
# ------------------------------------------------------------------ #

class DomainClassifier(nn.Module):
    """
    Predicts domain (scanner/institution) index from domain-specific features.
    n_domains: number of distinct scanners/institutions in training set.
    """

    def __init__(self, in_channels: int, n_domains: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)   # (B, C)
        return self.fc(pooled)             # (B, n_domains)


# ------------------------------------------------------------------ #
# Spectral Domain Modulator                                            #
# ------------------------------------------------------------------ #

class SpectralDomainModulator(nn.Module):
    """
    Splits FFT branch output into domain-invariant and domain-specific components.

    Decomposition:
      - low-freq (|k| < threshold): anatomy structure, cross-scanner stable
      - high-freq (|k| >= threshold): scanner noise/texture, domain-specific

    Both components are processed by small conv blocks.
    The domain_specific features then go through GRL + DomainClassifier for
    adversarial training.
    """

    def __init__(self, channels: int, n_domains: int = 2, freq_threshold: float = 0.3):
        super().__init__()
        self.channels = channels
        self.freq_threshold = freq_threshold

        # Invariant branch: process low-freq features
        self.invariant_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Specific branch: process high-freq features
        self.specific_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # GRL + domain classifier
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = DomainClassifier(channels, n_domains)

        # Fusion: combine invariant + attenuated specific → output
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, D, H, W) — FFT branch output
        returns:
          fused_feat:      (B, C, D, H, W) — for segmentation
          domain_logits:   (B, n_domains)  — for adversarial domain loss
        """
        B, C, D, H, W = x.shape
        W_rfft = W // 2 + 1

        # FFT decomposition
        fft_x = torch.fft.rfftn(x, dim=(-3, -2, -1))   # (B, C, D, H, W_rfft) complex

        # Frequency radius mask
        device = x.device
        fd = torch.fft.fftfreq(D, device=device).abs()
        fh = torch.fft.fftfreq(H, device=device).abs()
        fw = torch.arange(W_rfft, dtype=torch.float32, device=device) / W
        radius = (fd[:, None, None] ** 2 + fh[None, :, None] ** 2 + fw[None, None, :] ** 2).sqrt()
        radius = radius / (radius.max() + 1e-8)   # normalise to [0, 1]

        low_mask  = (radius < self.freq_threshold).float()
        high_mask = (radius >= self.freq_threshold).float()

        # Split in frequency domain, back to spatial
        x_low  = torch.fft.irfftn(fft_x * low_mask,  s=(D, H, W), dim=(-3, -2, -1))
        x_high = torch.fft.irfftn(fft_x * high_mask, s=(D, H, W), dim=(-3, -2, -1))

        invariant_feat = self.invariant_conv(x_low)    # (B, C, D, H, W)
        specific_feat  = self.specific_conv(x_high)    # (B, C, D, H, W)

        # Adversarial domain prediction from specific features
        domain_logits = self.domain_classifier(self.grl(specific_feat))

        # Fuse for main task: invariant + (weakened) specific
        fused_feat = self.fusion(torch.cat([invariant_feat, specific_feat], dim=1))

        return fused_feat, domain_logits


# ------------------------------------------------------------------ #
# AFS-DSN-V2 with Domain Generalization                                #
# ------------------------------------------------------------------ #

class AFS_DSN_V2_DG(nn.Module):
    """
    AFS-DSN-V2 (Stage 5) + Spectral Domain Modulator (Stage 6).
    The SDM is inserted between FFT branch output and Mamba fusion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        n_domains: int = 2,
        freq_threshold: float = 0.3,
        lambda_domain: float = 0.1,
        fusion_type: str = 'gated_mamba',
        n_experts: int = 4,
        top_k: int = 2,
        d_state: int = 16,
    ):
        super().__init__()
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from models.afs_dsn_original import DoubleConv, AdaptiveRouter
        from stage3_fft_branch.fft_branch import FFTFrequencyBranch3D
        from stage4_mamba_fusion.mamba_fusion import GatedFreqMamba
        from stage5_moe_router.moe_router import MultiExpertFrequencyRouter

        f = base_features
        self.lambda_domain = lambda_domain

        self.encoder1 = DoubleConv(in_channels, f)
        self.encoder2 = DoubleConv(f, f * 2)
        self.encoder3 = DoubleConv(f * 2, f * 4)
        self.encoder4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.freq_branch = FFTFrequencyBranch3D(f * 16)
        self.sdm = SpectralDomainModulator(f * 16, n_domains, freq_threshold)
        self.fusion = GatedFreqMamba(f * 16, d_state)
        self.router = MultiExpertFrequencyRouter(f * 16, n_experts, top_k)

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

        freq_raw, band_energies = self.freq_branch(b)
        freq_feat, domain_logits = self.sdm(freq_raw)

        spatial_refined, freq_refined = self.fusion(b, freq_feat)
        fused, aux_loss = self.router(spatial_refined, freq_refined, band_energies)

        d4 = self.decoder4(torch.cat([self.up4(fused), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return {
            'output': self.final(d1),
            'domain_logits': domain_logits,
            'aux_loss': aux_loss,
            'band_energies': band_energies,
        }
