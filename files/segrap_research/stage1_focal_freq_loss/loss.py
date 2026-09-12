"""
Stage 1: Focal Frequency Loss (3D) for segmentation mask frequency supervision.

Reference: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
           adapted to 3-D volumetric segmentation masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.losses import CombinedLoss


class FocalFrequencyLoss3D(nn.Module):
    """
    3-D Focal Frequency Loss for segmentation.

    Steps:
      1. pred → softmax → (B, C, D, H, W)
      2. target → one-hot → (B, C, D, H, W)
      3. For each class channel: 3-D rFFT (Hermitian symmetry saves ~half compute)
      4. Per-bin squared error: err = |FFT(pred) - FFT(target)|^2
      5. Focal weight: w = err^(alpha/2)  (auto-upweights hard/high-freq bins)
      6. loss = mean(w * err)

    The rFFT output shape along the last dim is W//2+1, so the frequency tensor
    has shape (B, C, D, H, W//2+1) complex → real-valued magnitude ops.
    """

    def __init__(self, alpha: float = 1.0, patch_factor: int = 2, ave_spectrum: bool = False):
        super().__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (B, C, D, H, W) logits
        target : (B, D, H, W) long
        """
        B, C, D, H, W = pred.shape
        pred_soft = F.softmax(pred, dim=1)                          # (B, C, D, H, W)
        target_oh = F.one_hot(target, C).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        pf = self.patch_factor
        assert D % pf == 0 and H % pf == 0 and W % pf == 0, (
            f"target_size must be divisible by patch_factor={pf}"
        )
        pd, ph, pw = D // pf, H // pf, W // pf

        loss = pred.new_zeros(1)

        for di in range(pf):
            for hi in range(pf):
                for wi in range(pf):
                    p_patch = pred_soft[:, :,
                                        di*pd:(di+1)*pd,
                                        hi*ph:(hi+1)*ph,
                                        wi*pw:(wi+1)*pw]
                    t_patch = target_oh[:, :,
                                        di*pd:(di+1)*pd,
                                        hi*ph:(hi+1)*ph,
                                        wi*pw:(wi+1)*pw]

                    # 3-D rFFT: (B, C, pd, ph, pw//2+1) complex
                    fft_pred = torch.fft.rfftn(p_patch, dim=(-3, -2, -1))
                    fft_tgt  = torch.fft.rfftn(t_patch, dim=(-3, -2, -1))

                    if self.ave_spectrum:
                        fft_pred = fft_pred.mean(dim=0, keepdim=True)
                        fft_tgt  = fft_tgt.mean(dim=0, keepdim=True)

                    diff = fft_pred - fft_tgt
                    # Squared distance in complex plane = real^2 + imag^2
                    err = diff.real ** 2 + diff.imag ** 2  # (B, C, pd, ph, pw//2+1)

                    # Focal weight: normalise err to [0,1] per sample then raise to alpha
                    with torch.no_grad():
                        err_max = err.view(B, -1).max(dim=1).values.view(B, 1, 1, 1, 1).clamp(min=1e-8)
                        weight = (err / err_max) ** self.alpha

                    loss = loss + (weight * err).mean()

        loss = loss / (pf ** 3)
        return loss


class Stage1CombinedLoss(nn.Module):
    """
    CE + Dice + lambda_ffl * FocalFrequencyLoss3D.
    Drop-in replacement for the original CombinedLoss in Stage 1.
    """

    def __init__(self, lambda_ffl: float = 0.1, num_classes: int = 2, ffl_alpha: float = 1.0,
                 ffl_patch_factor: int = 2):
        super().__init__()
        self.lambda_ffl = lambda_ffl
        self.base_loss = CombinedLoss(num_classes=num_classes)
        self.ffl = FocalFrequencyLoss3D(alpha=ffl_alpha, patch_factor=ffl_patch_factor)

    def forward(self, pred, target):
        logits = pred['output'] if isinstance(pred, dict) else pred
        loss = self.base_loss(logits, target)
        if self.lambda_ffl > 0:
            loss = loss + self.lambda_ffl * self.ffl(logits, target)
        return loss
