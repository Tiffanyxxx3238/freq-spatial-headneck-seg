"""
Stage 2: nnU-Net + FcaNet plug-in integration.

Strategy: use MONAI's DynUNet as a drop-in nnU-Net backbone.
After each encoder stage output, register a forward hook that passes the
feature map through a FcaResBlock3D. This keeps backbone and plug-in decoupled.

Usage:
    python stage2_fcanet_plugin/nnunet_wrapper.py  (self-test)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from stage2_fcanet_plugin.fca_module import FcaResBlock3D

try:
    from monai.networks.nets import DynUNet
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


class FcaPluginHook:
    """
    Forward hook that wraps a FcaResBlock3D around an encoder stage output.
    Registered on a named module; output is modified in-place conceptually.
    """

    def __init__(self, fca_block: FcaResBlock3D):
        self.fca_block = fca_block

    def __call__(self, module, inputs, output):
        return self.fca_block(output)


class NNUNetWithFca(nn.Module):
    """
    DynUNet backbone with FcaResBlock3D plug-ins attached to encoder stages.

    Args:
        num_classes:    number of output classes (1 for binary, or N for multi-class)
        base_features:  first encoder stage feature count
        n_stages:       number of encoder stages (excluding bottleneck)
        use_fca:        if False, acts as plain nnU-Net baseline
        reduction:      FcaNet channel reduction ratio
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 46,  # 45 OARs + background
        spatial_size: tuple = (128, 128, 128),
        base_features: int = 32,
        n_stages: int = 5,
        use_fca: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        if not _MONAI_AVAILABLE:
            raise ImportError("monai is required for NNUNetWithFca. pip install monai")

        strides = [1] + [2] * (n_stages - 1)
        features = [min(base_features * (2 ** i), 512) for i in range(n_stages)]
        kernel_sizes = [[3, 3, 3]] * n_stages

        self.backbone = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernel_sizes,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=features,
            deep_supervision=False,
        )
        self.use_fca = use_fca
        self._hooks = []

        if use_fca:
            self._attach_fca_hooks(features, reduction)

    def _attach_fca_hooks(self, features, reduction):
        """Attach FcaResBlock3D hooks to each downsampling encoder block."""
        fca_blocks = nn.ModuleList()
        for i, ch in enumerate(features):
            fca_blocks.append(FcaResBlock3D(ch, reduction=reduction))
        # Store as a sub-module so parameters are registered
        self.fca_blocks = fca_blocks

        for i, (name, module) in enumerate(self.backbone.named_modules()):
            if 'downsamples' in name and name.count('.') == 1:
                hook = FcaPluginHook(self.fca_blocks[i % len(self.fca_blocks)])
                h = module.register_forward_hook(hook)
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x):
        return self.backbone(x)


# ------------------------------------------------------------------ #
# Training script for Stage 2                                          #
# ------------------------------------------------------------------ #

def train_stage2(args):
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from data.segrap_dataset import SegRapDataset, TASK001_PURE_OARS
    from models.losses import CombinedLoss
    from utils.train_utils import (
        WarmupCosineScheduler, train_one_epoch, validate, save_checkpoint, log_csv
    )
    from utils.metrics import evaluate_dataset

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'checkpoints/{args.exp_name}', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    train_ds = SegRapDataset(args.data_root, split='train', mode='all_oars')
    val_ds   = SegRapDataset(args.data_root, split='val',   mode='all_oars')
    test_ds  = SegRapDataset(args.data_root, split='test',  mode='all_oars')
    n_classes = len(TASK001_PURE_OARS) + 1  # background + 45

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    model = NNUNetWithFca(
        in_channels=1,
        num_classes=n_classes,
        use_fca=args.use_fca,
        base_features=args.base_features,
    ).to(device)

    criterion = CombinedLoss(num_classes=n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_dice = 0.0
    log_path = f'results/{args.exp_name}_log.csv'
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        print(f"[{epoch+1:03d}] train={train_loss:.4f}  val={val_loss:.4f}  dice={val_dice:.4f}")
        log_csv(log_path, {'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                           'val_dice': val_dice, 'use_fca': args.use_fca})
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(model, optimizer, epoch+1, best_val_dice,
                            f'checkpoints/{args.exp_name}/best.pth')

    from utils.train_utils import load_checkpoint
    load_checkpoint(model, None, f'checkpoints/{args.exp_name}/best.pth', device)
    evaluate_dataset(model, test_loader, device,
                     output_csv=f'results/{args.exp_name}_test_results.csv')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--base_features', type=int, default=32)
    p.add_argument('--use_fca', action='store_true', default=True)
    p.add_argument('--no_fca', dest='use_fca', action='store_false')
    p.add_argument('--exp_name', type=str, default='stage2_fcanet')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()
    train_stage2(args)
