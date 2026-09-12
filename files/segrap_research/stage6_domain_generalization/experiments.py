"""
Stage 6: Domain generalization experiments.

Experiment 1 — Leave-one-scanner-out (simulated via case-ID splits):
  SegRap2023 doesn't publish scanner metadata, so we use 3 case groups
  as proxy scanner cohorts:
    Group A: segrap_0000 – segrap_0039 (40 cases)
    Group B: segrap_0040 – segrap_0079 (40 cases)
    Group C: segrap_0080 – segrap_0119 (40 cases)
  Train on A+B, test on C; repeat for each held-out group.

Experiment 2 — Cross-dataset (if RADCURE labels are available):
  Train on SegRap2023, zero-shot evaluate on RADCURE thin-wall OAR subset.

Usage:
    cd segrap_research
    python stage6_domain_generalization/experiments.py \\
        --data_root ../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --exp leave_one_out
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

from data.segrap_dataset import SegRapDataset, THIN_WALL_OARS
from stage6_domain_generalization.spectral_domain_modulator import AFS_DSN_V2_DG
from stage1_focal_freq_loss.loss import Stage1CombinedLoss
from utils.train_utils import WarmupCosineScheduler, save_checkpoint, load_checkpoint, log_csv
from utils.metrics import evaluate_dataset

# Proxy scanner groups (40 cases each)
SCANNER_GROUPS = {
    'A': list(range(0, 40)),
    'B': list(range(40, 80)),
    'C': list(range(80, 120)),
}


def get_domain_id(case_id: str) -> int:
    """Map case to proxy domain index 0/1/2."""
    idx = int(case_id.split('_')[1])
    if idx < 40:
        return 0
    if idx < 80:
        return 1
    return 2


class SegRapDatasetWithDomain(SegRapDataset):
    """Extends SegRapDataset to return domain label."""

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        case_id = result[2]
        domain_id = get_domain_id(case_id)
        return result + (domain_id,)


def train_dg(data_root, train_cases, val_cases, exp_name, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)

    train_ds = SegRapDatasetWithDomain(data_root, split='train', mode='thin_wall')
    val_ds   = SegRapDatasetWithDomain(data_root, split='val', mode='thin_wall')

    # Filter to requested case indices
    def _filter(ds, allowed_ids):
        keep = [i for i, (cid, _) in enumerate(ds.samples) if cid in allowed_ids]
        return Subset(ds, keep)

    train_loader = DataLoader(_filter(train_ds, set(train_cases)),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(_filter(val_ds, set(val_cases)),
                              batch_size=1, shuffle=False, num_workers=2)

    model = AFS_DSN_V2_DG(in_channels=1, num_classes=2, base_features=32, n_domains=3).to(device)
    seg_criterion = Stage1CombinedLoss(lambda_ffl=args.lambda_ffl, num_classes=2)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=args.epochs)

    best_val_dice = 0.0
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            domain_labels = torch.tensor(
                [get_domain_id(cid) for cid in batch['case_id']], dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            out = model(images)
            loss_seg = seg_criterion(out, labels)
            loss_dom = domain_criterion(out['domain_logits'], domain_labels)
            aux = out['aux_loss'] if out['aux_loss'] is not None else 0.0
            loss = loss_seg + args.lambda_domain * loss_dom + 0.01 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Quick val
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                out = model(images)
                from utils.metrics import batch_dice
                val_dice += batch_dice(out['output'], labels)
        val_dice /= max(len(val_loader), 1)
        print(f"[{exp_name}][{epoch+1:03d}] val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(model, optimizer, epoch+1, best_val_dice,
                            f'checkpoints/{exp_name}/best.pth')

    return model, best_val_dice


def leave_one_scanner_out(args):
    results = []
    for held_out in ['A', 'B', 'C']:
        train_ids = []
        val_ids = []
        for g, idxs in SCANNER_GROUPS.items():
            ids = [f'segrap_{i:04d}' for i in idxs]
            if g == held_out:
                val_ids = ids
            else:
                train_ids.extend(ids)

        exp_name = f'stage6_loso_{held_out}'
        print(f"\n{'='*60}")
        print(f"Leave-One-Scanner-Out: held out group {held_out}")
        print('='*60)

        model, best_val_dice = train_dg(
            args.data_root, train_ids, val_ids, exp_name, args
        )

        # Test on held-out scanner
        test_ds = SegRapDataset(args.data_root, split='val', mode='thin_wall')
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        load_checkpoint(model, None, f'checkpoints/{exp_name}/best.pth',
                        args.device if torch.cuda.is_available() else 'cpu')
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        test_df = evaluate_dataset(model, test_loader, device,
                                   output_csv=f'results/{exp_name}_test.csv')
        results.append({
            'held_out_group': held_out,
            'overall_dice': test_df['dice'].mean(),
            'thinwall_dice': test_df[test_df['is_thin_wall']]['dice'].mean(),
            'surface_dice_1mm': test_df['surface_dice_1mm'].mean(),
        })

    summary = pd.DataFrame(results)
    summary.to_csv('results/stage6_loso_summary.csv', index=False)
    print("\nLeave-One-Scanner-Out summary:")
    print(summary.to_string(index=False))

    # Domain gap
    summary['domain_gap'] = summary['overall_dice'].max() - summary['overall_dice'].min()
    print(f"Domain gap (max-min Dice): {summary['domain_gap'].iloc[0]:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--exp', type=str, default='leave_one_out',
                   choices=['leave_one_out'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda_ffl', type=float, default=0.1)
    p.add_argument('--lambda_domain', type=float, default=0.1)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.exp == 'leave_one_out':
        leave_one_scanner_out(args)
