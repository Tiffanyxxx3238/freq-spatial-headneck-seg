"""
Stage 1 training: original AFS-DSN architecture + Focal Frequency Loss.

Usage:
    cd segrap_research
    python stage1_focal_freq_loss/train.py \\
        --data_root ../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --lambda_ffl 0.1 --epochs 100 --exp_name stage1_ffl_0.1

To run the baseline (no FFL):
    python stage1_focal_freq_loss/train.py --lambda_ffl 0.0 --exp_name stage1_baseline
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats
from tqdm import tqdm
import numpy as np

from data.segrap_dataset import SegRapDataset
from models.afs_dsn_original import AFS_DSN_V4, AFS_DSN_Lite
from stage1_focal_freq_loss.loss import Stage1CombinedLoss
from utils.metrics import evaluate_dataset, batch_dice
from utils.train_utils import (
    save_checkpoint, load_checkpoint,
    WarmupCosineScheduler, train_one_epoch, validate, log_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--lambda_ffl', type=float, default=0.1)
    p.add_argument('--ffl_alpha', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--exp_name', type=str, default='stage1_ffl')
    p.add_argument('--output_dir', type=str, default='.',
                   help='root dir; checkpoints go to <output_dir>/checkpoints/, '
                        'results to <output_dir>/results/')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    p.add_argument('--warm_start', type=str, default='', help='path to pre-trained checkpoint for warm start')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers (0 = main process, safe on Windows)')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--model', type=str, default='v4', choices=['v4', 'lite'],
                   help='v4 = Full 414M (needs ~24 GB VRAM), lite = 27M (runs on 4 GB)')
    p.add_argument('--base_features', type=int, default=32)
    p.add_argument('--max_samples', type=int, default=0,
                   help='Truncate dataset to N samples (0 = all). Use 4-8 for smoke tests.')
    p.add_argument('--cache_dir', type=str, default='/workspace/data/cache',
                   help='Directory for preprocessed .npz cache. Empty string = no caching.')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pin = torch.cuda.is_available()
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints', args.exp_name)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{args.exp_name}_log.csv')

    # ---- Data ----
    train_ds = SegRapDataset(args.data_root, split='train', mode='thin_wall',
                             max_samples=args.max_samples, cache_dir=args.cache_dir)
    val_ds   = SegRapDataset(args.data_root, split='val',   mode='thin_wall',
                             max_samples=args.max_samples, cache_dir=args.cache_dir)
    test_ds  = SegRapDataset(args.data_root, split='test',  mode='thin_wall',
                             max_samples=args.max_samples, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ---- Model ----
    model_cls = AFS_DSN_Lite if args.model == 'lite' else AFS_DSN_V4
    model = model_cls(in_channels=1, num_classes=2, base_features=args.base_features,
                      use_freq_branch=True, use_cross_attention=True, use_router=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: AFS_DSN_{args.model.upper()}  params={n_params:.1f}M")
    model = model.to(device)

    print(f"Device: {device}")
    print(f"Model is on: {next(model.parameters()).device}")
    assert device.type == 'cuda', "必須使用 GPU，請檢查 --device 參數"

    # ---- Loss ----
    criterion = Stage1CombinedLoss(
        lambda_ffl=args.lambda_ffl,
        num_classes=2,
        ffl_alpha=args.ffl_alpha,
    )

    # ---- Optimizer / Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if args.amp and torch.cuda.is_available() else None

    start_epoch = 0
    best_val_dice = 0.0

    if args.resume:
        start_epoch, best_val_dice = load_checkpoint(model, optimizer, args.resume, device)
    elif args.warm_start:
        load_checkpoint(model, None, args.warm_start, device)

    # ---- Training loop ----
    epoch_times = []
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc='Progress',
        unit='epoch',
        total=args.epochs,
        initial=start_epoch,
        position=0,
        leave=True,
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n}/{total} epochs [{elapsed}<{remaining}]',
    )
    for epoch in epoch_bar:
        t0 = time.time()
        scheduler.step(epoch)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch + 1, total_epochs=args.epochs,
        )
        gc.collect()                    # release CPU RAM from training batches
        torch.cuda.empty_cache()        # release GPU cache before validation forward
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        gc.collect()
        torch.cuda.empty_cache()

        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)
        mins, secs = divmod(int(t_epoch), 60)
        time_str = f'{mins}m{secs:02d}s'

        lr = scheduler.get_last_lr()[0]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice

        # Rolling ETA from last 5 epochs
        avg_t = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        remaining_epochs = args.epochs - (epoch + 1)
        eta_m, eta_s = divmod(int(avg_t * remaining_epochs), 60)
        epoch_bar.set_postfix(
            dice=f'{val_dice:.4f}',
            best=f'{best_val_dice:.4f}',
            ETA=f'{eta_m}m{eta_s:02d}s',
        )

        best_tag = '  ← best!' if is_best else ''
        tqdm.write(
            f'   [{epoch+1:03d}/{args.epochs}] '
            f'train={train_loss:.4f}  val={val_loss:.4f}  dice={val_dice:.4f}  '
            f'lr={lr:.1e}  ⏱ {time_str}{best_tag}'
        )

        log_csv(log_path, {
            'epoch': epoch + 1,
            'train_loss': f'{train_loss:.6f}',
            'val_loss':   f'{val_loss:.6f}',
            'val_dice':   f'{val_dice:.6f}',
            'lr':         f'{lr:.2e}',
            'lambda_ffl': args.lambda_ffl,
        })

        if is_best:
            save_checkpoint(model, optimizer, epoch + 1, best_val_dice,
                            os.path.join(ckpt_dir, 'best.pth'), {'exp_name': args.exp_name})

    # ---- Final test evaluation ----
    print("\n=== Loading best model for test evaluation ===")
    load_checkpoint(model, None, os.path.join(ckpt_dir, 'best.pth'), device)
    test_df = evaluate_dataset(
        model, test_loader, device,
        output_csv=os.path.join(results_dir, f'{args.exp_name}_test_results.csv'),
    )

    # ---- If baseline exists, compute paired t-test ----
    baseline_csv = os.path.join(results_dir, 'stage1_baseline_test_results.csv')
    if args.lambda_ffl > 0 and os.path.exists(baseline_csv):
        import pandas as pd
        baseline_df = pd.read_csv(baseline_csv)
        merged = test_df.merge(baseline_df, on=['case_id', 'oar_name'], suffixes=('_ffl', '_base'))
        t_stat, p_val = scipy_stats.ttest_rel(merged['dice_ffl'], merged['dice_base'])
        print(f"\nPaired t-test (Dice FFL vs Baseline):  t={t_stat:.3f},  p={p_val:.4f}")
        thin = merged[merged['is_thin_wall_ffl']]
        t2, p2 = scipy_stats.ttest_rel(thin['dice_ffl'], thin['dice_base'])
        print(f"Paired t-test (Thin-wall only):         t={t2:.3f},  p={p2:.4f}")
        comp = {
            'exp': args.exp_name, 'lambda_ffl': args.lambda_ffl,
            'overall_dice_ffl': test_df['dice'].mean(),
            'overall_dice_base': baseline_df['dice'].mean(),
            'thinwall_dice_ffl': test_df[test_df['is_thin_wall']]['dice'].mean(),
            'thinwall_dice_base': baseline_df[baseline_df['is_thin_wall']]['dice'].mean(),
            'p_value_overall': p_val, 'p_value_thinwall': p2,
        }
        import csv
        comp_path = os.path.join(results_dir, 'stage1_comparison.csv')
        file_exists = os.path.exists(comp_path)
        with open(comp_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(comp.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(comp)
        print(f"Comparison saved to {comp_path}")


if __name__ == '__main__':
    main()
