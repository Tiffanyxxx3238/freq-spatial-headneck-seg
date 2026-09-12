"""
Stage 3A training: AFS-DSN with FFT frequency branch + standard CE+Dice loss.

Usage (from segrap_research/):
    python stage3_fft_branch/train.py \\
        --data_root ../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --cache_dir /workspace/data/cache \\
        --exp_name stage3_fft \\
        --epochs 100

Differences from stage1_focal_freq_loss/train.py:
  - Model: AFS_DSN_FFT (FFT spectral-gate branch, ~25 M params vs 414.6 M for V4)
  - Loss:  CombinedLoss (CE + Dice; no Focal Frequency Loss)
  - Removed: --lambda_ffl, --ffl_alpha
  - Removed: --model flag (only one model variant here)
  - Added:   param-count comparison print at startup
"""
import argparse
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.segrap_dataset import SegRapDataset
from models.afs_dsn_original import AFS_DSN_V4
from models.losses import CombinedLoss
from stage3_fft_branch.model import AFS_DSN_FFT
from utils.metrics import evaluate_dataset, batch_dice
from utils.train_utils import (
    save_checkpoint, load_checkpoint,
    WarmupCosineScheduler, train_one_epoch, validate, log_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=2)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--exp_name',     type=str,   default='stage3_fft')
    p.add_argument('--output_dir',   type=str,   default='.',
                   help='root dir; checkpoints go to <output_dir>/checkpoints/, '
                        'results to <output_dir>/results/')
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--resume',       type=str,   default='',
                   help='path to checkpoint to resume from')
    p.add_argument('--warm_start',   type=str,   default='',
                   help='path to pre-trained checkpoint for warm start')
    p.add_argument('--num_workers',  type=int,   default=0,
                   help='DataLoader workers (0 = main process, safe on Windows)')
    p.add_argument('--amp',          action='store_true', default=True)
    p.add_argument('--base_features', type=int,  default=32)
    p.add_argument('--max_samples',  type=int,   default=0,
                   help='Truncate dataset to N samples (0 = all). Use 4-8 for smoke tests.')
    p.add_argument('--cache_dir',    type=str,   default='/workspace/data/cache',
                   help='Directory for preprocessed .npz cache. Empty string = no caching.')
    p.add_argument('--seed',         type=int,   default=-1,
                   help='Random seed for reproducibility (torch / numpy / random). '
                        '-1 = no fixed seed, preserves original non-deterministic behaviour.')
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    if args.seed >= 0:
        _set_seed(args.seed)
        print(f"Random seed: {args.seed}  (torch / numpy / random all seeded)")
    else:
        print("Random seed: not fixed (--seed -1, default behaviour)")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pin = torch.cuda.is_available()
    ckpt_dir   = os.path.join(args.output_dir, 'checkpoints', args.exp_name)
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
    model = AFS_DSN_FFT(in_channels=1, num_classes=2, base_features=args.base_features,
                        use_freq_branch=True, use_cross_attention=True, use_router=True)
    n_params_fft = sum(p.numel() for p in model.parameters()) / 1e6

    # Param comparison with baseline V4 (instantiated cpu-only, not trained)
    _v4_ref = AFS_DSN_V4(in_channels=1, num_classes=2, base_features=args.base_features)
    n_params_v4 = sum(p.numel() for p in _v4_ref.parameters()) / 1e6
    del _v4_ref

    print(f"AFS_DSN_FFT params : {n_params_fft:.1f} M")
    print(f"AFS_DSN_V4  params : {n_params_v4:.1f} M  (baseline reference)")
    print(f"Param reduction    : {n_params_v4 / max(n_params_fft, 1e-3):.1f}×")

    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model is on: {next(model.parameters()).device}")
    assert device.type == 'cuda', "必須使用 GPU，請檢查 --device 參數"

    # ---- Loss / Optimizer / Scheduler ----
    criterion = CombinedLoss(num_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=args.epochs)
    scaler    = torch.amp.GradScaler('cuda') if args.amp and torch.cuda.is_available() else None

    start_epoch   = 0
    best_val_dice = 0.0

    if args.resume:
        start_epoch, best_val_dice = load_checkpoint(model, optimizer, args.resume, device)
    elif args.warm_start:
        load_checkpoint(model, None, args.warm_start, device)

    # ---- Training loop ----
    epoch_times = []
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc='Progress', unit='epoch',
        total=args.epochs, initial=start_epoch,
        position=0, leave=True, dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n}/{total} epochs [{elapsed}<{remaining}]',
    )
    for epoch in epoch_bar:
        t0 = time.time()
        scheduler.step(epoch)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch + 1, total_epochs=args.epochs,
        )
        gc.collect()
        torch.cuda.empty_cache()
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        gc.collect()
        torch.cuda.empty_cache()

        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)
        mins, secs = divmod(int(t_epoch), 60)
        time_str = f'{mins}m{secs:02d}s'

        lr      = scheduler.get_last_lr()[0]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice

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
            'epoch':      epoch + 1,
            'train_loss': f'{train_loss:.6f}',
            'val_loss':   f'{val_loss:.6f}',
            'val_dice':   f'{val_dice:.6f}',
            'lr':         f'{lr:.2e}',
        })

        if is_best:
            save_checkpoint(model, optimizer, epoch + 1, best_val_dice,
                            os.path.join(ckpt_dir, 'best.pth'), {'exp_name': args.exp_name})

    # ---- Final test evaluation ----
    print("\n=== Loading best model for test evaluation ===")
    load_checkpoint(model, None, os.path.join(ckpt_dir, 'best.pth'), device)
    evaluate_dataset(
        model, test_loader, device,
        output_csv=os.path.join(results_dir, f'{args.exp_name}_test_results.csv'),
    )


if __name__ == '__main__':
    main()
