"""
External Baseline Study training: 3D U-Net / SegResNet / MedNeXt-S on the
EXACT SAME SegRap2023 thin-wall OAR binary crop pipeline as Stage5C.

This is the "External Baseline Study" (in any paper text: "Comparison with
External Medical Segmentation Baselines") — NOT Stage6 (domain
generalisation, deferred), NOT Stage7. See external_baselines/README.md.

Train loop / eval flow copied verbatim from
stage5_moe_router/train_fft_residual.py — dataset construction, DataLoader
settings, loss, optimizer, scheduler, AMP, checkpoint selection (best val
Dice), and final-test evaluation via utils.metrics.evaluate_dataset are all
unchanged. Only the model differs (and there is no alpha/beta diagnostic to
log, since these are plain backbones with no learnable fusion weight).

Usage:
    python external_baselines/train_external.py \\
        --model_name unet \\
        --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --amp --num_workers 4 --cache_dir /workspace/data/cache \\
        --seed 2 --exp_name external_unet_seed2 --epochs 100 --batch_size 1 \\
        --output_dir ./results/external_baselines
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
from models.losses import CombinedLoss
from external_baselines.models_external import build_external_model, MODEL_NAMES
from utils.metrics import evaluate_dataset, batch_dice
from utils.train_utils import (
    save_checkpoint, load_checkpoint,
    WarmupCosineScheduler, train_one_epoch, validate, log_csv,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',   type=str, required=True, choices=MODEL_NAMES)
    p.add_argument('--data_root',    type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=1)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--exp_name',     type=str,   default='external_baseline')
    p.add_argument('--output_dir',   type=str,   default='./results/external_baselines',
                   help='checkpoints -> <output_dir>/checkpoints/<exp_name>/, '
                        'test CSV -> <output_dir>/results/, '
                        'train log -> <output_dir>/logs/')
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--resume',       type=str,   default='',
                   help='path to checkpoint to resume from')
    p.add_argument('--warm_start',   type=str,   default='',
                   help='path to pre-trained checkpoint for warm start')
    p.add_argument('--num_workers',  type=int,   default=0,
                   help='DataLoader workers (0 = main process, safe on Windows; use 4 on RunPod).')
    p.add_argument('--amp',          action='store_true', default=True)
    p.add_argument('--max_samples',  type=int,   default=0,
                   help='Truncate dataset to N samples (0 = all). Use 4-8 for smoke tests.')
    p.add_argument('--cache_dir',    type=str,   default='/workspace/data/cache',
                   help='Directory for preprocessed .npz cache. Empty string = no caching.')
    p.add_argument('--seed',         type=int,   default=-1,
                   help='Random seed for reproducibility (torch / numpy / random). '
                        '-1 = no fixed seed.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed >= 0:
        _set_seed(args.seed)
        print(f"Random seed: {args.seed}  (torch / numpy / random all seeded)")
    else:
        print("Random seed: not fixed (--seed -1)")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pin = torch.cuda.is_available()
    ckpt_dir    = os.path.join(args.output_dir, 'checkpoints', args.exp_name)
    results_dir = os.path.join(args.output_dir, 'results')
    logs_dir    = os.path.join(args.output_dir, 'logs')
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir,    exist_ok=True)
    log_path = os.path.join(logs_dir, f'{args.exp_name}_train_log.csv')

    # ---- Data: identical thin_wall pipeline (split / crop / cache untouched) ----
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

    # ---- Model: the ONLY thing that differs from Stage5C ----
    model = build_external_model(args.model_name, in_channels=1, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model_name}  params={n_params:.2f}M")

    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model is on: {next(model.parameters()).device}")
    assert device.type == 'cuda', "必須使用 GPU，請檢查 --device 參數"

    # ---- Loss / Optimizer / Scheduler: identical to Stage5C ----
    criterion = CombinedLoss(num_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=max(args.epochs, 1))
    scaler    = torch.amp.GradScaler('cuda') if args.amp and torch.cuda.is_available() else None

    start_epoch   = 0
    best_val_dice = 0.0

    if args.resume:
        start_epoch, best_val_dice = load_checkpoint(model, optimizer, args.resume, device)
    elif args.warm_start:
        load_checkpoint(model, None, args.warm_start, device)

    # ---- Training loop: identical to Stage5C (train_one_epoch / validate unmodified) ----
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
        # NOTE: validate() only computes loss + Dice (utils/train_utils.py,
        # via batch_dice) -- it does NOT compute HD95/SurfaceDice per epoch
        # (those need scipy distance transforms, too slow to run every
        # epoch). This matches Stage5C's train_fft_residual.py exactly,
        # which logs the same fields; HD95/SurfaceDice are only computed
        # once at the end via evaluate_dataset on the test set below.
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

    # ---- Final test evaluation: identical call, identical CSV columns ----
    print("\n=== Loading best model for test evaluation ===")
    best_path = os.path.join(ckpt_dir, 'best.pth')
    if os.path.exists(best_path) or os.path.exists(best_path[:-4] + '_weights.pth'):
        load_checkpoint(model, None, best_path, device)
    elif args.resume:
        print(f"  No best.pth saved this run (--epochs {args.epochs}); "
              f"evaluating the --resume checkpoint as-is.")
    evaluate_dataset(
        model, test_loader, device,
        output_csv=os.path.join(results_dir, f'{args.exp_name}_test_results.csv'),
    )


if __name__ == '__main__':
    main()
