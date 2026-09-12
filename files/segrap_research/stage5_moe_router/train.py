"""
Stage 5 training: AFS-DSN with Anatomy-aware Multi-Expert Frequency Routing.

Usage (from segrap_research/):
    python stage5_moe_router/train.py \\
        --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --cache_dir /workspace/data/cache \\
        --exp_name  s5_moe_topk2 \\
        --top_k 2 --seed 1 --epochs 100 --output_dir ./results/stage5

--top_k defaults to 2 (training default — see moe_router.py for why top_k=1
gives the routing network zero gradient with only 3 experts). top_k=1 is only
meaningful as an EVAL-time stress test of an already-trained top_k=2 model:
    python stage5_moe_router/train.py --top_k 1 --resume <ckpt> --epochs 0 \\
        --exp_name s5_moe_eval_topk1 ...
(--epochs 0 skips training and goes straight to the test-set evaluation block.)

Differences from stage3_fft_branch/train.py:
  - Model: AFS_DSN_MoE (3-expert frequency router: FcaNet, FFT, Identity)
  - Added --top_k
  - Evaluation: custom evaluate_with_diagnostics() (not utils.metrics.evaluate_dataset)
    so per-sample expert_weights / selected_experts / routing_entropy are saved
    alongside the usual dice/hd95/surface_dice metrics. Duplicated rather than
    added to the shared evaluate_dataset() so Stage 1-4 stay untouched.
"""
import argparse
import ast
import gc
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.segrap_dataset import SegRapDataset
from models.afs_dsn_original import AFS_DSN_V4
from models.losses import CombinedLoss
from stage2_fcanet_plugin.model import AFS_DSN_FcaNet
from stage3_fft_branch.model import AFS_DSN_FFT
from stage5_moe_router.model import AFS_DSN_MoE
from utils.metrics import evaluate_case, batch_dice
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


def evaluate_with_diagnostics(model, dataloader, device, output_csv: str = None) -> pd.DataFrame:
    """
    Same per-case metrics as utils.metrics.evaluate_dataset, plus per-sample
    MoE routing diagnostics read from model.freq_branch.last_diagnostics
    (populated as a side effect of the forward call right above it).

    Deliberately NOT added to utils/metrics.py: that function is shared by
    Stage 1-4, which have no router_diagnostics concept at all. Duplicating
    the per-case metric logic here keeps that shared file untouched.
    """
    model.eval()
    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images        = batch['image'].to(device)
            labels        = batch['label']
            case_ids      = batch['case_id']
            oar_names     = batch['oar_name']
            batch_spacing = batch.get('spacing', None)

            out = model(images)
            logits = out['output']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            diag = model.freq_branch.last_diagnostics
            expert_weights   = diag['expert_weights'].cpu().numpy()    # (B, 3)
            selected_experts = diag['selected_experts'].cpu().numpy()  # (B, top_k)
            routing_entropy  = diag['routing_entropy'].cpu().numpy()   # (B,)

            for b in range(images.shape[0]):
                pred_b = (preds[b] > 0).astype(np.uint8)
                tgt_b  = (labels_np[b] > 0).astype(np.uint8)
                if batch_spacing is not None:
                    sample_spacing = tuple(float(batch_spacing[i][b]) for i in range(3))
                else:
                    sample_spacing = (1.0, 1.0, 1.0)

                rec = evaluate_case(pred_b, tgt_b, sample_spacing, oar_names[b])
                rec['case_id']          = case_ids[b]
                # Stored as Python-literal strings (e.g. "[0.45, 0.55, 0.0]");
                # parse back with ast.literal_eval() for downstream analysis.
                rec['expert_weights']   = expert_weights[b].tolist()
                rec['selected_experts'] = selected_experts[b].tolist()
                rec['routing_entropy']  = float(routing_entropy[b])
                records.append(rec)

    df = pd.DataFrame(records)

    overall = df[['dice', 'iou', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']].mean()
    print("\n=== Overall ===")
    print(overall.to_string())

    mean_entropy = df['routing_entropy'].mean()
    expert_pick_rate = pd.Series(
        [e for row in df['selected_experts'] for e in row]
    ).value_counts(normalize=True).sort_index()
    print(f"\n=== Routing diagnostics ===")
    print(f"Mean routing entropy: {mean_entropy:.4f}")
    print(f"Expert selection rate (0=fcanet, 1=fft, 2=identity):")
    print(expert_pick_rate.to_string())

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")

    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',    type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--epochs',       type=int,   default=100,
                   help='Set to 0 to skip training and go straight to test-set evaluation '
                        '(use with --resume and a different --top_k for eval-time sparsity tests).')
    p.add_argument('--batch_size',   type=int,   default=2)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--exp_name',     type=str,   default='s5_moe')
    p.add_argument('--output_dir',   type=str,   default='.',
                   help='root dir; checkpoints -> <output_dir>/checkpoints/, '
                        'results -> <output_dir>/results/')
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
    p.add_argument('--top_k',        type=int,   default=2, choices=[1, 2, 3],
                   help='Experts selected per sample out of 3 (fcanet, fft, identity). '
                        'Default 2 for training (top_k=1 gives the router zero gradient — '
                        'see moe_router.py). top_k=1 is only for eval-time sparsity stress tests.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.top_k == 1 and args.epochs > 0:
        print("WARNING: --top_k 1 with --epochs > 0. With 3 experts, softmax of a single "
              "selected logit is identically 1.0, so the routing network gets essentially "
              "no gradient through the mixing weight. top_k=1 is intended for evaluating an "
              "already-trained top_k=2 checkpoint (--resume + --epochs 0), not for training "
              "from scratch. Proceeding anyway since you explicitly set epochs > 0.")

    if args.seed >= 0:
        _set_seed(args.seed)
        print(f"Random seed: {args.seed}  (torch / numpy / random all seeded)")
    else:
        print("Random seed: not fixed (--seed -1)")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pin = torch.cuda.is_available()
    ckpt_dir    = os.path.join(args.output_dir, 'checkpoints', args.exp_name)
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(ckpt_dir,    exist_ok=True)
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
    model = AFS_DSN_MoE(
        in_channels=1, num_classes=2, base_features=args.base_features,
        use_freq_branch=True, use_cross_attention=True, use_router=True,
        top_k=args.top_k,
    )
    n_moe = sum(p.numel() for p in model.parameters()) / 1e6

    _fca_ref = AFS_DSN_FcaNet(in_channels=1, num_classes=2, base_features=args.base_features)
    _fft_ref = AFS_DSN_FFT  (in_channels=1, num_classes=2, base_features=args.base_features)
    n_fca = sum(p.numel() for p in _fca_ref.parameters()) / 1e6
    n_fft = sum(p.numel() for p in _fft_ref.parameters()) / 1e6
    del _fca_ref, _fft_ref

    print(f"AFS_DSN_MoE total params : {n_moe:.1f} M  (top_k={args.top_k} of 3 experts active per sample)")
    print(f"AFS_DSN_FcaNet (Stage2)  : {n_fca:.1f} M  (reference)")
    print(f"AFS_DSN_FFT    (Stage3)  : {n_fft:.1f} M  (reference)")

    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model is on: {next(model.parameters()).device}")
    assert device.type == 'cuda', "必須使用 GPU，請檢查 --device 參數"

    # ---- Loss / Optimizer / Scheduler ----
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
    best_path = os.path.join(ckpt_dir, 'best.pth')
    if os.path.exists(best_path) or os.path.exists(best_path[:-4] + '_weights.pth'):
        load_checkpoint(model, None, best_path, device)
    elif args.resume:
        print(f"  No best.pth saved this run (--epochs {args.epochs}); "
              f"evaluating the --resume checkpoint as-is at top_k={args.top_k}.")
    evaluate_with_diagnostics(
        model, test_loader, device,
        output_csv=os.path.join(results_dir, f'{args.exp_name}_test_results.csv'),
    )


if __name__ == '__main__':
    main()
