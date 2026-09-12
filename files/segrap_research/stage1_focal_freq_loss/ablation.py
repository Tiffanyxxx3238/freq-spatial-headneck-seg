"""
Stage 1 ablation: sweep lambda_ffl in {0, 0.05, 0.1, 0.2}.
Runs each config sequentially and aggregates results.

Usage:
    cd segrap_research
    python stage1_focal_freq_loss/ablation.py --data_root <path> --epochs 100
"""
import argparse
import subprocess
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ABLATION_CONFIGS = [
    {'lambda_ffl': 0.0,  'name': 'stage1_baseline'},
    {'lambda_ffl': 0.05, 'name': 'stage1_ffl_0.05'},
    {'lambda_ffl': 0.10, 'name': 'stage1_ffl_0.10'},
    {'lambda_ffl': 0.20, 'name': 'stage1_ffl_0.20'},
]


def run_ablation(data_root, epochs, device, num_workers):
    for cfg in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['name']}  (lambda_ffl={cfg['lambda_ffl']})")
        print('='*60)
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'train.py'),
            '--data_root', data_root,
            '--lambda_ffl', str(cfg['lambda_ffl']),
            '--exp_name', cfg['name'],
            '--epochs', str(epochs),
            '--device', device,
            '--num_workers', str(num_workers),
        ]
        subprocess.run(cmd, check=True)

    # Aggregate results
    _aggregate_results()


def _aggregate_results():
    import pandas as pd
    results_dir = 'results'
    rows = []
    for cfg in ABLATION_CONFIGS:
        csv_path = f'{results_dir}/{cfg["name"]}_test_results.csv'
        if not os.path.exists(csv_path):
            print(f"  [warn] {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        row = {
            'exp': cfg['name'],
            'lambda_ffl': cfg['lambda_ffl'],
            'overall_dice':  df['dice'].mean(),
            'overall_iou':   df['iou'].mean(),
            'overall_hd95':  df['hd95'].mean(),
            'overall_sdice1': df['surface_dice_1mm'].mean(),
            'overall_sdice2': df['surface_dice_2mm'].mean(),
        }
        thin = df[df['is_thin_wall']]
        row.update({
            'thinwall_dice':   thin['dice'].mean(),
            'thinwall_iou':    thin['iou'].mean(),
            'thinwall_hd95':   thin['hd95'].mean(),
            'thinwall_sdice1': thin['surface_dice_1mm'].mean(),
            'thinwall_sdice2': thin['surface_dice_2mm'].mean(),
        })
        rows.append(row)

    if not rows:
        print("No results to aggregate.")
        return

    ablation_df = pd.DataFrame(rows)
    out_path = f'{results_dir}/stage1_ablation.csv'
    ablation_df.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}")
    print(ablation_df.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='../data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--aggregate_only', action='store_true',
                   help='Skip training; just aggregate existing result CSVs')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.aggregate_only:
        _aggregate_results()
    else:
        run_ablation(args.data_root, args.epochs, args.device, args.num_workers)
