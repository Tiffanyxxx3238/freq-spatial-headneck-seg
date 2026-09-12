"""
External Baseline Study — parameter counting.

Usage:
    python external_baselines/count_params.py --model_name unet
    python external_baselines/count_params.py --all --output ./results/external_baselines/external_model_params.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from external_baselines.models_external import build_external_model, MODEL_NAMES


def count_one(model_name: str) -> dict:
    try:
        model = build_external_model(model_name, in_channels=1, num_classes=2)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'model_name':       model_name,
            'total_params':     total,
            'trainable_params': trainable,
            'total_params_M':   round(total / 1e6, 4),
            'status':           'ok',
        }
    except Exception as e:
        return {
            'model_name':       model_name,
            'total_params':     '',
            'trainable_params': '',
            'total_params_M':   '',
            'status':           f'FAILED: {type(e).__name__}: {e}',
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, choices=MODEL_NAMES)
    p.add_argument('--all',        action='store_true')
    p.add_argument('--output',     type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        rows = [count_one(m) for m in MODEL_NAMES]
    elif args.model_name:
        rows = [count_one(args.model_name)]
    else:
        raise SystemExit('Specify --model_name or --all')

    for r in rows:
        print(r['model_name'])
        print('  total_params     :', r['total_params'])
        print('  trainable_params :', r['trainable_params'])
        print('  total_params_M   :', r['total_params_M'])
        print('  status           :', r['status'])
        print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'model_name', 'total_params', 'trainable_params', 'total_params_M', 'status',
            ])
            w.writeheader()
            w.writerows(rows)
        print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
