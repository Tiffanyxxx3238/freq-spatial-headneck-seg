"""
External Baseline Study — forward/backward smoke test.

Usage:
    python external_baselines/smoke_test.py --model_name unet
    python external_baselines/smoke_test.py --model_name segresnet
    python external_baselines/smoke_test.py --model_name mednext_s

Checks (in order): import, forward shape, NaN/Inf, param counts, a backward
pass with peak CUDA memory, and a final BS=1-feasibility verdict. Does NOT
train anything — this is purely a pre-flight check before committing to a
100-epoch run.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from external_baselines.models_external import build_external_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, required=True,
                   choices=['unet', 'segresnet', 'mednext_s'])
    p.add_argument('--input_size', type=int, default=128,
                   help='Cubic crop size; matches the pipeline default target_size.')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[MODEL] {args.model_name}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Import / build ----
    try:
        model = build_external_model(args.model_name, in_channels=1, num_classes=2)
        print("[IMPORT] OK")
    except ImportError as e:
        print("[IMPORT] FAILED")
        print(f"[IMPORT ERROR] {e}")
        print("[FORWARD] SKIPPED")
        print("[BS=1 FEASIBLE] No (import failed)")
        return
    except Exception as e:
        print("[IMPORT] FAILED (unexpected error)")
        print(f"[IMPORT ERROR] {type(e).__name__}: {e}")
        print("[FORWARD] SKIPPED")
        print("[BS=1 FEASIBLE] No (import failed)")
        return

    model = model.to(device)
    x = torch.randn(1, 1, args.input_size, args.input_size, args.input_size, device=device)
    print(f"[INPUT SHAPE] {tuple(x.shape)}")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Forward ----
    try:
        out = model(x)
        logits = out['output'] if isinstance(out, dict) else out
        print("[FORWARD] OK")
    except Exception as e:
        print("[FORWARD] FAILED")
        print(f"[FORWARD ERROR] {type(e).__name__}: {e}")
        print("[BS=1 FEASIBLE] No (forward failed)")
        return

    print(f"[OUTPUT SHAPE] {tuple(logits.shape)}")
    expected_shape = (1, 2, args.input_size, args.input_size, args.input_size)
    if tuple(logits.shape) != expected_shape:
        print(f"[SHAPE WARNING] expected {expected_shape}, got {tuple(logits.shape)}")

    is_finite = torch.isfinite(logits).all().item()
    print(f"[NAN/INF] {'No' if is_finite else 'Yes -- PROBLEM'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TOTAL PARAMS] {total_params}")
    print(f"[TRAINABLE PARAMS] {trainable_params}")
    print(f"[PARAMS M] {total_params / 1e6:.2f}")

    # ---- Backward + peak memory ----
    bs1_feasible = is_finite
    if device.type == 'cuda' and is_finite:
        try:
            y = torch.randint(0, 2, (1, args.input_size, args.input_size, args.input_size), device=device)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"[PEAK CUDA MEMORY] {peak_mem_mb:.1f} MB")
        except RuntimeError as e:
            print(f"[PEAK CUDA MEMORY] backward FAILED: {e}")
            bs1_feasible = False
    elif device.type != 'cuda':
        print("[PEAK CUDA MEMORY] N/A (no CUDA device)")
    else:
        print("[PEAK CUDA MEMORY] SKIPPED (NaN/Inf in forward)")
        bs1_feasible = False

    print(f"[BS=1 FEASIBLE] {'Yes' if bs1_feasible else 'No'}")


if __name__ == '__main__':
    main()
