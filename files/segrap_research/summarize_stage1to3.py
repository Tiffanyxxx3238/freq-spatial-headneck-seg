"""
Stage 1-3 summary table generator.

Reads all available test-result CSVs across Stage 1 (DWT baseline + FFL variants),
Stage 2 (FcaNet), and Stage 3 (FFT branch), and builds:

  1. Overall summary table — one row per method:
       method, params(M), Dice, HD95, SurfaceDice@1mm, SurfaceDice@2mm
  2. Per-OAR table — one row per OAR, Dice + HD95 for every method side by side
     (lets you scan which method wins on which OAR).

Usage (from segrap_research/):
    python summarize_stage1to3.py

Outputs:
    ./results/stage1to3_summary.csv
    ./results/stage1to3_per_oar.csv
"""
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# OAR ordering (matches data/segrap_dataset.py THIN_WALL_OARS;        #
# duplicated here so this script has no torch/nibabel dependency)     #
# ------------------------------------------------------------------ #
THIN_WALL_OARS = [
    'Cochlea_L', 'Cochlea_R',
    'VestibulSemi_L', 'VestibulSemi_R',
    'IAC_L', 'IAC_R',
    'TympanicCavity_L', 'TympanicCavity_R',
    'MiddleEar_L', 'MiddleEar_R',
]

# ------------------------------------------------------------------ #
# Method registry: (display label, csv path, param count in M)       #
# ------------------------------------------------------------------ #
METHODS = [
    ('Stage1 DWT-Full (baseline)', './results/stage1/results/s1_baseline_full_test_results.csv', 414.6),
    ('Stage1 DWT-Lite',            './results/stage1/results/s1_baseline_lite_test_results.csv', 27.0),
    ('Stage1 FFL lambda=0.05',     './results/stage1/results/s1_ffl005_test_results.csv',         414.6),
    ('Stage1 FFL lambda=0.10',     './results/stage1/results/s1_ffl010_test_results.csv',         414.6),
    ('Stage1 FFL lambda=0.20',     './results/stage1/results/s1_ffl020_test_results.csv',         414.6),
    ('Stage2 FcaNet (seed2)',      './results/stage2/results/s2_fcanet_seed2_test_results.csv',   24.0),
    ('Stage3 FFT (seed1)',         './results/stage3/results/s3_fft_branch_test_results.csv',     29.4),
    ('Stage3 FFT (seed2)',         './results/stage3/results/s3_fft_seed2_test_results.csv',      29.4),
]

METRIC_COLS = ['dice', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']
METRIC_LABELS = {
    'dice':             'Dice',
    'hd95':             'HD95(mm)',
    'surface_dice_1mm': 'SDice@1mm',
    'surface_dice_2mm': 'SDice@2mm',
}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _safe_mean(s: pd.Series) -> float:
    """Mean over finite values only (hd95 may be inf for empty-mask cases)."""
    arr = np.asarray(s, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if len(finite) else np.nan


def load_methods() -> list:
    """Load all available CSVs; report (and skip) any missing files."""
    loaded = []
    for label, path, params in METHODS:
        if not os.path.exists(path):
            print(f'  [MISSING] {label}: {path}')
            continue
        df = pd.read_csv(path)
        loaded.append((label, path, params, df))
        print(f'  [OK]      {label}: {len(df)} rows  ({os.path.basename(path)})')
    return loaded


def build_overall_table(loaded: list) -> pd.DataFrame:
    rows = []
    for label, path, params, df in loaded:
        row = {'method': label, 'params_M': params, 'n_rows': len(df)}
        for col in METRIC_COLS:
            row[col] = _safe_mean(df[col]) if col in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_oar_table(loaded: list) -> pd.DataFrame:
    """Wide table: one row per OAR, '<method> | Dice' / '<method> | HD95' columns."""
    per_method_oar = {}
    all_oars = set()
    for label, _, _, df in loaded:
        grp = df.groupby('oar_name').agg(
            dice=('dice', _safe_mean),
            hd95=('hd95', _safe_mean),
        )
        per_method_oar[label] = grp
        all_oars.update(grp.index.tolist())

    ordered_oars = [o for o in THIN_WALL_OARS if o in all_oars] + \
                   sorted(o for o in all_oars if o not in THIN_WALL_OARS)

    rows = []
    for oar in ordered_oars:
        row = {'oar_name': oar}
        for label, _, _, _ in loaded:
            grp = per_method_oar[label]
            if oar in grp.index:
                row[f'{label} | Dice'] = grp.loc[oar, 'dice']
                row[f'{label} | HD95'] = grp.loc[oar, 'hd95']
            else:
                row[f'{label} | Dice'] = np.nan
                row[f'{label} | HD95'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Printing                                                             #
# ------------------------------------------------------------------ #

def _rule(c: str = '=', width: int = 96) -> None:
    print(c * width)


def _fnum(v: float, fmt: str = '.4f') -> str:
    return format(v, fmt) if np.isfinite(v) else 'nan'


def print_overall_table(df: pd.DataFrame) -> None:
    _rule()
    print('  STAGE 1-3 OVERALL COMPARISON')
    _rule()
    print(f'  {"Method":<28}  {"Params(M)":>9}  {"N":>5}  '
          f'{"Dice":>8}  {"HD95(mm)":>9}  {"SDice@1mm":>10}  {"SDice@2mm":>10}')
    print(f'  {"-"*28}  {"-"*9}  {"-"*5}  {"-"*8}  {"-"*9}  {"-"*10}  {"-"*10}')
    for _, row in df.iterrows():
        print(f'  {row["method"]:<28}  {row["params_M"]:>9.1f}  {int(row["n_rows"]):>5}  '
              f'{_fnum(row["dice"]):>8}  {_fnum(row["hd95"]):>9}  '
              f'{_fnum(row["surface_dice_1mm"]):>10}  {_fnum(row["surface_dice_2mm"]):>10}')
    _rule()


def print_per_oar_table(df: pd.DataFrame, loaded: list) -> None:
    labels = [label for label, _, _, _ in loaded]
    short = {lbl: (lbl[:14] + '..') if len(lbl) > 16 else lbl for lbl in labels}

    for metric_name, metric_key in [('Dice (higher better)', 'Dice'), ('HD95 mm (lower better)', 'HD95')]:
        _rule('-')
        print(f'  PER-OAR COMPARISON — {metric_name}')
        _rule('-')
        header = f'  {"OAR":<20}' + ''.join(f'  {short[lbl]:>16}' for lbl in labels)
        print(header)
        print(f'  {"-"*20}' + ''.join(f'  {"-"*16}' for _ in labels))
        for _, row in df.iterrows():
            line = f'  {row["oar_name"]:<20}'
            for lbl in labels:
                v = row[f'{lbl} | {metric_key}']
                line += f'  {_fnum(v):>16}'
            print(line)
        print()


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    print('\nLoading CSVs...')
    loaded = load_methods()
    if not loaded:
        sys.exit('\nERROR: none of the expected result CSVs were found.')

    overall_df = build_overall_table(loaded)
    per_oar_df = build_per_oar_table(loaded)

    print()
    print_overall_table(overall_df)
    print()
    print_per_oar_table(per_oar_df, loaded)

    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    overall_path = os.path.join(out_dir, 'stage1to3_summary.csv')
    per_oar_path = os.path.join(out_dir, 'stage1to3_per_oar.csv')
    overall_df.to_csv(overall_path, index=False, float_format='%.6f')
    per_oar_df.to_csv(per_oar_path, index=False, float_format='%.6f')
    print(f'Saved overall summary -> {overall_path}')
    print(f'Saved per-OAR table   -> {per_oar_path}')


if __name__ == '__main__':
    main()
