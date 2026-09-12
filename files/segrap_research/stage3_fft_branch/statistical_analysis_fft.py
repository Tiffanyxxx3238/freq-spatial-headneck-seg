"""
Statistical comparison: Stage-1 DWT baseline vs Stage-3 FFT branch.

Usage (from segrap_research/):
    python stage3_fft_branch/statistical_analysis_fft.py
    python stage3_fft_branch/statistical_analysis_fft.py \\
        --baseline ./results/stage1/results/s1_baseline_full_test_results.csv \\
        --fft      ./results/stage3/results/s3_fft_branch_test_results.csv \\
        --out_dir  ./results/stage3

Analysis levels:
  Level 1 — per-OAR paired tests (18 cases × 10 OARs × 4 metrics = 40 tests).
  Level 2 — case-level thin-wall mean: average 10 OARs per case, test 18 case means.
  FDR     — Benjamini-Hochberg per metric (family = 10 OARs; 4 families of 10).
             t-test and Wilcoxon corrected separately within each family.
  Direction — count OARs where FFT improves over DWT for each metric.
  Interpretation — text verdict printed at the end.

Metric direction: dice / surface_dice ↑ better;  hd95 ↓ better.
diff = FFT − baseline  (positive = FFT better for ↑-metrics; negative = FFT better for hd95).

Outputs:
  <out_dir>/fft_vs_baseline_statistical_summary.csv  — Level 1 + Level 2
  <out_dir>/fft_vs_baseline_fdr_summary.csv          — Level 1 with FDR-corrected p-values
"""
import argparse
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests as _sm_multipletests
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

THIN_WALL_OARS = [
    'Cochlea_L', 'Cochlea_R',
    'VestibulSemi_L', 'VestibulSemi_R',
    'IAC_L', 'IAC_R',
    'TympanicCavity_L', 'TympanicCavity_R',
    'MiddleEar_L', 'MiddleEar_R',
]

PRIORITY_OARS = ['MiddleEar_L', 'MiddleEar_R', 'IAC_R', 'VestibulSemi_L']
OTHER_OARS    = [o for o in THIN_WALL_OARS if o not in PRIORITY_OARS]
ORDERED_OARS  = PRIORITY_OARS + OTHER_OARS

METRIC_MAP_CANDIDATES = {
    'dice':             ['dice'],
    'hd95':             ['hd95'],
    'surface_dice_1mm': ['surface_dice_1mm', 'surface_dice_1', 'sdice_1mm'],
    'surface_dice_2mm': ['surface_dice_2mm', 'surface_dice_2', 'sdice_2mm'],
}
METRIC_LABELS = {
    'dice':             'Dice           ',
    'hd95':             'HD95 (mm)      ',
    'surface_dice_1mm': 'SurfDice @1 mm ',
    'surface_dice_2mm': 'SurfDice @2 mm ',
}
HIGHER_BETTER = {'dice', 'surface_dice_1mm', 'surface_dice_2mm'}  # hd95: lower is better


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ' '
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '* '
    return '  '


def _detect_metrics(df: pd.DataFrame) -> dict:
    """Map canonical metric names to actual column names found in df."""
    mapping = {}
    for canon, candidates in METRIC_MAP_CANDIDATES.items():
        for c in candidates:
            if c in df.columns:
                mapping[canon] = c
                break
    return mapping


def _paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Paired t-test + Wilcoxon on finite pairs only.
    diff = b − a  (FFT − baseline).
    """
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(mask.sum())
    if n < 3:
        return dict(n=n, mean_a=np.nan, mean_b=np.nan, diff=np.nan, t_p=np.nan, w_p=np.nan)

    diff = b - a
    try:
        _, t_p = stats.ttest_rel(b, a)
    except Exception:
        t_p = np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_p = np.nan

    return dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()),
                diff=float(diff.mean()), t_p=float(t_p), w_p=float(w_p))


# ------------------------------------------------------------------ #
# Analysis routines                                                    #
# ------------------------------------------------------------------ #

def level1_oar(df_base: pd.DataFrame, df_fft: pd.DataFrame,
               col_map: dict) -> pd.DataFrame:
    """Per-OAR paired tests. Returns one row per (oar, metric)."""
    rows = []
    for oar in ORDERED_OARS:
        b_oar = df_base[df_base['oar_name'] == oar].sort_values('case_id')
        f_oar = df_fft [df_fft ['oar_name'] == oar].sort_values('case_id')
        merged = b_oar.set_index('case_id')[list(col_map.values())].join(
            f_oar.set_index('case_id')[list(col_map.values())],
            lsuffix='_b', rsuffix='_f', how='inner',
        )
        for canon, col in col_map.items():
            a   = merged[f'{col}_b'].values.astype(float)
            b   = merged[f'{col}_f'].values.astype(float)
            res = _paired_tests(a, b)
            rows.append({
                'level':     1,
                'oar':       oar,
                'priority':  oar in PRIORITY_OARS,
                'metric':    canon,
                'N':         res['n'],
                'mean_base': res['mean_a'],
                'mean_fft':  res['mean_b'],
                'diff':      res['diff'],
                't_pval':    res['t_p'],
                'w_pval':    res['w_p'],
                't_sig':     _stars(res['t_p']),
                'w_sig':     _stars(res['w_p']),
            })
    return pd.DataFrame(rows)


def level2_case_mean(df_base: pd.DataFrame, df_fft: pd.DataFrame,
                     col_map: dict) -> pd.DataFrame:
    """
    Case-level mean: average 10 OAR metric values per case first,
    then paired tests on 18 case-level means.
    Avoids pseudoreplication from treating each OAR as an independent observation.
    """
    rows = []
    for canon, col in col_map.items():
        case_b = df_base.groupby('case_id')[col].apply(
            lambda x: np.nanmean(x[np.isfinite(x)])
        )
        case_f = df_fft.groupby('case_id')[col].apply(
            lambda x: np.nanmean(x[np.isfinite(x)])
        )
        common = case_b.index.intersection(case_f.index).sort_values()
        a   = case_b.loc[common].values.astype(float)
        b   = case_f.loc[common].values.astype(float)
        res = _paired_tests(a, b)
        rows.append({
            'level':     2,
            'metric':    canon,
            'N_cases':   res['n'],
            'mean_base': res['mean_a'],
            'mean_fft':  res['mean_b'],
            'diff':      res['diff'],
            't_pval':    res['t_p'],
            'w_pval':    res['w_p'],
            't_sig':     _stars(res['t_p']),
            'w_sig':     _stars(res['w_p']),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# FDR correction: per-metric families                                  #
# ------------------------------------------------------------------ #

def apply_fdr_per_metric(df_l1: pd.DataFrame) -> pd.DataFrame:
    """
    Benjamini-Hochberg correction applied per metric (4 independent families).
    Family = 10 OARs within each metric.
    t-test and Wilcoxon corrected separately within each family.
    Returns df_l1 with added t_pval_fdr and w_pval_fdr columns.
    """
    if not _STATSMODELS:
        raise RuntimeError('statsmodels not installed: pip install statsmodels')

    df = df_l1.copy()
    df['t_pval_fdr'] = np.nan
    df['w_pval_fdr'] = np.nan

    for metric in df['metric'].unique():
        idx = df.index[df['metric'] == metric]
        for raw_col, fdr_col in [('t_pval', 't_pval_fdr'), ('w_pval', 'w_pval_fdr')]:
            pvals  = df.loc[idx, raw_col].values.astype(float)
            finite = np.isfinite(pvals)
            corrected = np.full(len(pvals), np.nan)
            if finite.sum() >= 2:
                _, adj, _, _ = _sm_multipletests(
                    pvals[finite], alpha=0.05, method='fdr_bh'
                )
                corrected[finite] = adj
            df.loc[idx, fdr_col] = corrected

    return df


# ------------------------------------------------------------------ #
# Direction consistency                                                 #
# ------------------------------------------------------------------ #

def direction_summary(df_l1: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric, count OARs where FFT improves over DWT.
    Improvement: diff > 0 for ↑-better metrics; diff < 0 for hd95.
    """
    rows = []
    for metric in ['dice', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']:
        sub = df_l1[df_l1['metric'] == metric]
        if sub.empty:
            continue
        diffs = sub['diff'].values.astype(float)
        finite_diffs = diffs[np.isfinite(diffs)]
        n_total = len(finite_diffs)
        if metric in HIGHER_BETTER:
            n_better  = int((finite_diffs > 0).sum())
            direction = '↑'
        else:
            n_better  = int((finite_diffs < 0).sum())
            direction = '↓'
        rows.append({
            'metric':    metric,
            'direction': direction,
            'n_better':  n_better,
            'n_total':   n_total,
            'pct':       100.0 * n_better / n_total if n_total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Print helpers                                                         #
# ------------------------------------------------------------------ #

W = 88

def _rule(char: str = '=') -> None:
    print(char * W)

def _fp(v: float, width: int = 7) -> str:
    """Format p-value or nan."""
    return f'{v:.4f}' if np.isfinite(v) else '   nan'


def print_level1_table(df: pd.DataFrame, metric: str) -> None:
    sub = df[df['metric'] == metric].copy()
    if sub.empty:
        return
    label    = METRIC_LABELS.get(metric, metric)
    dir_hint = '(↑ better)' if metric in HIGHER_BETTER else '(↓ better)'
    _rule('-')
    print(f'  Level 1 | DWT-baseline vs FFT branch | {label} {dir_hint}')
    _rule('-')
    print(f'  {"★":<3}  {"OAR":<22}  {"N":>2}  {"Baseline":>9}  {"FFT":>9}  '
          f'{"Diff":>9}  {"t-p":>7}     {"Wilcox-p":>9}   ')
    print(f'  {"-"*3}  {"-"*22}  {"--":>2}  {"-"*9}  {"-"*9}  '
          f'{"-"*9}  {"-"*7}  {"--"}  {"-"*9}  {"--"}')
    for _, row in sub.iterrows():
        star  = '★' if row['priority'] else ' '
        diff_s = f'{row["diff"]:+.4f}'   if np.isfinite(row['diff'])      else '      nan'
        base_s = f'{row["mean_base"]:.4f}' if np.isfinite(row['mean_base']) else '      nan'
        fft_s  = f'{row["mean_fft"]:.4f}'  if np.isfinite(row['mean_fft'])  else '      nan'
        tp_s   = _fp(row['t_pval'])
        wp_s   = _fp(row['w_pval'])
        print(f'  {star:<3}  {row["oar"]:<22}  {int(row["N"]):>2}  '
              f'{base_s:>9}  {fft_s:>9}  {diff_s:>9}  '
              f'{tp_s:>7}  {row["t_sig"]}  {wp_s:>9}  {row["w_sig"]}')
    n_t = int((sub['t_pval'] < 0.05).sum())
    n_w = int((sub['w_pval'] < 0.05).sum())
    print(f'  Significant (raw p<0.05): t-test {n_t}/{len(sub)}  '
          f'Wilcoxon {n_w}/{len(sub)}')


def print_level2_table(df_l2: pd.DataFrame) -> None:
    _rule('-')
    print('  Level 2 | DWT-baseline vs FFT branch | Case-level thin-wall means')
    _rule('-')
    print(f'  {"Metric":<18}  {"N":>2}  {"Baseline":>9}  {"FFT":>9}  '
          f'{"Diff":>9}  {"t-p":>7}     {"Wilcox-p":>9}   ')
    print(f'  {"-"*18}  {"--":>2}  {"-"*9}  {"-"*9}  '
          f'{"-"*9}  {"-"*7}  {"--"}  {"-"*9}  {"--"}')
    for _, row in df_l2.iterrows():
        label  = METRIC_LABELS.get(row['metric'], row['metric'])
        diff_s = f'{row["diff"]:+.4f}'   if np.isfinite(row['diff'])      else '      nan'
        base_s = f'{row["mean_base"]:.4f}' if np.isfinite(row['mean_base']) else '      nan'
        fft_s  = f'{row["mean_fft"]:.4f}'  if np.isfinite(row['mean_fft'])  else '      nan'
        tp_s   = _fp(row['t_pval'])
        wp_s   = _fp(row['w_pval'])
        print(f'  {label:<18}  {int(row["N_cases"]):>2}  '
              f'{base_s:>9}  {fft_s:>9}  {diff_s:>9}  '
              f'{tp_s:>7}  {row["t_sig"]}  {wp_s:>9}  {row["w_sig"]}')


def print_direction_table(df_dir: pd.DataFrame) -> None:
    _rule('-')
    print('  Direction Consistency | FFT vs DWT-baseline  (fraction of OARs improved)')
    _rule('-')
    print(f'  {"Metric":<22}  {"Better":>6}  {"Improved / Total":>16}  {"Rate":>6}')
    print(f'  {"-"*22}  {"-"*6}  {"-"*16}  {"-"*6}')
    for _, row in df_dir.iterrows():
        label   = METRIC_LABELS.get(row['metric'], row['metric']).strip()
        pct_s   = f'{row["pct"]:.0f}%' if np.isfinite(row['pct']) else ' nan%'
        count_s = f'{int(row["n_better"])}/{int(row["n_total"])}'
        print(f'  {label:<22}  {row["direction"]:>6}  {count_s:>16}  {pct_s:>6}')


def print_fdr_table(df_fdr: pd.DataFrame, out_dir: str) -> None:
    """Print per-metric FDR results; save fft_vs_baseline_fdr_summary.csv."""
    print()
    _rule('#')
    print('  FDR CORRECTION (Benjamini-Hochberg, α=0.05)')
    print('  Family definition: 10 OARs per metric  →  4 families of 10 tests each')
    print('  t-test and Wilcoxon corrected separately within each family')
    _rule('#')

    for metric in ['dice', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']:
        sub = df_fdr[df_fdr['metric'] == metric].copy()
        if sub.empty:
            continue
        sig_mask  = (sub['t_pval'] < 0.05) | (sub['w_pval'] < 0.05)
        fdr_mask  = sig_mask & (
            (sub['t_pval_fdr'] < 0.05) | (sub['w_pval_fdr'] < 0.05)
        )
        n_raw = int(sig_mask.sum())
        n_fdr = int(fdr_mask.sum())
        label = METRIC_LABELS.get(metric, metric)
        _rule('-')
        print(f'  {label}  |  family size: {len(sub)}  |  '
              f'raw sig: {n_raw}/{len(sub)}  |  FDR sig: {n_fdr}/{len(sub)}')
        _rule('-')
        if n_raw == 0:
            print('  (no nominally significant results in this family)')
            continue
        df_sig = sub[sig_mask].sort_values('t_pval')
        print(f'  {"★":<2}  {"OAR":<22}  '
              f'{"t-raw":>7}  {"t-FDR":>7}  {"t✓":>3}  '
              f'{"w-raw":>7}  {"w-FDR":>7}  {"w✓":>3}')
        print(f'  {"--":<2}  {"-"*22}  '
              f'{"-"*7}  {"-"*7}  {"---":>3}  '
              f'{"-"*7}  {"-"*7}  {"---":>3}')
        for _, row in df_sig.iterrows():
            t_ok = '✓' if (np.isfinite(row['t_pval_fdr']) and row['t_pval_fdr'] < 0.05) else '✗'
            w_ok = '✓' if (np.isfinite(row['w_pval_fdr']) and row['w_pval_fdr'] < 0.05) else '✗'
            pri  = '★' if row['priority'] else ' '
            print(f'  {pri:<2}  {row["oar"]:<22}  '
                  f'{_fp(row["t_pval"]):>7}  {_fp(row["t_pval_fdr"]):>7}  {t_ok:>3}  '
                  f'{_fp(row["w_pval"]):>7}  {_fp(row["w_pval_fdr"]):>7}  {w_ok:>3}')

    # Grand totals across all 40 tests
    all_raw = (df_fdr['t_pval'] < 0.05) | (df_fdr['w_pval'] < 0.05)
    all_fdr = all_raw & (
        (df_fdr['t_pval_fdr'] < 0.05) | (df_fdr['w_pval_fdr'] < 0.05)
    )
    print()
    _rule()
    print(f'  全域原始顯著 (raw p<0.05, either test)：{int(all_raw.sum())}/{len(df_fdr)}  '
          f'(across 4 families × 10 OARs)')
    print(f'  全域 FDR 後仍顯著 (adj p<0.05, either test)：{int(all_fdr.sum())}/{len(df_fdr)}')
    _rule()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    fdr_path = os.path.join(out_dir, 'fft_vs_baseline_fdr_summary.csv')
    save_cols = ['oar', 'priority', 'metric', 'N',
                 'mean_base', 'mean_fft', 'diff',
                 't_pval', 't_pval_fdr', 'w_pval', 'w_pval_fdr']
    df_fdr[save_cols].to_csv(fdr_path, index=False, float_format='%.6f')
    print(f'  Saved → {fdr_path}')
    _rule()


# ------------------------------------------------------------------ #
# Interpretation                                                        #
# ------------------------------------------------------------------ #

def print_interpretation(df_l1: pd.DataFrame, df_l2: pd.DataFrame,
                         df_fdr: Optional[pd.DataFrame],
                         df_dir: pd.DataFrame) -> None:
    """Final text verdict from all evidence streams."""
    print()
    _rule('#')
    print('  INTERPRETATION SUMMARY  —  FFT branch vs DWT-baseline')
    _rule('#')

    # Case-level
    print('\n  [Level 2 — Case-level paired tests on thin-wall mean]')
    l2_any_sig = False
    for _, row in df_l2.iterrows():
        label  = METRIC_LABELS.get(row['metric'], row['metric']).strip()
        t_sig  = np.isfinite(row['t_pval']) and row['t_pval'] < 0.05
        w_sig  = np.isfinite(row['w_pval']) and row['w_pval'] < 0.05
        sig    = t_sig or w_sig
        if sig:
            l2_any_sig = True
        diff_s = f'{row["diff"]:+.4f}' if np.isfinite(row['diff']) else 'nan'
        tag    = 'SIGNIFICANT ✓' if sig else 'not significant'
        print(f'  • {label:<22} diff={diff_s}  '
              f't-p={_fp(row["t_pval"])}  w-p={_fp(row["w_pval"])}  →  {tag}')

    # Per-OAR raw significance
    print('\n  [Level 1 — Per-OAR raw p<0.05 (either test)]')
    raw_sig_total = 0
    for metric in ['dice', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']:
        sub      = df_l1[df_l1['metric'] == metric]
        sig_oars = sub[(sub['t_pval'] < 0.05) | (sub['w_pval'] < 0.05)]['oar'].tolist()
        raw_sig_total += len(sig_oars)
        label = METRIC_LABELS.get(metric, metric).strip()
        oar_s = ', '.join(sig_oars) if sig_oars else '(none)'
        print(f'  • {label}: {oar_s}')

    # FDR-corrected OARs
    if df_fdr is not None:
        print('\n  [FDR — per-metric BH correction, α=0.05]')
        fdr_sig_total = 0
        for metric in ['dice', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']:
            sub      = df_fdr[df_fdr['metric'] == metric]
            fdr_oars = sub[
                (sub['t_pval_fdr'] < 0.05) | (sub['w_pval_fdr'] < 0.05)
            ]['oar'].tolist()
            fdr_sig_total += len(fdr_oars)
            label = METRIC_LABELS.get(metric, metric).strip()
            oar_s = ', '.join(fdr_oars) if fdr_oars else '(none survive FDR)'
            print(f'  • {label}: {oar_s}')
    else:
        fdr_sig_total = None

    # Direction consistency
    print('\n  [Direction Consistency — fraction of OARs where FFT improves]')
    consistent_count = 0
    for _, row in df_dir.iterrows():
        label     = METRIC_LABELS.get(row['metric'], row['metric']).strip()
        pct       = row['pct']
        consistent = np.isfinite(pct) and pct >= 60
        if consistent:
            consistent_count += 1
        tag = '✓ consistent (≥60%)' if consistent else '✗ mixed (<60%)'
        pct_s = f'{pct:.0f}%' if np.isfinite(pct) else 'nan'
        print(f'  • {label:<22}  {int(row["n_better"])}/{int(row["n_total"])} OARs  '
              f'({pct_s})  {tag}')
    n_metrics = len(df_dir)

    # Verdict
    print()
    _rule('-')
    print('  VERDICT')
    _rule('-')

    if l2_any_sig and consistent_count >= 3:
        verdict = (
            f'FFT branch shows case-level statistical significance in at least one metric '
            f'AND direction-consistent improvement in {consistent_count}/{n_metrics} metrics.\n'
            f'  → Evidence STRONGLY SUPPORTS explicit FFT branch over DWT-baseline.'
        )
    elif l2_any_sig and consistent_count >= 2:
        verdict = (
            f'FFT branch shows case-level statistical significance with direction-consistent '
            f'improvement in {consistent_count}/{n_metrics} metrics.\n'
            f'  → Evidence SUPPORTS FFT branch; results are promising but not uniform across metrics.'
        )
    elif consistent_count >= 3:
        verdict = (
            f'FFT branch shows direction-consistent improvement in {consistent_count}/{n_metrics} '
            f'metrics, but case-level significance is not reached (N=18 may be underpowered).\n'
            f'  → Trend FAVORS FFT; statistical confirmation requires more test cases.'
        )
    elif raw_sig_total >= 5 and consistent_count >= 2:
        verdict = (
            f'{raw_sig_total} OAR-metric pairs are nominally significant, '
            f'with consistent direction in {consistent_count}/{n_metrics} metrics.\n'
            f'  → Weak evidence for FFT branch; interpret with caution given multiple testing.'
        )
    else:
        verdict = (
            f'FFT branch does not show consistent improvement over DWT-baseline '
            f'(consistent in {consistent_count}/{n_metrics} metrics; '
            f'{raw_sig_total} nominally significant OAR-metric pairs).\n'
            f'  → Current evidence does NOT support replacing DWT with FFT branch in this form.'
        )

    print(f'\n  {verdict}')
    print()
    _rule('#')


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description='Compare DWT-baseline vs FFT-branch test results.'
    )
    ap.add_argument(
        '--baseline', default='./results/stage1/results/s1_baseline_full_test_results.csv',
        help='Path to Stage-1 baseline test-results CSV',
    )
    ap.add_argument(
        '--fft', default='./results/stage3/results/s3_fft_branch_test_results.csv',
        help='Path to Stage-3 FFT-branch test-results CSV',
    )
    ap.add_argument(
        '--out_dir', default='./results/stage3',
        help='Directory for output CSVs',
    )
    args = ap.parse_args()

    # ---- Load ----
    print('\nLoading CSVs...')
    for label, path in [('baseline', args.baseline), ('fft', args.fft)]:
        if not os.path.exists(path):
            sys.exit(f'  [MISSING] {label}: {path}')
        print(f'  [OK] {label}: {os.path.basename(path)}')

    df_base = pd.read_csv(args.baseline)
    df_fft  = pd.read_csv(args.fft)
    print(f'  Baseline: {len(df_base)} rows   FFT: {len(df_fft)} rows')
    print(f'  Baseline cols: {list(df_base.columns)}')

    col_map = _detect_metrics(df_base)
    if not col_map:
        sys.exit('ERROR: could not detect metric columns in baseline CSV.')
    print(f'  Detected metrics: {col_map}')

    # Verify OAR column
    oar_col = 'oar_name'
    if oar_col not in df_base.columns:
        sys.exit(f'ERROR: expected column "{oar_col}" not found in baseline CSV. '
                 f'Available: {list(df_base.columns)}')

    # ---- Level 1: per-OAR ----
    print('\nRunning Level 1 per-OAR analysis...')
    df_l1 = level1_oar(df_base, df_fft, col_map)
    print(f'  {len(df_l1)} rows (10 OARs × {len(col_map)} metrics)')

    # ---- Level 2: case-level mean ----
    print('Running Level 2 case-level analysis...')
    df_l2 = level2_case_mean(df_base, df_fft, col_map)

    # ---- Direction summary ----
    df_dir = direction_summary(df_l1)

    # ---- Print tables ----
    print()
    _rule('#')
    print('  STATISTICAL ANALYSIS: DWT-baseline vs Stage-3 FFT branch')
    _rule('#')
    print(f'  Priority OARs (★): {", ".join(PRIORITY_OARS)}')
    print('  sig codes: * p<0.05   ** p<0.01   (no correction at this stage)')
    print('  diff = FFT − baseline')

    print()
    for metric in col_map:
        print_level1_table(df_l1, metric)
        print()

    print_level2_table(df_l2)
    print()
    print_direction_table(df_dir)

    # ---- FDR ----
    df_l1_fdr = None
    if _STATSMODELS:
        df_l1_fdr = apply_fdr_per_metric(df_l1)
        print_fdr_table(df_l1_fdr, args.out_dir)
    else:
        print('\n  [SKIP] FDR correction requires statsmodels: pip install statsmodels')

    # ---- Save summary CSV ----
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, 'fft_vs_baseline_statistical_summary.csv')

    # Level 1
    save_l1 = (df_l1
               .drop(columns=['t_sig', 'w_sig'])
               .rename(columns={'oar': 'oar_or_agg'}))

    # Level 2
    save_l2 = (df_l2
               .drop(columns=['t_sig', 'w_sig'])
               .rename(columns={'N_cases': 'N'}))
    save_l2 = save_l2.assign(oar_or_agg='thin_wall_mean', priority=False)

    combined = pd.concat([save_l1, save_l2], ignore_index=True)
    combined.to_csv(summary_path, index=False, float_format='%.6f')
    print(f'\nSaved summary → {summary_path}  ({len(combined)} rows)')

    # ---- Interpretation ----
    print_interpretation(df_l1, df_l2, df_l1_fdr, df_dir)


if __name__ == '__main__':
    main()
