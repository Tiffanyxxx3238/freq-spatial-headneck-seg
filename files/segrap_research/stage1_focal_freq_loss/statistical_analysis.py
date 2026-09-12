"""
Statistical comparison of Stage-1 FFL variants vs Baseline.

Usage (from segrap_research/):
    python stage1_focal_freq_loss/statistical_analysis.py
    python stage1_focal_freq_loss/statistical_analysis.py --results_dir /path/to/results/stage1/results

Two analysis levels:
  Level 1 — per-OAR paired tests (18 cases each).
  Level 2 — case-level thin-wall mean tests (aggregates the 10 OARs per case first,
             avoids pseudoreplication from treating OARs as independent observations).

Priority OARs are printed first in every table (observed to show consistent trends
across lambdas): MiddleEar_L, MiddleEar_R, IAC_R, VestibulSemi_L.
"""
import argparse
import os
import sys
import warnings

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
OTHER_OARS = [o for o in THIN_WALL_OARS if o not in PRIORITY_OARS]
ORDERED_OARS = PRIORITY_OARS + OTHER_OARS

LAMBDAS = [0.05, 0.10, 0.20]

# Default metric columns; remapped to canonical names after CSV load
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
HIGHER_BETTER = {'dice', 'surface_dice_1mm', 'surface_dice_2mm'}   # hd95: lower is better


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _stars(p: float) -> str:
    if np.isnan(p):
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
    Paired t-test and Wilcoxon signed-rank on finite pairs.
    diff = b - a (FFL - baseline); positive = FFL better for dice/surface_dice,
    negative = FFL better for hd95.
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

def level1_oar(df_base: pd.DataFrame, df_ffl: pd.DataFrame,
               lam: float, col_map: dict) -> pd.DataFrame:
    """Per-OAR paired tests. Returns one row per (oar, metric)."""
    rows = []
    for oar in ORDERED_OARS:
        b_oar = df_base[df_base['oar_name'] == oar].sort_values('case_id')
        f_oar = df_ffl[df_ffl['oar_name'] == oar].sort_values('case_id')
        # Inner-join on case_id to guard against mismatched rows
        merged = b_oar.set_index('case_id')[list(col_map.values())].join(
            f_oar.set_index('case_id')[list(col_map.values())],
            lsuffix='_b', rsuffix='_f', how='inner',
        )
        for canon, col in col_map.items():
            a = merged[f'{col}_b'].values.astype(float)
            b = merged[f'{col}_f'].values.astype(float)
            res = _paired_tests(a, b)
            rows.append({
                'lambda':   lam,
                'level':    1,
                'oar':      oar,
                'priority': oar in PRIORITY_OARS,
                'metric':   canon,
                'N':        res['n'],
                'mean_base': res['mean_a'],
                'mean_ffl':  res['mean_b'],
                'diff':      res['diff'],
                't_pval':    res['t_p'],
                'w_pval':    res['w_p'],
                't_sig':     _stars(res['t_p']),
                'w_sig':     _stars(res['w_p']),
            })
    return pd.DataFrame(rows)


def level2_case_mean(df_base: pd.DataFrame, df_ffl: pd.DataFrame,
                     lam: float, col_map: dict) -> pd.DataFrame:
    """
    Case-level mean: average the 10 OAR metric values per case first,
    then do paired tests on those 18 case-level means.
    """
    rows = []
    # Restrict to the 10 thin-wall OARs (they are all thin-wall already in these CSVs)
    for canon, col in col_map.items():
        case_b = (df_base.groupby('case_id')[col]
                  .apply(lambda x: np.nanmean(x[np.isfinite(x)])))
        case_f = (df_ffl.groupby('case_id')[col]
                  .apply(lambda x: np.nanmean(x[np.isfinite(x)])))
        common = case_b.index.intersection(case_f.index).sort_values()
        a = case_b.loc[common].values.astype(float)
        b = case_f.loc[common].values.astype(float)
        res = _paired_tests(a, b)
        rows.append({
            'lambda':    lam,
            'level':     2,
            'metric':    canon,
            'N_cases':   res['n'],
            'mean_base': res['mean_a'],
            'mean_ffl':  res['mean_b'],
            'diff':      res['diff'],
            't_pval':    res['t_p'],
            'w_pval':    res['w_p'],
            't_sig':     _stars(res['t_p']),
            'w_sig':     _stars(res['w_p']),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Printing                                                             #
# ------------------------------------------------------------------ #

W = 82

def _rule(char='='):
    print(char * W)

def _header(title: str):
    _rule()
    print(f'  {title}')
    _rule()


def print_level1_table(df: pd.DataFrame, lam: float, metric: str):
    sub = df[(df['lambda'] == lam) & (df['metric'] == metric)].copy()
    if sub.empty:
        return
    label = METRIC_LABELS.get(metric, metric)
    higher = metric in HIGHER_BETTER
    sign_hint = '(↑ better)' if higher else '(↓ better)'
    _rule('-')
    print(f'  Level 1 | λ={lam} vs Baseline | {label} {sign_hint}')
    _rule('-')
    print(f'  {"★":<3}  {"OAR":<22}  {"N":>2}  {"Base":>8}  {"FFL":>8}  '
          f'{"Diff":>8}  {"t-p":>7}  {"":>2}  {"Wilcox-p":>9}  {"":>2}')
    print(f'  {"-"*3}  {"-"*22}  {"--":>2}  {"-"*8}  {"-"*8}  '
          f'{"-"*8}  {"-"*7}  {"--":>2}  {"-"*9}  {"--":>2}')
    for _, row in sub.iterrows():
        star = '★' if row['priority'] else ' '
        diff_str = f'{row["diff"]:+.4f}' if not np.isnan(row['diff']) else '   nan  '
        base_str = f'{row["mean_base"]:.4f}' if not np.isnan(row['mean_base']) else '    nan '
        ffl_str  = f'{row["mean_ffl"]:.4f}'  if not np.isnan(row['mean_ffl'])  else '    nan '
        tp_str   = f'{row["t_pval"]:.4f}'    if not np.isnan(row['t_pval'])    else '    nan '
        wp_str   = f'{row["w_pval"]:.4f}'    if not np.isnan(row['w_pval'])    else '    nan '
        print(f'  {star:<3}  {row["oar"]:<22}  {int(row["N"]):>2}  '
              f'{base_str:>8}  {ffl_str:>8}  {diff_str:>8}  '
              f'{tp_str:>7}  {row["t_sig"]:>2}  {wp_str:>9}  {row["w_sig"]:>2}')
    # Count significant per test
    n_t = (sub['t_pval'] < 0.05).sum()
    n_w = (sub['w_pval'] < 0.05).sum()
    print(f'  Significant (p<0.05): t-test {n_t}/{len(sub)}  Wilcoxon {n_w}/{len(sub)}')


def print_level2_table(df: pd.DataFrame, lam: float):
    sub = df[df['lambda'] == lam].copy()
    if sub.empty:
        return
    _rule('-')
    print(f'  Level 2 | λ={lam} vs Baseline | Case-level thin-wall means (N cases)')
    _rule('-')
    print(f'  {"Metric":<18}  {"N":>2}  {"Base":>8}  {"FFL":>8}  '
          f'{"Diff":>8}  {"t-p":>7}  {"":>2}  {"Wilcox-p":>9}  {"":>2}')
    print(f'  {"-"*18}  {"--":>2}  {"-"*8}  {"-"*8}  '
          f'{"-"*8}  {"-"*7}  {"--":>2}  {"-"*9}  {"--":>2}')
    for _, row in sub.iterrows():
        label = METRIC_LABELS.get(row['metric'], row['metric'])
        diff_str = f'{row["diff"]:+.4f}' if not np.isnan(row['diff']) else '   nan  '
        base_str = f'{row["mean_base"]:.4f}' if not np.isnan(row['mean_base']) else '    nan '
        ffl_str  = f'{row["mean_ffl"]:.4f}'  if not np.isnan(row['mean_ffl'])  else '    nan '
        tp_str   = f'{row["t_pval"]:.4f}'    if not np.isnan(row['t_pval'])    else '    nan '
        wp_str   = f'{row["w_pval"]:.4f}'    if not np.isnan(row['w_pval'])    else '    nan '
        print(f'  {label:<18}  {int(row["N_cases"]):>2}  '
              f'{base_str:>8}  {ffl_str:>8}  {diff_str:>8}  '
              f'{tp_str:>7}  {row["t_sig"]:>2}  {wp_str:>9}  {row["w_sig"]:>2}')


# ------------------------------------------------------------------ #
# FDR (Benjamini-Hochberg) correction                                  #
# ------------------------------------------------------------------ #

def _apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add t_pval_fdr and w_pval_fdr columns to df_l1 (Level-1 rows only).

    BH correction is applied to all N rows simultaneously — across all
    lambda values, OARs, and metrics — separately for t-test and Wilcoxon.
    NaN p-values (too few finite pairs) are excluded from the correction
    pool and mapped back as NaN in the output.
    """
    if not _STATSMODELS:
        raise RuntimeError(
            'statsmodels not installed. Run: pip install statsmodels'
        )
    df = df.copy()
    for col, out_col in [('t_pval', 't_pval_fdr'), ('w_pval', 'w_pval_fdr')]:
        pvals = df[col].values.astype(float)
        finite = np.isfinite(pvals)
        corrected = np.full(len(pvals), np.nan)
        if finite.sum() >= 2:
            _, adj, _, _ = _sm_multipletests(
                pvals[finite], alpha=0.05, method='fdr_bh'
            )
            corrected[finite] = adj
        df[out_col] = corrected
    return df


def print_fdr_table(df_fdr: pd.DataFrame, out_dir: str):
    """
    Print FDR-corrected results and save fdr_corrected_summary.csv.

    df_fdr is the Level-1 df after _apply_fdr; it has t_pval_fdr and
    w_pval_fdr columns in addition to the original t_pval / w_pval.

    Only rows where the *raw* p-value is < 0.05 for at least one test
    are shown — i.e., the items that looked significant before correction.
    """
    n_total = len(df_fdr)
    # Rows nominally significant in at least one test
    sig_mask = (df_fdr['t_pval'] < 0.05) | (df_fdr['w_pval'] < 0.05)
    n_raw_sig = int(sig_mask.sum())

    df_sig = df_fdr[sig_mask].copy()

    # After FDR: significant in at least one corrected test
    post_mask = (df_sig['t_pval_fdr'] < 0.05) | (df_sig['w_pval_fdr'] < 0.05)
    n_fdr_sig = int(post_mask.sum())

    print()
    _rule('#')
    print(f'  FDR CORRECTION (Benjamini-Hochberg, α=0.05)')
    print(f'  Family size: {n_total} comparisons  '
          f'(t-test corrected separately from Wilcoxon)')
    _rule('#')

    if df_sig.empty:
        print('  No nominally significant results to display.')
    else:
        # Sort: lambda, metric, then by raw t_pval
        df_sig = df_sig.sort_values(['lambda', 'metric', 't_pval'])
        print(f'\n  Showing {n_raw_sig} rows with raw p<0.05 (either test)')
        print()
        hdr = (f'  {"λ":>4}  {"OAR":<22}  {"Metric":<16}  '
               f'{"t-raw":>7}  {"t-FDR":>7}  {"t✓":>3}  '
               f'{"w-raw":>7}  {"w-FDR":>7}  {"w✓":>3}')
        sep = (f'  {"----":>4}  {"-"*22}  {"-"*16}  '
               f'{"-"*7}  {"-"*7}  {"--":>3}  '
               f'{"-"*7}  {"-"*7}  {"--":>3}')
        print(hdr)
        print(sep)
        for _, row in df_sig.iterrows():
            def _fmt_p(v):
                return f'{v:.4f}' if np.isfinite(v) else '   nan'
            t_ok = '✓' if (np.isfinite(row['t_pval_fdr']) and row['t_pval_fdr'] < 0.05) else '✗'
            w_ok = '✓' if (np.isfinite(row['w_pval_fdr']) and row['w_pval_fdr'] < 0.05) else '✗'
            pri  = '★' if row['priority'] else ' '
            print(f'  {row["lambda"]:>4.2f}  {pri}{row["oar"]:<21}  '
                  f'{row["metric"]:<16}  '
                  f'{_fmt_p(row["t_pval"]):>7}  {_fmt_p(row["t_pval_fdr"]):>7}  {t_ok:>3}  '
                  f'{_fmt_p(row["w_pval"]):>7}  {_fmt_p(row["w_pval_fdr"]):>7}  {w_ok:>3}')

    print()
    _rule()
    print(f'  原始顯著項目數（raw p<0.05，either test）：{n_raw_sig}')
    print(f'  FDR校正後仍顯著項目數（adj p<0.05，either test）：{n_fdr_sig}')
    _rule()

    # Save CSV
    os.makedirs(out_dir, exist_ok=True)
    fdr_path = os.path.join(out_dir, 'fdr_corrected_summary.csv')
    save_cols = ['lambda', 'oar', 'priority', 'metric', 'N',
                 'mean_base', 'mean_ffl', 'diff',
                 't_pval', 't_pval_fdr', 'w_pval', 'w_pval_fdr']
    df_fdr[save_cols].to_csv(fdr_path, index=False, float_format='%.6f')
    print(f'  Saved FDR table → {fdr_path}  ({len(df_fdr)} rows, all comparisons)')
    _rule()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./results/stage1/results',
                    help='Directory containing the four test-result CSVs')
    args = ap.parse_args()
    rd = args.results_dir

    csv_paths = {
        'baseline': os.path.join(rd, 's1_baseline_full_test_results.csv'),
        0.05:       os.path.join(rd, 's1_ffl005_test_results.csv'),
        0.10:       os.path.join(rd, 's1_ffl010_test_results.csv'),
        0.20:       os.path.join(rd, 's1_ffl020_test_results.csv'),
    }

    # ---- Load CSVs ----
    dfs = {}
    print('\nLoading CSVs...')
    for key, path in csv_paths.items():
        if not os.path.exists(path):
            print(f'  [MISSING] {path}')
            continue
        df = pd.read_csv(path)
        dfs[key] = df
        print(f'  [OK] {os.path.basename(path)}  {len(df)} rows  cols={list(df.columns)}')

    if 'baseline' not in dfs:
        sys.exit(f'\nERROR: baseline CSV not found: {csv_paths["baseline"]}')

    df_base = dfs['baseline']
    col_map = _detect_metrics(df_base)
    if not col_map:
        sys.exit('ERROR: could not detect any metric columns in baseline CSV.')
    print(f'\nDetected metric columns: {col_map}')

    available_lambdas = [lam for lam in LAMBDAS if lam in dfs]
    if not available_lambdas:
        sys.exit('ERROR: no FFL CSVs found.')

    # ---- Run analyses ----
    all_l1_frames = []
    all_l2_frames = []
    for lam in available_lambdas:
        df_ffl = dfs[lam]
        all_l1_frames.append(level1_oar(df_base, df_ffl, lam, col_map))
        all_l2_frames.append(level2_case_mean(df_base, df_ffl, lam, col_map))

    df_l1 = pd.concat(all_l1_frames, ignore_index=True)
    df_l2 = pd.concat(all_l2_frames, ignore_index=True)

    # ---- Print ----
    print()
    _rule('#')
    print(f'  STATISTICAL ANALYSIS: Stage-1 FFL variants vs Baseline')
    _rule('#')
    print(f'\n  Priority OARs (printed first): {", ".join(PRIORITY_OARS)}')
    print('  sig: * p<0.05  ** p<0.01  (no multiple-comparison correction)')

    for lam in available_lambdas:
        print()
        _rule('#')
        print(f'  λ = {lam}')
        _rule('#')
        for metric in col_map:
            print_level1_table(df_l1, lam, metric)
        print()
        print_level2_table(df_l2, lam)

    # ---- FDR correction (Level-1 only; Level-2 has only 12 tests) ----
    out_dir = os.path.normpath(os.path.join(rd, '..'))    # ./results/stage1/
    if _STATSMODELS:
        df_l1_fdr = _apply_fdr(df_l1)
        print_fdr_table(df_l1_fdr, out_dir)
    else:
        print('\n  [SKIP] FDR correction requires statsmodels: pip install statsmodels')

    # ---- Save summary CSV ----
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'statistical_analysis_summary.csv')

    # Level-1 rows (level col already present from row dicts)
    save_l1 = df_l1.drop(columns=['t_sig', 'w_sig']).rename(columns={'oar': 'oar_or_agg'})

    # Level-2 rows
    save_l2 = df_l2.drop(columns=['t_sig', 'w_sig']).rename(columns={'N_cases': 'N'})
    save_l2 = save_l2.assign(oar_or_agg='thin_wall_mean', priority=False)

    combined = pd.concat([save_l1, save_l2], ignore_index=True)
    combined.to_csv(summary_path, index=False, float_format='%.6f')
    print(f'\n\nSaved summary → {summary_path}')
    _rule()


if __name__ == '__main__':
    main()
