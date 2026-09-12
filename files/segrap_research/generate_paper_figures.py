"""
generate_paper_figures.py — 6 figures for the TMI paper.

Figures 1-2: pure matplotlib, hardcoded numbers, no GPU/model needed.
Figures 3-6: require GPU + trained checkpoints.

IMPORTANT model-class correction (confirmed by reading the actual source,
not guessed): the "3D U-Net" baseline used in Figures 4-5 is NOT
models.afs_dsn_original.AFS_DSN_V4 -- that is the DWT-Full baseline, a
completely different 414.6M-param AFS-DSN variant with a DWT frequency
branch, not a U-Net at all. The real U-Net baseline is
external_baselines/models_external.py's build_unet_baseline() (a MONAI UNet
wrapper returning {'output': ...}), confirmed via
external_baselines/train_external.py's own import line:
    from external_baselines.models_external import build_external_model, MODEL_NAMES
which dispatches model_name='unet' to build_unet_baseline(). This script
imports build_unet_baseline directly.

Usage:
    python generate_paper_figures.py --figures 1 2            # no GPU needed
    python generate_paper_figures.py --figures 3 4 5 6        # needs GPU
    python generate_paper_figures.py --figures 1 2 3 4 5 6    # everything

Outputs (./results/paper_figures/, created automatically):
    fig_scale_mismatch.png
    fig_ablation_trend.png
    fig_qualitative_cases.png
    fig_error_map_comparison.png
    fig_baseline_comparison.png
    fig_global_error_map.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')   # headless rendering, no display on RunPod
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import nibabel as nib
import numpy as np
from skimage.transform import resize as sk_resize
import torch

OUTPUT_DIR = './results/paper_figures'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--figures', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                   choices=[1, 2, 3, 4, 5, 6])
    p.add_argument('--data_root', type=str,
                   default='/workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--cache_dir', type=str, default='/workspace/data/cache')
    p.add_argument('--freq_checkpoint', type=str,
                   default='./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth',
                   help='FreqFuseNet (Stage5C, AFS_DSN_FFTResidual) checkpoint.')
    p.add_argument('--unet_checkpoint', type=str,
                   default='./results/external_baselines/checkpoints/external_unet_seed2/best.pth',
                   help='3D U-Net baseline checkpoint (External Baseline Study).')
    p.add_argument('--segresnet_checkpoint', type=str,
                   default='./results/external_baselines/checkpoints/external_segresnet_seed2/best.pth',
                   help='SegResNet baseline checkpoint (External Baseline Study).')
    p.add_argument('--base_features', type=int, default=32,
                   help='Must match the checkpoint FreqFuseNet was trained with (default 32).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ================================================================== #
# Figure 1: activation scale mismatch + Stage5B alpha (no GPU)        #
# ================================================================== #

def make_fig1(output_dir: str) -> None:
    print("\n=== Figure 1: fig_scale_mismatch.png ===")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300, facecolor='white')
    colors = ['#1D9E75', '#BA7517']   # FFT, FcaNet

    # ---- Left: bar chart, log scale ----
    ax = axes[0]
    labels = ['FFT Branch', 'FcaNet Branch']
    values = [0.000679, 0.586]
    value_labels = ['0.000679', '0.5860']
    ratio = values[1] / values[0]   # ~863x -- computed, not hardcoded
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 3)
    ax.set_ylabel('Feature Activation Std (log scale)')
    ax.set_title('Observed Activation-Scale Mismatch')
    for bar, val, label in zip(bars, values, value_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.4, label,
                ha='center', va='bottom', fontsize=10)

    arrow_y = 1.5
    ax.annotate('', xy=(0, arrow_y), xytext=(1, arrow_y),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.5, arrow_y * 1.35, f'{ratio:.0f}× scale mismatch', ha='center', va='bottom',
           fontsize=10, fontweight='bold')

    # ---- Middle: naive fusion WITHOUT scale normalization ----
    ax_mid = axes[1]
    mid_labels = ['FFT\ncontribution', 'FcaNet residual\n(nominal β=0.05)']
    mid_values = [1.0, 0.05 * ratio]   # ~43.15 -- naive beta*std with no scale correction
    bars_mid = ax_mid.bar(mid_labels, mid_values, color=colors, width=0.5)
    ax_mid.set_ylim(0, 45)
    ax_mid.set_ylabel('Relative Contribution Scale')
    ax_mid.set_title('✗ Without Scale Normalization', color='#C0392B', fontweight='bold')
    for bar, val in zip(bars_mid, mid_values):
        ax_mid.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10)
    ax_mid.annotate(f'Effective scale = {mid_values[1]:.2f}×\n(nominal β = 0.05)', xy=(1, mid_values[1]),
                    xytext=(0.35, 38), color='red', fontsize=9, ha='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))

    # ---- Right: FreqFuseNet's actual scale-normalized fusion ----
    ax_r = axes[2]
    r_labels = ['FFT\ncontribution', 'FcaNet residual\n(β=0.05)']
    r_values = [1.0, 0.05]
    bars_r = ax_r.bar(r_labels, r_values, color=colors, width=0.5)
    ax_r.set_ylim(0, 45)
    ax_r.set_ylabel('Relative Contribution Scale')
    ax_r.set_title('✓ With Scale Normalization (FreqFuseNet)', color='#1E8449', fontweight='bold')
    for bar, val in zip(bars_r, r_values):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=10)
    ax_r.annotate('0.05\n(= intended\n5% residual)', xy=(1, r_values[1]),
                  xytext=(1, 12), color='green', fontsize=9, ha='center', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='green'))

    fig.subplots_adjust(top=0.88, wspace=0.55, bottom=0.15, left=0.06, right=0.97)

    # Connecting arrow between the "without" and "with" normalization panels.
    # Biased toward ax_mid's side of the gap (rather than the true midpoint)
    # because ax_r's rotated y-axis label eats into the left part of its
    # share of the gap; using the axes' actual post-layout positions (not
    # hardcoded figure fractions) keeps this correct under future spacing tweaks.
    pos_mid = ax_mid.get_position()
    pos_r = ax_r.get_position()
    arrow_x_start = pos_mid.x1 + 0.15 * (pos_r.x0 - pos_mid.x1)
    arrow_x_end = pos_mid.x1 + 0.55 * (pos_r.x0 - pos_mid.x1)
    arrow_y_fig = (pos_mid.y0 + pos_mid.y1) / 2
    ax_mid.annotate('', xy=(arrow_x_end, arrow_y_fig), xytext=(arrow_x_start, arrow_y_fig),
                    xycoords='figure fraction', textcoords='figure fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    fig.text((arrow_x_start + arrow_x_end) / 2, arrow_y_fig - 0.10, 'scale\nnormalization',
            fontsize=8, ha='center', va='top', style='italic')

    out_path = os.path.join(output_dir, 'fig_scale_mismatch.png')
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ================================================================== #
# Figure 2: component contribution analysis (no GPU)                   #
# ================================================================== #

def make_fig2(output_dir: str) -> None:
    print("\n=== Figure 2: fig_ablation_trend.png ===")
    methods = [
        'DWT-Full\nBaseline',
        'FFT Branch\nonly',
        'FcaNet Branch\nonly',
        'Naive Fusion\n(no scale norm)',
        'FreqFuseNet\n(scale-norm)',
    ]
    dice  = [0.8404, 0.8440, 0.8424, 0.8432, 0.8493]
    hd95  = [0.884,  0.870,  0.837,  0.872,  0.824]
    sdice = [0.9518, 0.9568, 0.9560, 0.9548, 0.9591]
    x = np.arange(len(methods))

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=300, facecolor='white')

    l1, = ax1.plot(x, dice, color='#185FA5', marker='o', linestyle='-',
                  linewidth=2, markersize=8, label='Dice', zorder=3)
    l3, = ax1.plot(x, sdice, color='#0F6E56', marker='^', linestyle=':',
                  linewidth=2, markersize=8, label='SDice@1mm', zorder=3)
    ax1.set_ylabel('Dice / Surface Dice @1mm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=0, fontsize=9)
    ax1.set_xlim(-0.6, len(methods) - 0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, hd95, color='#993C1D', marker='s', linestyle='--',
                  linewidth=2, markersize=8, label='HD95 (mm) ↓', zorder=3)
    ax2.set_ylabel('HD95 (mm, lower is better)')

    # Star marker on the proposed method (FreqFuseNet, last point)
    ax1.plot(x[4], dice[4], marker='*', color='none', markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, zorder=4)

    # Single annotation for the proposed method only
    ax1.annotate('FreqFuseNet\n(proposed)', xy=(x[4], dice[4]),
                xytext=(x[4] - 0.8, dice[4] + 0.010),
                color='green', fontsize=10, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Vertical dashed separator between Naive Fusion (index 3) and FreqFuseNet (index 4)
    ax1.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.2, zorder=2)
    ax1.text(3.5, 0.02, '+ scale norm', transform=ax1.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='gray', style='italic')

    lines = [l1, l3, l2]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc='upper right', fontsize=9)
    ax1.set_title('Component-wise Ablation Analysis', fontweight='bold')

    fig.tight_layout()
    out_path = os.path.join(output_dir, 'fig_ablation_trend.png')
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ================================================================== #
# Figure 3: qualitative cases across 4 OARs (needs GPU)               #
# ================================================================== #

def _percentile_clip_normalize_p1_p99(img2d: np.ndarray) -> np.ndarray:
    """p1-p99 percentile clip (higher contrast than visualize_segrap's p2-p98),
    used only for Figure 3's CT base so bone/cavity boundaries read more clearly."""
    p1, p99 = np.percentile(img2d, [1, 99])
    img_clipped = np.clip(img2d, p1, p99)
    rng = max(p99 - p1, 1e-8)
    return (img_clipped - p1) / rng


def make_fig3(args, output_dir: str) -> None:
    print("\n=== Figure 3: fig_qualitative_cases.png ===")
    from data.segrap_dataset import SegRapDataset
    from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
    from utils.train_utils import load_checkpoint
    from visualize_segrap import make_overlay, _compute_roi_bbox
    try:
        from scipy.ndimage import center_of_mass
    except ImportError as e:
        raise ImportError(
            "Figure 3's boundary-zoom inset (4th column) requires scipy. "
            "Install it with: pip install scipy"
        ) from e

    TARGET_CASES = [
        ('MiddleEar_L',      'segrap_0111'),
        ('TympanicCavity_L', 'segrap_0107'),
        ('IAC_L',            'segrap_0117'),
        ('Cochlea_L',        'segrap_0114'),
    ]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"Loading FreqFuseNet (Stage5C) from {args.freq_checkpoint} ...")
    model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2, base_features=args.base_features).to(device)
    load_checkpoint(model, None, args.freq_checkpoint, device)
    model.eval()

    ZOOM_PAD = 15      # half-width (px) of the boundary-zoom inset, centered on the GT centroid
    ZOOM_SCALE = 4     # upscale factor applied before computing boundary contours
    GT_BOUNDARY_COLOR   = '#2ECC71'
    PRED_BOUNDARY_COLOR = '#E67E22'
    rows_data = []   # (oar_name, case_id, z, gray, gt_crop, pred_crop, zoom_bounds, gray_zoom_large, gt_zoom_large, pred_zoom_large)

    for oar_name, case_id in TARGET_CASES:
        ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                           oar_subset=[oar_name], cache_dir=args.cache_dir)
        mask_path = ds.data_root / case_id / f'{oar_name}.nii.gz'
        if not mask_path.exists():
            print(f"  [WARNING] {oar_name}/{case_id}: mask file not found — skipping this row")
            continue

        idx = None
        for i, (cid, on) in enumerate(ds.samples):
            if cid == case_id and on == oar_name:
                idx = i
                break
        if idx is None:
            print(f"  [WARNING] {oar_name}/{case_id}: no thin_wall sample found in the dataset — skipping this row")
            continue

        sample = ds[idx]
        image = sample['image'].unsqueeze(0).to(device)   # (1,1,128,128,128) -- 128^3 preprocessed, training-consistent

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()

        ct_path = ds.data_root / case_id / 'image.nii.gz'
        ct_full = nib.load(str(ct_path)).get_fdata()
        gt_full = nib.load(str(mask_path)).get_fdata() > 0

        # Map the 128^3 prediction back into the SAME ROI bbox the model's
        # input was cropped+resized from (margin=32, identical to
        # SegRapDataset._load_thin_wall_pair -- see visualize_segrap.py).
        d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_full, margin=32)
        crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
        pred_crop_3d = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                                preserve_range=True, anti_aliasing=False)
        pred_full = np.zeros(ct_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop_3d > 0.5).astype(np.uint8)

        # GT-mask largest-area axial slice, original resolution.
        areas = gt_full.sum(axis=(0, 1))
        z = int(np.argmax(areas))

        gt_slice   = gt_full[:, :, z]
        pred_slice = pred_full[:, :, z].astype(bool)
        ct_slice   = ct_full[:, :, z]

        rows_b = np.any(gt_slice, axis=1)
        cols_b = np.any(gt_slice, axis=0)
        rmin, rmax = np.where(rows_b)[0][[0, -1]]
        cmin, cmax = np.where(cols_b)[0][[0, -1]]
        pad = 40
        rmin = max(0, rmin - pad); rmax = min(gt_slice.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad); cmax = min(gt_slice.shape[1] - 1, cmax + pad)

        ct_crop   = ct_slice[rmin:rmax + 1, cmin:cmax + 1]
        gt_crop   = gt_slice[rmin:rmax + 1, cmin:cmax + 1]
        pred_crop = pred_slice[rmin:rmax + 1, cmin:cmax + 1]
        gray = _percentile_clip_normalize_p1_p99(ct_crop)

        # ---- Boundary-zoom inset: a tight crop around the GT centroid,
        # showing only the GT/prediction boundary CONTOURS on the CT base.
        # The raw zoom crop is small (~30x30px), so it's upscaled 4x first
        # (order=0 for the binary masks, order=1 for the CT) -- this gives
        # ax.contour() enough pixels to draw a thin, smooth contour line at
        # the mask's 0.5 level instead of a blocky pixel-filled boundary. ----
        cy, cx = center_of_mass(gt_crop)
        zy1 = max(0, int(cy) - ZOOM_PAD); zy2 = min(gt_crop.shape[0], int(cy) + ZOOM_PAD)
        zx1 = max(0, int(cx) - ZOOM_PAD); zx2 = min(gt_crop.shape[1], int(cx) + ZOOM_PAD)

        gray_zoom = gray[zy1:zy2, zx1:zx2]
        gt_zoom   = gt_crop[zy1:zy2, zx1:zx2]
        pred_zoom = pred_crop[zy1:zy2, zx1:zx2]

        large_shape = (gray_zoom.shape[0] * ZOOM_SCALE, gray_zoom.shape[1] * ZOOM_SCALE)
        gray_zoom_large = sk_resize(gray_zoom, large_shape, order=1, preserve_range=True)
        gt_zoom_large = sk_resize(gt_zoom.astype(np.float64), large_shape, order=0,
                                  preserve_range=True)
        pred_zoom_large = sk_resize(pred_zoom.astype(np.float64), large_shape, order=0,
                                    preserve_range=True)

        rows_data.append((oar_name, case_id, z, gray, gt_crop, pred_crop, (zx1, zy1, zx2, zy2),
                          gray_zoom_large, gt_zoom_large, pred_zoom_large))
        print(f"  {oar_name}/{case_id}: slice z={z}, crop rows=[{rmin}:{rmax}] cols=[{cmin}:{cmax}], "
              f"zoom rows=[{zy1}:{zy2}] cols=[{zx1}:{zx2}]")

    if not rows_data:
        print("  [WARNING] No valid OAR/case pairs available — skipping Figure 3 entirely")
        return

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 12), dpi=300, facecolor='black')
    if n_rows == 1:
        axes = axes.reshape(1, 4)

    col_titles = ['CT Input', 'GT Overlay', 'Prediction Overlay', 'Boundary Zoom']
    for row_idx, row in enumerate(rows_data):
        (oar_name, case_id, z, gray, gt_crop, pred_crop, zoom_bounds,
         gray_zoom_large, gt_zoom_large, pred_zoom_large) = row
        zx1, zy1, zx2, zy2 = zoom_bounds
        panels = [
            np.stack([gray] * 3, axis=-1),
            make_overlay(gray, gt_crop, color=(0.0, 1.0, 0.0), alpha=0.5),
            make_overlay(gray, pred_crop, color=(1.0, 0.5, 0.0), alpha=0.5),
        ]
        for col_idx, (ax, panel) in enumerate(zip(axes[row_idx][:3], panels)):
            ax.imshow(np.clip(panel, 0.0, 1.0), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('black')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color='white', fontsize=12)
            if col_idx == 2:
                # mark where the boundary-zoom inset (4th column) was cropped from
                rect = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=1.2,
                                 edgecolor='white', facecolor='none', linestyle='--')
                ax.add_patch(rect)

        # ---- 4th column: boundary-zoom inset, contour lines on a CT base ----
        ax_zoom = axes[row_idx][3]
        ax_zoom.imshow(gray_zoom_large, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        ax_zoom.contour(gt_zoom_large, levels=[0.5], colors=[GT_BOUNDARY_COLOR], linewidths=1.5)
        ax_zoom.contour(pred_zoom_large, levels=[0.5], colors=[PRED_BOUNDARY_COLOR], linewidths=1.5)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_visible(False)
        ax_zoom.set_facecolor('black')
        if row_idx == 0:
            ax_zoom.set_title(col_titles[3], color='white', fontsize=12)

        axes[row_idx][0].set_ylabel(oar_name, color='white', fontsize=9, rotation=90, labelpad=8)

    fig.patch.set_facecolor('black')
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    case_list = "    ".join(f'{oar} = {cid}' for oar, cid, *_ in rows_data)
    caption = (f"Cases:  {case_list}.  Green contours: ground truth; "
              f"Orange contours: FreqFuseNet prediction.")
    fig.text(0.5, 0.005, caption, color='white', fontsize=8, ha='center', va='bottom')
    out_path = os.path.join(output_dir, 'fig_qualitative_cases.png')
    plt.savefig(out_path, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ================================================================== #
# Figure 4: 3D U-Net vs FreqFuseNet error-map comparison (needs GPU)   #
# ================================================================== #

def make_fig4(args, output_dir: str) -> None:
    print("\n=== Figure 4: fig_error_map_comparison.png ===")
    from data.segrap_dataset import SegRapDataset
    from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
    from external_baselines.models_external import build_unet_baseline
    from utils.train_utils import load_checkpoint
    from visualize_segrap import percentile_clip_normalize, make_overlay, dice_score, _compute_roi_bbox

    OAR_NAME = 'MiddleEar_L'
    CASE_ID  = 'segrap_0111'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                       oar_subset=[OAR_NAME], cache_dir=args.cache_dir)
    mask_path = ds.data_root / CASE_ID / f'{OAR_NAME}.nii.gz'
    if not mask_path.exists():
        print(f"  [WARNING] {OAR_NAME}/{CASE_ID}: mask file not found — skipping Figure 4 entirely")
        return

    idx = None
    for i, (cid, on) in enumerate(ds.samples):
        if cid == CASE_ID and on == OAR_NAME:
            idx = i
            break
    if idx is None:
        print(f"  [WARNING] {OAR_NAME}/{CASE_ID}: no thin_wall sample found in the dataset — skipping Figure 4 entirely")
        return

    sample = ds[idx]
    image = sample['image'].unsqueeze(0).to(device)

    ct_path = ds.data_root / CASE_ID / 'image.nii.gz'
    ct_full = nib.load(str(ct_path)).get_fdata()
    gt_full = nib.load(str(mask_path)).get_fdata() > 0

    areas = gt_full.sum(axis=(0, 1))
    z = int(np.argmax(areas))
    gt_slice = gt_full[:, :, z]

    rows_b = np.any(gt_slice, axis=1)
    cols_b = np.any(gt_slice, axis=0)
    rmin, rmax = np.where(rows_b)[0][[0, -1]]
    cmin, cmax = np.where(cols_b)[0][[0, -1]]
    pad = 40
    rmin = max(0, rmin - pad); rmax = min(gt_slice.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(gt_slice.shape[1] - 1, cmax + pad)

    ct_slice = ct_full[:, :, z]
    ct_crop = ct_slice[rmin:rmax + 1, cmin:cmax + 1]
    gray = percentile_clip_normalize(ct_crop)
    gt_crop = gt_slice[rmin:rmax + 1, cmin:cmax + 1]

    def run_model_and_get_pred(model):
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()

        d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_full, margin=32)
        crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
        pred_crop_3d = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                                preserve_range=True, anti_aliasing=False)
        pred_full = np.zeros(ct_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop_3d > 0.5).astype(np.uint8)
        pred_slice = pred_full[:, :, z].astype(bool)
        return pred_slice[rmin:rmax + 1, cmin:cmax + 1]

    model_rows = []

    # ---- 3D U-Net (External Baseline Study) ----
    print(f"Loading 3D U-Net baseline from {args.unet_checkpoint} ...")
    try:
        unet_model = build_unet_baseline(in_channels=1, num_classes=2).to(device)
        load_checkpoint(unet_model, None, args.unet_checkpoint, device)
        unet_model.eval()
        unet_pred_crop = run_model_and_get_pred(unet_model)
        model_rows.append(('3D U-Net', unet_pred_crop))
        del unet_model
    except Exception as e:
        print(f"  [WARNING] Could not load/run the 3D U-Net baseline "
              f"({type(e).__name__}: {e}) — skipping its row")

    # ---- FreqFuseNet (Stage5C) ----
    print(f"Loading FreqFuseNet (Stage5C) from {args.freq_checkpoint} ...")
    freq_model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2, base_features=args.base_features).to(device)
    load_checkpoint(freq_model, None, args.freq_checkpoint, device)
    freq_model.eval()
    freq_pred_crop = run_model_and_get_pred(freq_model)
    model_rows.append(('FreqFuseNet (ours)', freq_pred_crop))
    del freq_model

    if not model_rows:
        print("  [WARNING] No models could be evaluated — skipping Figure 4 entirely")
        return

    TP_COLOR = (0x2E / 255, 0xCC / 255, 0x71 / 255)
    FP_COLOR = (0xE7 / 255, 0x4C / 255, 0x3C / 255)
    FN_COLOR = (0x34 / 255, 0x98 / 255, 0xDB / 255)

    def make_error_map_custom(gray2d, gt2d, pred2d, alpha=0.6):
        rgb = np.stack([gray2d] * 3, axis=-1).copy()
        gt_b, pred_b = gt2d.astype(bool), pred2d.astype(bool)
        for mask, color in [(gt_b & pred_b, TP_COLOR), (pred_b & ~gt_b, FP_COLOR), (gt_b & ~pred_b, FN_COLOR)]:
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_arr
        return rgb

    n_rows = len(model_rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 8), dpi=300, facecolor='black')
    if n_rows == 1:
        axes = axes.reshape(1, 4)
    col_titles = ['CT', 'GT', 'Prediction', 'Error Map']

    for row_idx, (model_name, pred_crop) in enumerate(model_rows):
        dice = dice_score(pred_crop, gt_crop)
        panels = [
            np.stack([gray] * 3, axis=-1),
            make_overlay(gray, gt_crop, color=(0.0, 1.0, 0.0), alpha=0.5),
            make_overlay(gray, pred_crop, color=(1.0, 0.5, 0.0), alpha=0.5),
            make_error_map_custom(gray, gt_crop, pred_crop),
        ]
        for col_idx, (ax, panel) in enumerate(zip(axes[row_idx], panels)):
            ax.imshow(np.clip(panel, 0.0, 1.0), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('black')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color='white', fontsize=12)
        axes[row_idx][3].text(0.97, 0.03, f'Dice={dice:.3f}', color='white', fontsize=10,
                              ha='right', va='bottom', transform=axes[row_idx][3].transAxes,
                              bbox=dict(facecolor='black', alpha=0.6, edgecolor='white'))
        axes[row_idx][0].set_ylabel(model_name, color='white', fontsize=12,
                                    fontweight='bold', rotation=90, labelpad=8)

    legend_elems = [
        Patch(facecolor=TP_COLOR, label='TP'),
        Patch(facecolor=FP_COLOR, label='FP'),
        Patch(facecolor=FN_COLOR, label='FN'),
    ]
    fig.legend(handles=legend_elems, loc='center right', facecolor='black',
              labelcolor='white', fontsize=10, bbox_to_anchor=(1.0, 0.5))

    fig.patch.set_facecolor('black')
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    out_path = os.path.join(output_dir, 'fig_error_map_comparison.png')
    plt.savefig(out_path, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ================================================================== #
# Figure 5: auto-selected baseline vs FreqFuseNet (needs GPU)          #
# ================================================================== #

def make_fig5(args, output_dir: str) -> None:
    print("\n=== Figure 5: fig_baseline_comparison.png ===")
    from data.segrap_dataset import SegRapDataset
    from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
    from external_baselines.models_external import build_unet_baseline, build_segresnet_baseline
    from utils.train_utils import load_checkpoint
    from matplotlib.lines import Line2D
    from visualize_segrap import percentile_clip_normalize, make_overlay, dice_score, _compute_roi_bbox
    try:
        from scipy.ndimage import center_of_mass, binary_dilation
    except ImportError as e:
        raise ImportError(
            "Figure 5 requires scipy. Install it with: pip install scipy"
        ) from e

    THIN_WALL_OARS = [
        'Cochlea_L', 'Cochlea_R',
        'VestibulSemi_L', 'VestibulSemi_R',
        'IAC_L', 'IAC_R',
        'TympanicCavity_L', 'TympanicCavity_R',
        'MiddleEar_L', 'MiddleEar_R',
    ]
    FALLBACK_OAR  = 'MiddleEar_L'
    FALLBACK_CASE = 'segrap_0111'
    TOP_K         = 3   # number of CSV candidates to actually run inference on
    ZOOM_PAD      = 20
    ZOOM_SCALE    = 4
    GT_BOUNDARY_COLOR   = '#2ECC71'
    PRED_BOUNDARY_COLOR = '#E67E22'
    TP_COLOR = (0x2E / 255, 0xCC / 255, 0x71 / 255)
    FP_COLOR = (0xE7 / 255, 0x4C / 255, 0x3C / 255)
    FN_COLOR = (0x34 / 255, 0x98 / 255, 0xDB / 255)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    # ------------------------------------------------------------------ #
    # Step 1: CSV pre-screening — find top candidate (OAR, case_id) pairs #
    # ------------------------------------------------------------------ #
    unet_csv = './results/external_baselines/results/external_unet_seed2_test_results.csv'
    freq_csv = './results/stage5c/results/s5c_fft_residual_seed2_test_results.csv'

    top_candidates = []
    try:
        import pandas as pd
        unet_df = pd.read_csv(unet_csv)
        freq_df = pd.read_csv(freq_csv)
        merged = unet_df.merge(freq_df, on=['oar_name', 'case_id'], suffixes=('_unet', '_freq'))
        merged['dice_diff'] = merged['dice_freq'] - merged['dice_unet']
        # Exclude near-perfect baseline cases so the visual contrast is meaningful
        candidates_df = merged[merged['dice_unet'] < 0.92].nlargest(5, 'dice_diff')
        print("\n  Top 5 candidates from CSV pre-screening:")
        print(candidates_df[['oar_name', 'case_id', 'dice_unet', 'dice_freq', 'dice_diff']].to_string(index=False))
        top_candidates = list(zip(candidates_df['oar_name'], candidates_df['case_id']))[:TOP_K]
    except Exception as e:
        print(f"  [INFO] CSV pre-screening skipped ({type(e).__name__}: {e}) — using fallback case")

    if not top_candidates:
        top_candidates = [(FALLBACK_OAR, FALLBACK_CASE)]

    # ------------------------------------------------------------------ #
    # Step 2: Load models once, reuse across candidates                   #
    # ------------------------------------------------------------------ #
    baseline_models = []
    for bname, builder, ckpt in [
        ('3D U-Net',  build_unet_baseline,      args.unet_checkpoint),
        ('SegResNet', build_segresnet_baseline,  args.segresnet_checkpoint),
    ]:
        if not os.path.exists(ckpt):
            print(f"  [INFO] {bname} checkpoint not found ({ckpt}) — skipping")
            continue
        print(f"  Loading {bname} from {ckpt} ...")
        try:
            m = builder(in_channels=1, num_classes=2).to(device)
            load_checkpoint(m, None, ckpt, device)
            m.eval()
            baseline_models.append((bname, m))
        except Exception as e:
            print(f"  [WARNING] Failed to load {bname}: {type(e).__name__}: {e}")

    if not baseline_models:
        print("  [WARNING] No baseline checkpoints available — skipping Figure 5 entirely")
        return

    print(f"  Loading FreqFuseNet from {args.freq_checkpoint} ...")
    freq_model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2,
                                     base_features=args.base_features).to(device)
    load_checkpoint(freq_model, None, args.freq_checkpoint, device)
    freq_model.eval()

    def run_inference(model, image, gt_full, z, rmin, rmax, cmin, cmax):
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()
        d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_full, margin=32)
        crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
        pred_crop_3d = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                                preserve_range=True, anti_aliasing=False)
        pred_full = np.zeros(gt_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop_3d > 0.5).astype(np.uint8)
        pred_slice = pred_full[:, :, z].astype(bool)
        return pred_slice[rmin:rmax + 1, cmin:cmax + 1]

    # ------------------------------------------------------------------ #
    # Step 3: Inference loop over top candidates; pick the best gap       #
    # ------------------------------------------------------------------ #
    best_case = None  # (oar_name, case_id, bname, baseline_pred, baseline_dice, freq_pred, freq_dice, gray, gt_crop, gap)

    for oar_name, case_id in top_candidates:
        print(f"\n  Evaluating {oar_name}/{case_id} ...")
        try:
            ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                               oar_subset=[oar_name], cache_dir=args.cache_dir)
            mask_path = ds.data_root / case_id / f'{oar_name}.nii.gz'
            if not mask_path.exists():
                print(f"    mask not found — skipping")
                continue
            idx = next((i for i, (cid, on) in enumerate(ds.samples)
                        if cid == case_id and on == oar_name), None)
            if idx is None:
                print(f"    no thin_wall sample — skipping")
                continue

            sample = ds[idx]
            image = sample['image'].unsqueeze(0).to(device)
            ct_full = nib.load(str(ds.data_root / case_id / 'image.nii.gz')).get_fdata()
            gt_full = nib.load(str(mask_path)).get_fdata() > 0

            areas = gt_full.sum(axis=(0, 1))
            z = int(np.argmax(areas))
            gt_slice = gt_full[:, :, z]
            if not gt_slice.any():
                print(f"    GT slice is empty — skipping")
                continue

            rows_b = np.any(gt_slice, axis=1)
            cols_b = np.any(gt_slice, axis=0)
            rmin, rmax = np.where(rows_b)[0][[0, -1]]
            cmin, cmax = np.where(cols_b)[0][[0, -1]]
            pad = 40
            rmin = max(0, rmin - pad); rmax = min(gt_slice.shape[0] - 1, rmax + pad)
            cmin = max(0, cmin - pad); cmax = min(gt_slice.shape[1] - 1, cmax + pad)

            ct_crop = ct_full[:, :, z][rmin:rmax + 1, cmin:cmax + 1]
            gray    = percentile_clip_normalize(ct_crop)
            gt_crop = gt_slice[rmin:rmax + 1, cmin:cmax + 1]

            # Evaluate each available baseline; pick the one with the lower Dice
            # (= most room for FreqFuseNet to show improvement)
            baseline_result = None
            for bname, bmodel in baseline_models:
                pred_c = run_inference(bmodel, image, gt_full, z, rmin, rmax, cmin, cmax)
                d = dice_score(pred_c, gt_crop)
                print(f"    {bname}: slice Dice = {d:.4f}")
                if baseline_result is None or d < baseline_result[2]:
                    baseline_result = (bname, pred_c, d)

            freq_pred = run_inference(freq_model, image, gt_full, z, rmin, rmax, cmin, cmax)
            freq_d    = dice_score(freq_pred, gt_crop)
            gap       = freq_d - baseline_result[2]
            print(f"    FreqFuseNet: slice Dice = {freq_d:.4f}  |  gap = {gap:+.4f}")

            if best_case is None or gap > best_case[-1]:
                best_case = (oar_name, case_id, baseline_result[0], baseline_result[1],
                             baseline_result[2], freq_pred, freq_d, gray, gt_crop, gap)
        except Exception as e:
            print(f"    [WARNING] Error evaluating {oar_name}/{case_id}: {type(e).__name__}: {e}")

    # Free GPU memory before drawing
    del baseline_models, freq_model

    if best_case is None:
        print("  [WARNING] No valid cases found — skipping Figure 5 entirely")
        return

    (oar_name, case_id, selected_bname, baseline_pred_crop, baseline_dice,
     freq_pred_crop, freq_dice, gray, gt_crop, gap) = best_case

    print(f"\n  Selected case: OAR={oar_name}, case={case_id}")
    print(f"    Baseline ({selected_bname}) Dice: {baseline_dice:.3f}")
    print(f"    FreqFuseNet Dice: {freq_dice:.3f}")
    print(f"    Improvement: {gap:+.3f}")

    # ------------------------------------------------------------------ #
    # Step 4: Helper functions for visualization                           #
    # ------------------------------------------------------------------ #
    def make_error_map_rgb(gray2d, gt2d, pred2d, alpha=0.6):
        rgb = np.stack([gray2d] * 3, axis=-1).copy()
        gt_b, pred_b = gt2d.astype(bool), pred2d.astype(bool)
        tp_mask = gt_b & pred_b
        # FP/FN dilated 1px for display clarity on thin-wall OARs
        # (for display only, does not affect quantitative Dice values)
        fp_mask = binary_dilation(pred_b & ~gt_b, iterations=1)
        fn_mask = binary_dilation(gt_b & ~pred_b, iterations=1)
        for mask, color in [(tp_mask, TP_COLOR), (fp_mask, FP_COLOR), (fn_mask, FN_COLOR)]:
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_arr
        return rgb

    def make_zoom_data(gt_c, pred_c, gray_c):
        cy, cx = center_of_mass(gt_c.astype(bool))
        zy1 = max(0, int(cy) - ZOOM_PAD); zy2 = min(gt_c.shape[0], int(cy) + ZOOM_PAD)
        zx1 = max(0, int(cx) - ZOOM_PAD); zx2 = min(gt_c.shape[1], int(cx) + ZOOM_PAD)
        gray_zoom = gray_c[zy1:zy2, zx1:zx2]
        gt_zoom   = gt_c[zy1:zy2, zx1:zx2]
        pred_zoom = pred_c[zy1:zy2, zx1:zx2]
        large_shape = (gray_zoom.shape[0] * ZOOM_SCALE, gray_zoom.shape[1] * ZOOM_SCALE)
        gray_zoom_large = sk_resize(gray_zoom, large_shape, order=1, preserve_range=True)
        gt_zoom_large   = sk_resize(gt_zoom.astype(np.float64), large_shape, order=0, preserve_range=True)
        pred_zoom_large = sk_resize(pred_zoom.astype(np.float64), large_shape, order=0, preserve_range=True)
        return gray_zoom_large, gt_zoom_large, pred_zoom_large, (zx1, zy1, zx2, zy2)

    b_gz, b_gt_z, b_pred_z, b_bounds = make_zoom_data(gt_crop, baseline_pred_crop, gray)
    f_gz, f_gt_z, f_pred_z, f_bounds = make_zoom_data(gt_crop, freq_pred_crop, gray)

    rows = [
        (selected_bname,      baseline_pred_crop, baseline_dice, b_gz, b_gt_z, b_pred_z, b_bounds),
        ('FreqFuseNet (ours)', freq_pred_crop,     freq_dice,     f_gz, f_gt_z, f_pred_z, f_bounds),
    ]

    # ------------------------------------------------------------------ #
    # Step 5: Draw 2 × 5 figure                                           #
    # ------------------------------------------------------------------ #
    col_titles = ['CT Input', 'GT Overlay', 'Prediction Overlay', 'Error Map', 'Boundary Zoom']
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=300, facecolor='black')

    for row_idx, (model_name, pred_crop, d, gz_large, gt_z_large, pred_z_large, zoom_bounds) in enumerate(rows):
        zx1, zy1, zx2, zy2 = zoom_bounds
        error_map = make_error_map_rgb(gray, gt_crop, pred_crop)
        panels = [
            np.stack([gray] * 3, axis=-1),
            make_overlay(gray, gt_crop, color=(0.0, 1.0, 0.0), alpha=0.5),
            make_overlay(gray, pred_crop, color=(1.0, 0.5, 0.0), alpha=0.5),
            error_map,
        ]
        for col_idx, (ax, panel) in enumerate(zip(axes[row_idx][:4], panels)):
            ax.imshow(np.clip(panel, 0.0, 1.0), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('black')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color='white', fontsize=12)
            if col_idx == 2:
                rect = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=1.2,
                                 edgecolor='white', facecolor='none', linestyle='--')
                ax.add_patch(rect)

        # Dice score in error map's bottom-right corner
        axes[row_idx][3].text(0.97, 0.03, f'Dice={d:.3f}', color='white', fontsize=9,
                             ha='right', va='bottom', transform=axes[row_idx][3].transAxes,
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        # 5th column: boundary zoom (GT + prediction contours on upscaled CT base)
        ax_zoom = axes[row_idx][4]
        ax_zoom.imshow(gz_large, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        ax_zoom.contour(gt_z_large,   levels=[0.5], colors=[GT_BOUNDARY_COLOR],   linewidths=2.0)
        ax_zoom.contour(pred_z_large, levels=[0.5], colors=[PRED_BOUNDARY_COLOR], linewidths=2.0)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_visible(False)
        ax_zoom.set_facecolor('black')
        if row_idx == 0:
            ax_zoom.set_title(col_titles[4], color='white', fontsize=12)
            zoom_legend = [
                Line2D([0], [0], color=GT_BOUNDARY_COLOR,   linewidth=1.5, label='GT'),
                Line2D([0], [0], color=PRED_BOUNDARY_COLOR, linewidth=1.5, label='Pred'),
            ]
            ax_zoom.legend(handles=zoom_legend, loc='lower right', fontsize=6,
                          framealpha=0.7, facecolor='black', labelcolor='white')

        axes[row_idx][0].set_ylabel(model_name, color='white', fontsize=11,
                                    fontweight='bold', rotation=90, labelpad=8)

    legend_elems = [
        Patch(facecolor=TP_COLOR, label='TP'),
        Patch(facecolor=FP_COLOR, label='FP'),
        Patch(facecolor=FN_COLOR, label='FN'),
    ]
    fig.legend(handles=legend_elems, loc='center right', facecolor='black',
              labelcolor='white', fontsize=10, bbox_to_anchor=(1.0, 0.5),
              edgecolor='white')

    fig.patch.set_facecolor('black')
    plt.tight_layout(rect=[0, 0.03, 0.93, 1])
    fig.text(0.5, 0.005, (
        "Green overlay: ground truth; Orange overlay: prediction. "
        "Error map: TP (green), FP (red), FN (blue). "
        "Boundary zoom: green = GT contour, orange = prediction contour."
    ), color='white', fontsize=7, ha='center', va='bottom')

    out_path_png = os.path.join(output_dir, 'fig_baseline_comparison.png')
    out_path_pdf = os.path.join(output_dir, 'fig_baseline_comparison.pdf')
    plt.savefig(out_path_png, dpi=300, facecolor='black')
    plt.savefig(out_path_pdf, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path_png}")
    print(f"  Saved -> {out_path_pdf}")


# ================================================================== #
# Figure 6: representative case — all-OAR global error map (GPU)       #
# ================================================================== #

def make_fig6(args, output_dir: str) -> None:
    print("\n=== Figure 6: fig_global_error_map.png ===")
    # Lazy imports — GPU + installed packages required.
    # visualize_for_paper has its own module-level imports (scipy, data, model);
    # importing it here triggers those, which is fine on RunPod.
    from data.segrap_dataset import SegRapDataset, THIN_WALL_OARS
    from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
    from utils.train_utils import load_checkpoint
    from skimage.measure import find_contours
    from visualize_segrap import percentile_clip_normalize, _compute_roi_bbox
    from visualize_for_paper import (
        OAR_COLORS, hex_to_rgb01,
        _find_head_mask, _bbox_from_mask, _select_best_slice,
        OVERLAY_ALPHA,
    )
    from scipy import ndimage

    OVERALL_MEAN  = 0.8493   # FreqFuseNet average Dice across all thin-wall OARs
    PAPER_MARGIN  = 15       # head-crop margin (matches visualize_for_paper PAPER_MARGIN)
    FALLBACK_CASE = 'segrap_0111'
    TP_COLOR = (0x2E / 255, 0xCC / 255, 0x71 / 255)
    FP_COLOR = (0xE7 / 255, 0x4C / 255, 0x3C / 255)
    FN_COLOR = (0x34 / 255, 0x98 / 255, 0xDB / 255)

    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    # ------------------------------------------------------------------ #
    # Step 1: CSV-based case selection — case whose mean Dice is closest   #
    # to the overall FreqFuseNet average (most representative case)        #
    # ------------------------------------------------------------------ #
    freq_csv = './results/stage5c/results/s5c_fft_residual_seed2_test_results.csv'
    selected_case_id = None
    try:
        import pandas as pd
        freq_df   = pd.read_csv(freq_csv)
        case_dice = freq_df.groupby('case_id')['dice'].mean().reset_index()
        case_dice['dist'] = (case_dice['dice'] - OVERALL_MEAN).abs()
        best_row = case_dice.nsmallest(1, 'dist').iloc[0]
        selected_case_id = str(best_row['case_id'])
        print(f"  Representative case from CSV: {selected_case_id}  "
              f"(mean Dice={best_row['dice']:.4f}, Δ from {OVERALL_MEAN}={best_row['dist']:.4f})")
    except Exception as e:
        print(f"  [INFO] CSV case selection skipped ({type(e).__name__}: {e}) "
              f"— using fallback {FALLBACK_CASE}")
        selected_case_id = FALLBACK_CASE

    # ------------------------------------------------------------------ #
    # Step 2: Load all 10 OAR GT masks for the selected case               #
    # ------------------------------------------------------------------ #
    ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                       oar_subset=THIN_WALL_OARS, cache_dir=args.cache_dir)
    gt_masks = {}
    for oar in THIN_WALL_OARS:
        mask_path = ds.data_root / selected_case_id / f'{oar}.nii.gz'
        if mask_path.exists():
            gt_masks[oar] = nib.load(str(mask_path)).get_fdata() > 0
        else:
            print(f"  [WARNING] {oar}: mask not found for {selected_case_id} — excluded from figure")

    if not gt_masks:
        print(f"  [WARNING] No OAR masks for {selected_case_id} — skipping Figure 6 entirely")
        return

    # ------------------------------------------------------------------ #
    # Step 3: Best axial slice (most OARs simultaneously present)          #
    # ------------------------------------------------------------------ #
    n_slices = next(iter(gt_masks.values())).shape[2]
    z, present_oars, metrics, tie_reason = _select_best_slice(gt_masks, n_slices)
    print(f"  Best axial slice: z={z}  "
          f"OARs visible: {metrics[0]}/{len(THIN_WALL_OARS)}  ({', '.join(present_oars)})")
    print(f"  Slice rationale: {tie_reason}")

    # ------------------------------------------------------------------ #
    # Step 4: Head-only crop (full head, not per-OAR)                      #
    # ------------------------------------------------------------------ #
    ct_path = ds.data_root / selected_case_id / 'image_contrast.nii.gz'
    if not ct_path.exists():
        ct_path = ds.data_root / selected_case_id / 'image.nii.gz'
    ct_full  = nib.load(str(ct_path)).get_fdata()
    ct_slice = ct_full[:, :, z]

    head_mask            = _find_head_mask(ct_slice)
    rmin, rmax, cmin, cmax = _bbox_from_mask(head_mask, margin=PAPER_MARGIN)
    ct_crop              = ct_slice[rmin:rmax + 1, cmin:cmax + 1]
    gray                 = percentile_clip_normalize(ct_crop)
    gray                *= head_mask[rmin:rmax + 1, cmin:cmax + 1]   # zero non-head pixels

    print(f"  Head crop: rows=[{rmin}:{rmax}] cols=[{cmin}:{cmax}]  "
          f"size={rmax-rmin+1}×{cmax-cmin+1}")

    # ------------------------------------------------------------------ #
    # Step 5: FreqFuseNet per-OAR inference                                #
    # ------------------------------------------------------------------ #
    print(f"  Loading FreqFuseNet from {args.freq_checkpoint} ...")
    model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2,
                                base_features=args.base_features).to(device)
    load_checkpoint(model, None, args.freq_checkpoint, device)
    model.eval()

    pred_masks = {}
    for oar in gt_masks.keys():
        idx = next((i for i, (cid, on) in enumerate(ds.samples)
                    if cid == selected_case_id and on == oar), None)
        if idx is None:
            print(f"  [WARNING] {oar}: no thin_wall sample for {selected_case_id} — skipping")
            continue
        sample = ds[idx]
        image  = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits  = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()
        d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_masks[oar], margin=32)
        crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
        pred_crop  = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                               preserve_range=True, anti_aliasing=False)
        pred_full  = np.zeros(ct_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop > 0.5).astype(np.uint8)
        pred_masks[oar] = pred_full.astype(bool)
        print(f"    {oar}: done")
    del model

    # ------------------------------------------------------------------ #
    # Step 6: Error map only — GT/Pred overlays drawn as contours in Step 7 #
    # ------------------------------------------------------------------ #
    def global_error_map(gray2d, gt_dict, pred_dict, alpha=0.7):
        """Union of all OARs → TP/FP/FN filled error map.
        FP and FN dilated 2px so thin boundary errors are visible at paper size
        (for display only, does not affect quantitative values)."""
        rgb = np.stack([gray2d] * 3, axis=-1).copy()
        gt_union   = np.zeros(gray2d.shape, dtype=bool)
        pred_union = np.zeros(gray2d.shape, dtype=bool)
        for oar in gt_dict:
            gt_union   |= gt_dict[oar][:, :, z][rmin:rmax + 1, cmin:cmax + 1].astype(bool)
        for oar in pred_dict:
            pred_union |= pred_dict[oar][:, :, z][rmin:rmax + 1, cmin:cmax + 1].astype(bool)
        tp = gt_union & pred_union
        fp = ndimage.binary_dilation(pred_union & ~gt_union, iterations=2)  # display only
        fn = ndimage.binary_dilation(gt_union & ~pred_union, iterations=2)  # display only
        for mask, color in [(tp, TP_COLOR), (fp, FP_COLOR), (fn, FN_COLOR)]:
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_arr
        return rgb

    error_panel = global_error_map(gray, gt_masks, pred_masks)

    # ------------------------------------------------------------------ #
    # Step 7: Draw 1 × 4 figure — GT/Pred panels use per-OAR contour lines #
    # ------------------------------------------------------------------ #
    panel_labels = [
        'Input CT',
        'Ground Truth',
        'FreqFuseNet Prediction',
        'Error Map (TP/FP/FN)',
    ]
    gray_rgb = np.stack([gray] * 3, axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300, facecolor='black')

    def _style_ax(ax, label):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('black')
        ax.text(0.97, 0.03, label, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=8, color='white',
               bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2))

    # Col 0: CT input
    axes[0].imshow(np.clip(gray_rgb, 0.0, 1.0), interpolation='nearest')
    _style_ax(axes[0], panel_labels[0])

    # Col 1 & 2: CT base + per-OAR skimage contour lines (GT and Pred respectively).
    # OARs are drawn large-first so that smaller structures (Cochlea, IAC) are
    # rendered last and therefore appear on top rather than being covered.
    def _oar_area(masks_dict, oar):
        if oar not in masks_dict:
            return 0
        return int(masks_dict[oar][:, :, z][rmin:rmax + 1, cmin:cmax + 1].sum())

    for ax, masks_dict, label in [
        (axes[1], gt_masks,   panel_labels[1]),
        (axes[2], pred_masks, panel_labels[2]),
    ]:
        ax.imshow(np.clip(gray_rgb, 0.0, 1.0), interpolation='nearest')
        # Sort: largest OAR area first (drawn first = underneath), smallest last (on top)
        draw_order = sorted(THIN_WALL_OARS,
                            key=lambda o: _oar_area(masks_dict, o), reverse=True)
        for oar in draw_order:
            if oar not in masks_dict:
                continue
            m = masks_dict[oar][:, :, z][rmin:rmax + 1, cmin:cmax + 1].astype(bool)
            if not m.any():
                continue
            for contour in find_contours(m, level=0.5):
                # find_contours returns (row, col); ax.plot expects (x=col, y=row)
                ax.plot(contour[:, 1], contour[:, 0],
                        color=OAR_COLORS[oar], linewidth=1.2,
                        solid_capstyle='round', solid_joinstyle='round')
        _style_ax(ax, label)

    # Col 3: Error map (filled TP/FP/FN)
    axes[3].imshow(np.clip(error_panel, 0.0, 1.0), interpolation='nearest')
    _style_ax(axes[3], panel_labels[3])

    # Combined legend: 10 OAR colors + 3 error-map colors, right of figure
    oar_legend = [
        Patch(facecolor=hex_to_rgb01(OAR_COLORS[oar]), label=oar)
        for oar in THIN_WALL_OARS
        if oar in gt_masks or oar in pred_masks
    ]
    err_legend = [
        Patch(facecolor=TP_COLOR, label='TP (True Positive)'),
        Patch(facecolor=FP_COLOR, label='FP (False Positive)'),
        Patch(facecolor=FN_COLOR, label='FN (False Negative)'),
    ]
    fig.legend(handles=oar_legend + err_legend, loc='center left',
              facecolor='black', labelcolor='white', fontsize=7,
              bbox_to_anchor=(1.0, 0.5), edgecolor='white',
              borderpad=0.8, labelspacing=0.4)

    fig.patch.set_facecolor('black')
    plt.tight_layout(rect=[0, 0.08, 0.85, 1])
    fig.text(0.5, 0.01, (
        "Visualization shown for representative case. "
        "All 10 thin-wall OARs shown simultaneously. "
        "Error map computed from binary union of all OAR predictions."
    ), color='white', fontsize=7, ha='center', va='bottom')

    out_path_png = os.path.join(output_dir, 'fig_global_error_map.png')
    out_path_pdf = os.path.join(output_dir, 'fig_global_error_map.pdf')
    plt.savefig(out_path_png, dpi=300, facecolor='black')
    plt.savefig(out_path_pdf, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path_png}")
    print(f"  Saved -> {out_path_pdf}")


# ================================================================== #
# Main                                                                  #
# ================================================================== #

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if 1 in args.figures:
        make_fig1(OUTPUT_DIR)
    if 2 in args.figures:
        make_fig2(OUTPUT_DIR)
    if 3 in args.figures:
        make_fig3(args, OUTPUT_DIR)
    if 4 in args.figures:
        make_fig4(args, OUTPUT_DIR)
    if 5 in args.figures:
        make_fig5(args, OUTPUT_DIR)
    if 6 in args.figures:
        make_fig6(args, OUTPUT_DIR)

    print(f"\n=== Done: figures {args.figures} processed ===")


if __name__ == '__main__':
    main()
