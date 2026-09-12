"""
visualize_segrap.py — 4-panel CT/GT/prediction/error visualizations for the
10 thin-wall OARs, for paper architecture-figure use.

For each thin-wall OAR:
  1. Among the 18 test cases (segrap_0102..segrap_0119) that have a
     non-empty GT mask for this OAR, pick the one whose mask VOXEL COUNT is
     the median — a "typical" case, not the largest or smallest.
  2. Within that case's 128^3 GT mask, find the axis=2 slice index with the
     largest mask area.
  3. Run Stage5C inference on that sample and render:
       [CT input] [GT overlay] [Stage5C prediction overlay] [Error map]
     as one black-background PNG.

Inference vs. visualisation resolution: model inference always runs on the
128^3 SegRapDataset-preprocessed tensor (unchanged — the model needs a fixed
input size). Visualisation, however, reads the ORIGINAL-resolution NIfTI
files directly (nib.load(...).get_fdata()) for the CT and GT mask, rather
than displaying the 128^3 resized tensors.

Mapping the 128^3 prediction back to original resolution is NOT a plain
`skimage.resize(pred, ct_full.shape)`. SegRapDataset's thin_wall pipeline
crops to the GT mask's bounding box (+margin=32, see
SegRapDataset._load_thin_wall_pair) BEFORE resizing that crop to 128^3 — so
the 128^3 volume represents a small ROI, not the whole CT. Resizing it
directly to the full original shape would stretch that small ROI's content
across the entire volume, putting the rendered prediction in the wrong
place at the wrong scale. Instead, _compute_roi_bbox() below reproduces the
exact same bbox+margin=32 formula, the 128^3 prediction is resized
(order=0, nearest) to that bbox's native crop size, and placed back at the
bbox's original coordinates in a zero array shaped like the full CT.

Usage (from segrap_research/, on RunPod):
    python visualize_segrap.py \\
        --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --checkpoint ./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth \\
        --cache_dir /workspace/data/cache \\
        --output_dir ./results/visualizations
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')   # headless rendering, no display on RunPod
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from skimage.transform import resize as sk_resize
import torch

from data.segrap_dataset import SegRapDataset, THIN_WALL_OARS
from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
from utils.train_utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='/workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--checkpoint', type=str,
                   default='./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth')
    p.add_argument('--cache_dir', type=str, default='/workspace/data/cache')
    p.add_argument('--output_dir', type=str, default='./results/visualizations')
    p.add_argument('--base_features', type=int, default=32,
                   help='Must match the checkpoint Stage5C was trained with (default 32).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ------------------------------------------------------------------ #
# Display helpers                                                      #
# ------------------------------------------------------------------ #

def percentile_clip_normalize(img2d: np.ndarray) -> np.ndarray:
    """p2-p98 percentile clip on this 2D slice, then linearly map to [0,1]."""
    p2, p98 = np.percentile(img2d, [2, 98])
    img_clipped = np.clip(img2d, p2, p98)
    rng = max(p98 - p2, 1e-8)
    return (img_clipped - p2) / rng


def make_overlay(gray2d: np.ndarray, mask2d: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    """gray2d: (H,W) in [0,1]; mask2d: (H,W) boolean-able; color: (r,g,b) in [0,1]."""
    rgb = np.stack([gray2d] * 3, axis=-1).copy()
    m = mask2d.astype(bool)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color_arr
    return rgb


def make_error_map(gray2d: np.ndarray, gt2d: np.ndarray, pred2d: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """TP=green, FP=red, FN=blue, drawn on the grayscale CT background."""
    rgb = np.stack([gray2d] * 3, axis=-1).copy()
    gt_b, pred_b = gt2d.astype(bool), pred2d.astype(bool)
    for mask, color in [
        (gt_b & pred_b,  (0.0, 1.0, 0.0)),   # TP
        (pred_b & ~gt_b, (1.0, 0.0, 0.0)),   # FP
        (gt_b & ~pred_b, (0.0, 0.4, 1.0)),   # FN
    ]:
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_arr
    return rgb


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    return 1.0 if union == 0 else float(2.0 * inter / union)


def _compute_roi_bbox(mask_3d: np.ndarray, margin: int = 32):
    """
    Reproduces SegRapDataset._load_thin_wall_pair's bbox formula exactly:
    tight box around the mask, padded by `margin` voxels, clipped to volume
    bounds. Duplicated here (that method is private/instance-bound) so the
    128^3 prediction can be mapped back to the SAME ROI the model actually
    saw at training/inference time.
    """
    iD, iH, iW = mask_3d.shape
    coords = np.where(mask_3d > 0)
    d0, d1 = int(coords[0].min()), int(coords[0].max())
    h0, h1 = int(coords[1].min()), int(coords[1].max())
    w0, w1 = int(coords[2].min()), int(coords[2].max())
    d0 = max(0, d0 - margin);  d1 = min(iD - 1, d1 + margin)
    h0 = max(0, h0 - margin);  h1 = min(iH - 1, h1 + margin)
    w0 = max(0, w0 - margin);  w1 = min(iW - 1, w1 + margin)
    return d0, d1, h0, h1, w0, w1


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Stage5C model from {args.checkpoint} ...")
    model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2, base_features=args.base_features).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()
    use_amp = device.type == 'cuda'

    n_success = 0
    for oar_name in THIN_WALL_OARS:
        print(f"\n=== {oar_name} ===")

        ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                           oar_subset=[oar_name], cache_dir=args.cache_dir)
        if len(ds) == 0:
            print(f"  [WARNING] no label file found for {oar_name} in any test case — skipping")
            continue

        # Volume (voxel count) of the resized 128^3 GT mask per test case.
        entries = []
        for i in range(len(ds)):
            sample = ds[i]
            vol = int((sample['label'] > 0).sum().item())
            entries.append((i, sample['case_id'], vol))

        present = [(i, cid, v) for i, cid, v in entries if v > 0]
        if not present:
            print(f"  [WARNING] {oar_name}: mask file exists but is empty in every test case — skipping")
            continue

        # "Typical" case = median mask volume among non-empty test cases
        # (for an even count, this picks the upper-median element — a
        # deterministic, reproducible choice; the spec only requires
        # "not the largest, not the smallest", not a strict statistical median).
        present.sort(key=lambda t: t[2])
        idx, case_id, vol = present[len(present) // 2]
        print(f"  picked case={case_id}  mask_voxels={vol}  "
              f"(median of {len(present)} non-empty test cases)")

        sample = ds[idx]
        image = sample['image'].unsqueeze(0).to(device)   # (1, 1, 128, 128, 128) -- model input, unchanged

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()   # (128, 128, 128)

        # ---- From here on, visualisation uses ORIGINAL-resolution NIfTI ----
        # ---- data, not the 128^3 tensors above (see module docstring).  ----
        # Display CT from the contrast-enhanced scan (ceCT); mask + model
        # inference above still use image.nii.gz (ncCT) unchanged, since the
        # model was trained on ncCT and its input cannot be swapped.
        ct_path = ds.data_root / case_id / 'image_contrast.nii.gz'
        if not ct_path.exists():
            print(f"  [WARNING] {oar_name}: {ct_path.name} not found for case {case_id} "
                  f"— falling back to image.nii.gz")
            ct_path = ds.data_root / case_id / 'image.nii.gz'
        mask_path = ds.data_root / case_id / f'{oar_name}.nii.gz'
        ct_full = nib.load(str(ct_path)).get_fdata()             # native shape
        gt_full = nib.load(str(mask_path)).get_fdata() > 0        # native shape, bool

        # Map the 128^3 prediction back into the SAME ROI bbox the model's
        # input was cropped+resized from (NOT a direct resize to ct_full.shape
        # — see module docstring for why that would be wrong).
        d0, d1, h0, h1, w0, w1 = _compute_roi_bbox(gt_full, margin=32)
        crop_shape = (d1 - d0 + 1, h1 - h0 + 1, w1 - w0 + 1)
        pred_crop = sk_resize(pred_128.astype(np.float32), crop_shape,
                             order=0, preserve_range=True, anti_aliasing=False)
        pred_full = np.zeros(ct_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1] = (pred_crop > 0.5).astype(np.uint8)

        # Axial slice (axis=2) with the largest GT mask area, in ORIGINAL resolution.
        areas = gt_full.sum(axis=(0, 1))   # (W,)
        z = int(np.argmax(areas))
        if areas[z] == 0:
            print(f"  [WARNING] {oar_name}: case {case_id} has an empty mask at every axis=2 slice — skipping")
            continue

        # Defensive fallback: z is already guaranteed non-empty by the
        # argmax-of-area selection above, but guard anyway before computing
        # the bounding box below (e.g. if slice-selection logic ever changes).
        if gt_full[:, :, z].sum() == 0:
            for dz in range(1, 10):
                found = False
                for zz in (z - dz, z + dz):
                    if 0 <= zz < gt_full.shape[2] and gt_full[:, :, zz].sum() > 0:
                        z = zz
                        found = True
                        break
                if found:
                    break

        ct_slice   = ct_full[:, :, z]
        gt_slice   = gt_full[:, :, z]
        pred_slice = pred_full[:, :, z]
        gray = percentile_clip_normalize(ct_slice)
        dice = dice_score(pred_slice.astype(bool), gt_slice.astype(bool))

        # Bounding-box crop around the GT mask (+pad=40 -- larger than the
        # old 128^3 version's pad=24, since native resolution is much
        # bigger), clipped to bounds. Dice above is computed on the FULL
        # (uncropped) slice, before this crop, so it matches the real
        # evaluation metric rather than being affected by the crop window.
        rows = np.any(gt_slice, axis=1)
        cols = np.any(gt_slice, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 40
        rmin = max(0, rmin - pad)
        rmax = min(gt_slice.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(gt_slice.shape[1] - 1, cmax + pad)

        gray       = gray[rmin:rmax + 1, cmin:cmax + 1]
        gt_slice   = gt_slice[rmin:rmax + 1, cmin:cmax + 1]
        pred_slice = pred_slice[rmin:rmax + 1, cmin:cmax + 1]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
        panel_titles = ['CT input', 'GT overlay', 'Stage5C prediction overlay', 'Error map']
        panels = [
            np.stack([gray] * 3, axis=-1),
            make_overlay(gray, gt_slice, color=(0.0, 1.0, 0.0)),      # green = GT
            make_overlay(gray, pred_slice, color=(1.0, 0.5, 0.0)),    # orange = prediction
            make_error_map(gray, gt_slice, pred_slice),
        ]

        for ax, panel, title in zip(axes, panels, panel_titles):
            ax.imshow(np.clip(panel, 0.0, 1.0), vmin=0, vmax=1, interpolation='none')
            ax.set_title(title, color='white', fontsize=11)
            ax.axis('off')
            ax.set_facecolor('black')

        legend_elems = [
            Patch(facecolor=(0.0, 1.0, 0.0), label='TP'),
            Patch(facecolor=(1.0, 0.0, 0.0), label='FP'),
            Patch(facecolor=(0.0, 0.4, 1.0), label='FN'),
        ]
        axes[3].legend(handles=legend_elems, loc='lower right', framealpha=0.5,
                      facecolor='black', labelcolor='white', fontsize=8)

        fig.suptitle(f"{oar_name} | {case_id} | slice z={z} | Dice={dice:.3f}",
                    color='white', fontsize=13)
        fig.patch.set_facecolor('black')
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{oar_name}_viz.png")
        plt.savefig(out_path, dpi=300, facecolor='black')
        plt.close(fig)
        print(f"  Saved -> {out_path}")
        n_success += 1

    print(f"\n=== Done: {n_success}/{len(THIN_WALL_OARS)} visualizations generated ===")


if __name__ == '__main__':
    main()
