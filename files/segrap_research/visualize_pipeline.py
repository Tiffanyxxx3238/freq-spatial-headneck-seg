"""
visualize_pipeline.py — square, black-background, title/axis-free PNGs of
each Stage5C pipeline stage, for paper architecture-figure use.

Fixed to one case for a clean, consistent illustration:
    oar_name = 'MiddleEar_L'
    case_id  = 'segrap_0111'   (this OAR's median-volume test case)

Outputs (./results/pipeline_viz/):
    01_input_ct.png            — original-resolution CT (ceCT, fallback ncCT),
                                 GT-mask-cropped (pad=40), percentile-normalised
                                 grayscale
    02_bottleneck_feature.png  — model.freq_branch's INPUT (the bottleneck
                                 feature b, (1,512,8,8,8)): channel-max
                                 projection -> (8,8,8) -> most-active axial
                                 slice -> resize 128x128 (order=1) -> viridis
    03_fft_magnitude.png       — model.freq_branch.fft's OUTPUT, same
                                 treatment, plasma
    04_fcanet_feature.png      — model.freq_branch.fcanet's OUTPUT, same
                                 treatment, inferno
    05_fused_feature.png       — model.freq_branch's OUTPUT (the fused
                                 feature FFTMainResidualFusion.forward
                                 returns as element 0), same treatment, magma
    06_output_overlay.png      — original-resolution CT + prediction mask
                                 overlay (cyan #00FFCC, alpha=0.5), same
                                 slice/crop as 01

Hook targets (verified against stage5_moe_router/fft_residual_fusion.py —
its actual attribute names are `.fft` and `.fcanet`, NOT `.fft_branch` /
`.fca_branch`):
    model.freq_branch              -> forward hook: input[0] = bottleneck b,
                                       output[0] = fused_feature
    model.freq_branch.fft          -> forward hook: output[0] = fft_out
    model.freq_branch.fcanet       -> forward hook: output[0] = fca_out
(`b` and `fused_feature` are captured by ONE hook on model.freq_branch
itself, since both are available in the same forward-hook call.)

Slice selection for 02-05 (continuous feature maps, not binary masks):
"largest area" is generalised from visualize_segrap.py's GT-mask-area logic
(sum over axis=(0,1)) to "largest total |activation|" per axial index of the
8x8x8 channel-max-projected volume. This is computed INDEPENDENTLY for each
of the 4 feature maps (matching the per-image instruction literally), so
they do not necessarily share the same z-index with each other or with the
GT-mask-derived slice used for 01/06. If a single anatomically-consistent
slice across all 6 images is wanted instead, that needs an explicit
bbox -> bottleneck-grid z-index mapping; ask if that's what's actually wanted.

Usage (from segrap_research/, on RunPod):
    python visualize_pipeline.py \\
        --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --checkpoint ./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth \\
        --cache_dir /workspace/data/cache \\
        --output_dir ./results/pipeline_viz
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')   # headless rendering, no display on RunPod
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.transform import resize as sk_resize
import torch

from data.segrap_dataset import SegRapDataset
from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
from utils.train_utils import load_checkpoint
from visualize_segrap import percentile_clip_normalize, _compute_roi_bbox

OAR_NAME = 'MiddleEar_L'
CASE_ID  = 'segrap_0111'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='/workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--checkpoint', type=str,
                   default='./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth')
    p.add_argument('--cache_dir', type=str, default='/workspace/data/cache')
    p.add_argument('--output_dir', type=str, default='./results/pipeline_viz')
    p.add_argument('--base_features', type=int, default=32,
                   help='Must match the checkpoint Stage5C was trained with (default 32).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ------------------------------------------------------------------ #
# Display helpers                                                      #
# ------------------------------------------------------------------ #

def save_square(rgb: np.ndarray, out_path: str) -> None:
    """rgb: (H,W,3) in [0,1]. Square, black-background, title/axis-free PNG."""
    fig = plt.figure(figsize=(3, 3), dpi=200, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation='none')
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    plt.savefig(out_path, dpi=200, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def most_active_slice(feat_3d: np.ndarray) -> int:
    """Generalises GT-mask 'largest area' to a continuous feature map:
    argmax over axis=2 of the total |activation| per slice."""
    scores = np.abs(feat_3d).sum(axis=(0, 1))
    return int(np.argmax(scores))


def channel_max_projection(feat_5d: torch.Tensor) -> np.ndarray:
    """(1, C, 8, 8, 8) -> (8, 8, 8) numpy, max over the channel dim."""
    return torch.max(feat_5d[0], dim=0).values.numpy()


def render_feature_map(feat_8cubed: np.ndarray, cmap_name: str, out_path: str) -> None:
    """feat_8cubed: (8,8,8). Picks the most-active axial slice, resizes to
    128x128 (order=1), applies the named colormap, saves a square PNG."""
    z = most_active_slice(feat_8cubed)
    sl = feat_8cubed[:, :, z]
    sl_norm = percentile_clip_normalize(sl)
    sl_resized = sk_resize(sl_norm, (128, 128), order=1,
                          preserve_range=True, anti_aliasing=False)
    rgb = plt.get_cmap(cmap_name)(np.clip(sl_resized, 0.0, 1.0))[:, :, :3]
    save_square(rgb, out_path)
    print(f"    (most-active axis=2 slice z={z} of 8)")


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    use_amp = device.type == 'cuda'

    print(f"Fixed case: oar_name={OAR_NAME}  case_id={CASE_ID}")
    print(f"Loading Stage5C model from {args.checkpoint} ...")
    model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2, base_features=args.base_features).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    # ---- Locate the fixed sample within the test split ----
    ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                       oar_subset=[OAR_NAME], cache_dir=args.cache_dir)
    idx = None
    for i in range(len(ds)):
        if ds.samples[i][0] == CASE_ID:
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"case_id={CASE_ID} not found in the test split for oar={OAR_NAME}")

    sample = ds[idx]
    image = sample['image'].unsqueeze(0).to(device)   # (1,1,128,128,128) -- model input, 128^3 preprocessed

    # ---- Hooks: bottleneck input/output of freq_branch, fft output, fcanet output ----
    captured = {}

    def hook_freq_branch(module, inp, out):
        captured['bottleneck'] = inp[0].detach().float().cpu()   # b
        captured['fused']      = out[0].detach().float().cpu()    # fused_feature

    def hook_fft(module, inp, out):
        captured['fft_out'] = out[0].detach().float().cpu()

    def hook_fcanet(module, inp, out):
        captured['fca_out'] = out[0].detach().float().cpu()

    hook1 = model.freq_branch.register_forward_hook(hook_freq_branch)
    hook2 = model.freq_branch.fft.register_forward_hook(hook_fft)
    hook3 = model.freq_branch.fcanet.register_forward_hook(hook_fcanet)

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda'):
                out = model(image)
        else:
            out = model(image)
        logits = out['output'] if isinstance(out, dict) else out
        pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()   # (128,128,128)

    hook1.remove()
    hook2.remove()
    hook3.remove()
    print("Hooks captured:", list(captured.keys()))

    # ---- Original-resolution CT/GT for images 01 / 06 ----
    ct_path = ds.data_root / CASE_ID / 'image_contrast.nii.gz'
    if not ct_path.exists():
        print(f"  [WARNING] image_contrast.nii.gz not found for {CASE_ID} — falling back to image.nii.gz")
        ct_path = ds.data_root / CASE_ID / 'image.nii.gz'
    mask_path = ds.data_root / CASE_ID / f'{OAR_NAME}.nii.gz'
    ct_full = nib.load(str(ct_path)).get_fdata()
    gt_full = nib.load(str(mask_path)).get_fdata() > 0

    # Map the 128^3 prediction back into the ROI bbox the model's input was
    # actually cropped+resized from (see visualize_segrap.py for why a
    # direct resize to ct_full.shape would be wrong).
    d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_full, margin=32)
    crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
    pred_crop = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                         preserve_range=True, anti_aliasing=False)
    pred_full = np.zeros(ct_full.shape, dtype=np.uint8)
    pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop > 0.5).astype(np.uint8)

    # Axial slice (axis=2) with the largest GT mask area, original resolution.
    areas = gt_full.sum(axis=(0, 1))
    z = int(np.argmax(areas))

    ct_slice   = ct_full[:, :, z]
    gt_slice   = gt_full[:, :, z]
    pred_slice = pred_full[:, :, z]

    rows = np.any(gt_slice, axis=1)
    cols = np.any(gt_slice, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 40
    rmin = max(0, rmin - pad)
    rmax = min(gt_slice.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(gt_slice.shape[1] - 1, cmax + pad)

    ct_crop   = ct_slice[rmin:rmax + 1, cmin:cmax + 1]
    pred_crop_disp = pred_slice[rmin:rmax + 1, cmin:cmax + 1]
    gray = percentile_clip_normalize(ct_crop)
    print(f"Original-resolution slice z={z}, crop rows=[{rmin}:{rmax}] cols=[{cmin}:{cmax}]")

    # The crop region (rows x cols) is generally NOT square (its aspect
    # ratio follows the GT mask's bbox), so resize explicitly to a square
    # 256x256 here -- otherwise imshow's default aspect='equal' would
    # letterbox (black bars) inside the square 3x3in figure instead of
    # filling it. CT (continuous) uses order=1; the mask (binary) uses
    # order=0 to keep crisp edges instead of blurring them.
    gray_256 = sk_resize(gray, (256, 256), order=1, preserve_range=True, anti_aliasing=True)
    pred_256 = sk_resize(pred_crop_disp.astype(np.float32), (256, 256), order=0,
                        preserve_range=True, anti_aliasing=False) > 0.5

    # ---- 01: input CT ----
    save_square(np.stack([gray_256] * 3, axis=-1), os.path.join(args.output_dir, '01_input_ct.png'))

    # ---- 02-05: feature maps ----
    print("02_bottleneck_feature.png")
    render_feature_map(channel_max_projection(captured['bottleneck']), 'viridis',
                       os.path.join(args.output_dir, '02_bottleneck_feature.png'))

    print("03_fft_magnitude.png")
    render_feature_map(channel_max_projection(captured['fft_out']), 'plasma',
                       os.path.join(args.output_dir, '03_fft_magnitude.png'))

    print("04_fcanet_feature.png")
    render_feature_map(channel_max_projection(captured['fca_out']), 'inferno',
                       os.path.join(args.output_dir, '04_fcanet_feature.png'))

    print("05_fused_feature.png")
    render_feature_map(channel_max_projection(captured['fused']), 'magma',
                       os.path.join(args.output_dir, '05_fused_feature.png'))

    # ---- 06: output overlay (cyan #00FFCC, alpha=0.5) ----
    cyan = np.array([0.0, 1.0, 0.8], dtype=np.float32)   # #00FFCC
    overlay = np.stack([gray_256] * 3, axis=-1).copy()
    m = pred_256
    alpha = 0.5
    overlay[m] = (1 - alpha) * overlay[m] + alpha * cyan
    save_square(overlay, os.path.join(args.output_dir, '06_output_overlay.png'))

    print("\n=== Done: 6/6 pipeline visualizations generated ===")


if __name__ == '__main__':
    main()
