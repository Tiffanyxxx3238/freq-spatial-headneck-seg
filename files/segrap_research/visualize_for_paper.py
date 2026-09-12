"""
visualize_for_paper.py — TMI paper architecture-figure + result-figure
visualizations.

Case selection is AUTOMATIC across the whole test split (segrap_0102 ..
segrap_0119, 18 cases) — NOT fixed to one case. For every test case, the
best representative axial slice is found (see _select_best_slice), and the
case whose best slice has the most OARs simultaneously present (ties broken
by total overlay pixel count, then bilateral L/R completeness, then
proximity to that case's own volume-centre slice) is the one actually used
for output. The search also reports whether the winning slice reaches the
THEORETICAL maximum (10/10 OARs) achievable across ANY case/slice, or
whether no single axial slice anywhere in the test split can show all 10
simultaneously (see the "Slice-representativeness check" log block).

Outputs (./results/paper_viz/):
    01_input_ct.png                       — plain CT, head-only crop+mask
                                             (tight/"architecture" crop), no overlay
    02a_gt_overlay_for_architecture.png    — GT overlay, NO legend, tight
                                             head-filling crop, thicker outline
                                             on each OAR for legibility when
                                             shrunk — for embedding inside a
                                             method architecture figure
    02b_pred_overlay_for_architecture.png  — same, Stage5C predictions
    02a_gt_overlay_for_paper.png           — GT overlay, full legend, more
                                             generous crop — for a results figure
    02b_pred_overlay_for_paper.png         — same, Stage5C predictions

Head-only crop (_find_head_mask / _bbox_from_mask): thresholds the CT,
takes the LARGEST connected component (the head is one large contiguous
mass; the scanner table/headrest frame is a separate, smaller blob with an
air gap), fills internal holes (binary_fill_holes — otherwise the
high-HU-only threshold would leave the skull as a hollow ring and
incorrectly exclude the brain/soft tissue inside it). Pixels outside the
filled head silhouette are zeroed (pure black) AFTER percentile
normalisation (so stray table/strap pixels don't skew the contrast
statistics).

IMPORTANT crop-sizing change from the previous version: the bbox is NO
LONGER forced into a square by padding the shorter side out to match the
longer one. That padding was the dominant source of wasted black
background whenever the head's natural bbox wasn't already square. Instead,
each output figure's figsize is set to match the crop's own aspect ratio
(_aspect_figsize), so the head fills the entire frame with zero
letterboxing AND zero artificial square-padding. The "architecture" crop
uses a small margin (ARCH_MARGIN) for a tight, head-filling inset; the
"paper" crop uses a larger margin (PAPER_MARGIN) for a more conventional
result-figure framing.

Inference still runs PER-OAR on the 128^3 SegRapDataset-preprocessed tensor
(model input pipeline unchanged); each OAR's prediction is mapped back to
original resolution via _compute_roi_bbox() (copied from
visualize_segrap.py — margin=32, IDENTICAL to the formula
SegRapDataset._load_thin_wall_pair itself uses internally, so each OAR's
prediction lands at the exact ROI the model actually saw).

Usage (from segrap_research/, on RunPod):
    python visualize_for_paper.py \\
        --data_root /workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases \\
        --checkpoint ./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth \\
        --cache_dir /workspace/data/cache \\
        --output_dir ./results/paper_viz
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')   # headless rendering, no display on RunPod
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize as sk_resize
import torch

from data.segrap_dataset import SegRapDataset, THIN_WALL_OARS
from stage5_moe_router.model_fft_residual import AFS_DSN_FFTResidual
from utils.train_utils import load_checkpoint
from visualize_segrap import percentile_clip_normalize, _compute_roi_bbox

OAR_COLORS = {
    'Cochlea_L':         '#FF6B6B',  # red
    'Cochlea_R':         '#FF9F43',  # orange
    'VestibulSemi_L':    '#FECA57',  # yellow
    'VestibulSemi_R':    '#48DBFB',  # light blue
    'IAC_L':             '#FF9FF3',  # pink
    'IAC_R':             '#54A0FF',  # blue
    'TympanicCavity_L':  '#5F27CD',  # purple
    'TympanicCavity_R':  '#00D2D3',  # cyan
    'MiddleEar_L':       '#1DD1A1',  # green
    'MiddleEar_R':       '#C8D6E5',  # grey-white
}

OVERLAY_ALPHA  = 0.65   # bumped up again (was 0.45 -> 0.55 -> 0.65): "更清楚、不要太淡"
ARCH_MARGIN    = 5       # tight, head-filling crop for the architecture-figure inset
PAPER_MARGIN   = 15      # more generous crop for the result-figure version
ARCH_OUTLINE_W = 2       # thicker outline so OAR boundaries stay crisp when shrunk
PAPER_OUTLINE_W = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str,
                   default='/workspace/data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases')
    p.add_argument('--checkpoint', type=str,
                   default='./results/stage5c/checkpoints/s5c_fft_residual_seed2/best.pth')
    p.add_argument('--cache_dir', type=str, default='/workspace/data/cache')
    p.add_argument('--output_dir', type=str, default='./results/paper_viz')
    p.add_argument('--base_features', type=int, default=32,
                   help='Must match the checkpoint Stage5C was trained with (default 32).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ------------------------------------------------------------------ #
# Display helpers                                                      #
# ------------------------------------------------------------------ #

def hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _aspect_figsize(base_inches: float, h: int, w: int):
    """
    (width, height) in inches whose aspect ratio matches h:w exactly, with
    the LARGER side equal to base_inches. Used instead of a fixed square
    figsize so the crop's natural aspect ratio fills the whole frame with
    no letterboxing AND no artificial square-padding.
    """
    if h >= w:
        return (base_inches * w / h, base_inches)
    return (base_inches, base_inches * h / w)


def save_full(rgb: np.ndarray, out_path: str, figsize, dpi: int,
             legend_elems=None, legend_fontsize: int = 7) -> None:
    """rgb: (H,W,3) in [0,1]. Full (no crop), title/axis-free, black-bg PNG."""
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation='nearest')
    ax.axis('off')
    ax.set_facecolor('black')
    if legend_elems:
        ax.legend(handles=legend_elems, loc='lower right', framealpha=0.5,
                  facecolor='black', labelcolor='white', fontsize=legend_fontsize)
    fig.patch.set_facecolor('black')
    plt.savefig(out_path, dpi=dpi, facecolor='black')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _select_best_slice(gt_masks: dict, n_slices: int):
    """
    Pick the axis=2 slice that best represents ALL of the OARs in gt_masks:
      1. Maximise the COUNT of OARs with a non-empty mask at that slice.
      2. Tie-break: maximise the TOTAL overlay pixel count summed across
         all OARs at that slice (prefers more visible detail over masks
         that just barely touch the slice).
      3. Tie-break: prefer slices where more bilateral (_L/_R) pairs are
         BOTH simultaneously present (more visually balanced figure).
      4. Final deterministic tie-break: closest to the volume's centre slice.

    Returns (z, present_oars, metrics, tie_reason):
      metrics    = (oar_count, pixel_count, bilateral_count) at z -- used
                   for cross-CASE comparison with the same priority order.
      tie_reason = short human-readable string explaining which rule(s)
                   actually had to fire to break a tie (for logging).
    """
    oar_presence = np.zeros(n_slices, dtype=int)
    oar_pixels   = np.zeros(n_slices, dtype=np.int64)
    for mask in gt_masks.values():
        per_slice_area = mask.sum(axis=(0, 1))
        oar_presence += (per_slice_area > 0).astype(int)
        oar_pixels   += per_slice_area

    bilateral_bases = sorted({
        o[:-2] for o in gt_masks
        if o.endswith('_L') and (o[:-2] + '_R') in gt_masks
    })

    max_presence = int(oar_presence.max())
    candidates = np.where(oar_presence == max_presence)[0]
    tie_reason = f"max OAR-presence count ({max_presence}) -- unique slice" \
                if len(candidates) == 1 else f"max OAR-presence count ({max_presence}), {len(candidates)} slices tied"

    if len(candidates) > 1:
        best_pixels = oar_pixels[candidates].max()
        new_candidates = candidates[oar_pixels[candidates] == best_pixels]
        if len(new_candidates) < len(candidates):
            tie_reason += f" -> tie-broken by total overlay pixel count ({int(best_pixels)} px)"
        candidates = new_candidates

    if len(candidates) > 1:
        bilateral_counts = np.array([
            sum(1 for base in bilateral_bases
                if gt_masks[base + '_L'][:, :, zz].sum() > 0
                and gt_masks[base + '_R'][:, :, zz].sum() > 0)
            for zz in candidates
        ])
        best_bilateral = bilateral_counts.max()
        new_candidates = candidates[bilateral_counts == best_bilateral]
        if len(new_candidates) < len(candidates):
            tie_reason += f" -> tie-broken by bilateral L/R completeness ({int(best_bilateral)} pairs)"
        candidates = new_candidates

    if len(candidates) > 1:
        centre = n_slices / 2.0
        z = int(min(candidates, key=lambda zz: abs(zz - centre)))
        tie_reason += f" -> final tie broken by proximity to volume centre (z={z})"
        candidates = np.array([z])

    z = int(candidates[0])
    present_oars = [oar for oar, mask in gt_masks.items() if mask[:, :, z].sum() > 0]

    bilateral_count_at_z = sum(
        1 for base in bilateral_bases
        if gt_masks[base + '_L'][:, :, z].sum() > 0
        and gt_masks[base + '_R'][:, :, z].sum() > 0
    )
    metrics = (int(oar_presence[z]), int(oar_pixels[z]), bilateral_count_at_z)
    return z, present_oars, metrics, tie_reason


def _find_head_mask(ct_slice: np.ndarray) -> np.ndarray:
    """
    Robust head-only silhouette mask (no bbox/margin yet — see
    _bbox_from_mask for that). A plain intensity threshold can't tell the
    head apart from the scanner table/headrest frame (both are high-HU,
    like bone) -- so threshold first, then take the LARGEST connected
    component: the head is one large contiguous mass, while the table/
    frame is a separate, smaller blob with an air gap between them.

    The thresholded component is a hollow "skull ring" (low-HU brain/soft
    tissue inside the skull falls BELOW the threshold, so it would
    otherwise be wrongly excluded) -- binary_fill_holes() fills that
    interior so the returned mask is a solid head silhouette covering all
    internal anatomy, not just bone.
    """
    threshold = ct_slice.mean()
    fg = ct_slice > threshold * 0.3
    labeled, n_components = ndimage.label(fg)
    if n_components == 0:
        return np.ones_like(fg, dtype=bool)
    sizes = ndimage.sum(fg, labeled, index=range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(labeled == largest_label)


def _bbox_from_mask(mask: np.ndarray, margin: int):
    """
    Tight bounding box around `mask`, padded by `margin`, clipped to the
    array bounds. Deliberately NOT forced square -- the caller sizes its
    output figure to match this bbox's own aspect ratio instead
    (_aspect_figsize), so there is no wasted black padding from
    artificially squaring a non-square head silhouette.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(mask.shape[0] - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(mask.shape[1] - 1, cmax + margin)
    return int(rmin), int(rmax), int(cmin), int(cmax)


def render_overlay(gray: np.ndarray, masks_dict: dict, z: int, crop_bounds,
                   legend_elems, out_path: str, figsize, legend_fontsize: int = 7,
                   outline_width: int = 1) -> None:
    """
    gray: (H,W) in [0,1], already cropped + head-masked.
    masks_dict: {oar_name: (D,H,W) bool array} -- FULL resolution; each
    mask's z-slice is cropped here with the SAME crop_bounds as gray so
    shapes line up.
    crop_bounds: (rmin, rmax, cmin, cmax), inclusive, matching how gray was cropped.
    legend_elems: pass None/[] to omit the legend entirely (architecture-figure version).
    outline_width: erosion depth (px) used to draw a full-opacity outline
        of each OAR's own colour on top of the semi-transparent fill --
        keeps boundaries crisp and identifiable after the image is shrunk
        for an inset/thumbnail. Thicker (e.g. 2) for the architecture
        version, thinner (1) for the larger paper version.
    """
    rmin, rmax, cmin, cmax = crop_bounds
    rgb = np.stack([gray] * 3, axis=-1).copy()
    for oar, mask_3d in masks_dict.items():
        mask_slice = mask_3d[:, :, z][rmin:rmax + 1, cmin:cmax + 1]
        if mask_slice.sum() == 0:
            print(f"    [SKIP] {oar}: empty mask at z={z} within the head crop — not drawn (no error)")
            continue
        color = np.array(hex_to_rgb01(OAR_COLORS[oar]), dtype=np.float32)
        m = mask_slice.astype(bool)
        rgb[m] = (1 - OVERLAY_ALPHA) * rgb[m] + OVERLAY_ALPHA * color
        # Full-opacity outline on the mask's own boundary, for legibility
        # after the figure is shrunk down for an inset/thumbnail.
        eroded = ndimage.binary_erosion(m, iterations=outline_width)
        edge = m & ~eroded
        rgb[edge] = color
    save_full(rgb, out_path, figsize=figsize, dpi=300,
             legend_elems=legend_elems, legend_fontsize=legend_fontsize)


# ------------------------------------------------------------------ #
# Case + slice search across the whole test split                      #
# ------------------------------------------------------------------ #

def find_best_case_and_slice(ds: SegRapDataset):
    """
    Searches every test case (ds.case_ids) for the (case, slice) pair that
    best represents all 10 thin-wall OARs simultaneously, using the same
    priority order as _select_best_slice -- (oar_count, pixel_count,
    bilateral_count) -- now compared ACROSS cases too. Only mask files are
    read during this search (no CT, no model) to keep it cheap.

    Returns (best_case_id, best_z, best_present_oars, best_gt_masks,
             best_tie_reason, best_metrics).
    """
    best_metrics = None
    best_case_id = None
    best_z = None
    best_present_oars = None
    best_gt_masks = None
    best_tie_reason = None

    print(f"Searching {len(ds.case_ids)} test cases for the best (case, slice) pair...")
    for case_id in ds.case_ids:
        gt_masks_case = {}
        for oar in THIN_WALL_OARS:
            mask_path = ds.data_root / case_id / f'{oar}.nii.gz'
            if mask_path.exists():
                gt_masks_case[oar] = nib.load(str(mask_path)).get_fdata() > 0
        if not gt_masks_case:
            print(f"  [WARNING] {case_id}: no thin-wall OAR mask files found — skipping this case")
            continue

        n_slices = next(iter(gt_masks_case.values())).shape[2]
        z, present_oars, metrics, tie_reason = _select_best_slice(gt_masks_case, n_slices)
        print(f"  {case_id}: best slice z={z}  "
              f"OARs={metrics[0]}/{len(gt_masks_case)}  pixels={metrics[1]}  bilateral={metrics[2]}")

        if best_metrics is None or metrics > best_metrics:
            best_metrics = metrics
            best_case_id = case_id
            best_z = z
            best_present_oars = present_oars
            best_gt_masks = gt_masks_case
            best_tie_reason = tie_reason

    if best_case_id is None:
        raise RuntimeError("No test case had any thin-wall OAR mask files — cannot proceed.")

    return best_case_id, best_z, best_present_oars, best_gt_masks, best_tie_reason, best_metrics


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    ds = SegRapDataset(args.data_root, split='test', mode='thin_wall',
                       oar_subset=THIN_WALL_OARS, cache_dir=args.cache_dir)

    # ---- Search all 18 test cases for the best (case, slice) pair ----
    case_id, z, present_oars, gt_masks, tie_reason, metrics = find_best_case_and_slice(ds)
    n_present = metrics[0]
    n_total_oars = len(THIN_WALL_OARS)
    print()
    print(f"Selected case: {case_id}")
    print(f"Selected axial slice: z={z}")
    print(f"  OARs present at this slice ({n_present}/{n_total_oars}): {', '.join(present_oars)}")
    print(f"  Slice-selection rationale: {tie_reason}")

    # ---- Slice-representativeness check (re-confirms this really is the ----
    # ---- best achievable result, not just locally optimal) ----
    print("  Slice-representativeness check:")
    if n_present == n_total_oars:
        print(f"    All {n_total_oars}/{n_total_oars} target OARs are simultaneously visible at this "
              f"slice -- this is the GLOBAL maximum achievable across all "
              f"{len(ds.case_ids)} test cases and every axial slice in each; "
              f"no better single-slice representation exists in this test split.")
    else:
        print(f"    No single axial slice in ANY of the {len(ds.case_ids)} test cases shows more than "
              f"{n_present}/{n_total_oars} OARs simultaneously (confirmed by exhaustively checking the "
              f"best slice of every case). This case/slice is therefore the best achievable "
              f"trade-off, but a fully complete {n_total_oars}-OAR single-slice figure is not possible "
              f"with this dataset's anatomy -- the missing OAR(s) "
              f"({', '.join(sorted(set(THIN_WALL_OARS) - set(present_oars)))}) would need a separate "
              f"slice or a different rendering (e.g. MIP, multi-slice panel) to include.")

    # ---- Original-resolution CT for the winning case ----
    ct_path = ds.data_root / case_id / 'image_contrast.nii.gz'
    if not ct_path.exists():
        print(f"  [WARNING] image_contrast.nii.gz not found for {case_id} — falling back to image.nii.gz")
        ct_path = ds.data_root / case_id / 'image.nii.gz'
    ct_full = nib.load(str(ct_path)).get_fdata()
    ct_slice = ct_full[:, :, z]

    # ---- Head silhouette (computed once) + two crops: tight "architecture" ----
    # ---- inset and a more generous "paper" result-figure framing.        ----
    head_mask = _find_head_mask(ct_slice)

    arch_bounds  = _bbox_from_mask(head_mask, margin=ARCH_MARGIN)
    paper_bounds = _bbox_from_mask(head_mask, margin=PAPER_MARGIN)
    print(f"Head-region crop [architecture, margin={ARCH_MARGIN}]: "
          f"rows=[{arch_bounds[0]}:{arch_bounds[1]}] cols=[{arch_bounds[2]}:{arch_bounds[3]}]")
    print(f"Head-region crop [paper, margin={PAPER_MARGIN}]: "
          f"rows=[{paper_bounds[0]}:{paper_bounds[1]}] cols=[{paper_bounds[2]}:{paper_bounds[3]}]")

    def _masked_gray(bounds):
        rmin, rmax, cmin, cmax = bounds
        ct_crop = ct_slice[rmin:rmax + 1, cmin:cmax + 1]
        g = percentile_clip_normalize(ct_crop)        # normalise on REAL CT values first
        g = g * head_mask[rmin:rmax + 1, cmin:cmax + 1]  # then zero out non-head pixels
        return g

    gray_arch  = _masked_gray(arch_bounds)
    gray_paper = _masked_gray(paper_bounds)

    arch_figsize  = _aspect_figsize(3.0, gray_arch.shape[0],  gray_arch.shape[1])
    paper_figsize = _aspect_figsize(5.0, gray_paper.shape[0], gray_paper.shape[1])

    # ---- 01: plain CT, head-only crop+mask (tight "architecture" framing) ----
    save_full(np.stack([gray_arch] * 3, axis=-1),
             os.path.join(args.output_dir, '01_input_ct.png'),
             figsize=arch_figsize, dpi=300)

    # ---- Model ----
    print(f"Loading Stage5C model from {args.checkpoint} ...")
    model = AFS_DSN_FFTResidual(in_channels=1, num_classes=2, base_features=args.base_features).to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    # ---- Per-OAR inference for the winning case, mapped back to original resolution ----
    pred_masks = {}
    for oar in gt_masks.keys():
        idx = None
        for i, (cid, on) in enumerate(ds.samples):
            if cid == case_id and on == oar:
                idx = i
                break
        if idx is None:
            print(f"  [WARNING] {oar}: no thin_wall sample found for {case_id} in the dataset — skipping prediction")
            continue

        sample = ds[idx]
        image = sample['image'].unsqueeze(0).to(device)   # (1,1,128,128,128)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(image)
            else:
                out = model(image)
            logits = out['output'] if isinstance(out, dict) else out
            pred_128 = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # Map the 128^3 prediction back into the SAME ROI bbox the model's
        # input was cropped+resized from for THIS OAR (margin=32, identical
        # to SegRapDataset._load_thin_wall_pair — see _compute_roi_bbox).
        d0, d1, h0, h1b, w0, w1 = _compute_roi_bbox(gt_masks[oar], margin=32)
        crop_shape = (d1 - d0 + 1, h1b - h0 + 1, w1 - w0 + 1)
        pred_crop = sk_resize(pred_128.astype(np.float32), crop_shape, order=0,
                             preserve_range=True, anti_aliasing=False)
        pred_full = np.zeros(ct_full.shape, dtype=np.uint8)
        pred_full[d0:d1 + 1, h0:h1b + 1, w0:w1 + 1] = (pred_crop > 0.5).astype(np.uint8)
        pred_masks[oar] = pred_full.astype(bool)

    # ---- Legend (full version only; built from OARs present in this case) ----
    legend_elems_full = [Patch(facecolor=hex_to_rgb01(OAR_COLORS[oar]), label=oar)
                         for oar in THIN_WALL_OARS if oar in gt_masks]

    # ---- Version A: for architecture figure -- NO legend, tight head-filling crop ----
    print("02a_gt_overlay_for_architecture.png")
    render_overlay(gray_arch, gt_masks, z, arch_bounds, legend_elems=None,
                   out_path=os.path.join(args.output_dir, '02a_gt_overlay_for_architecture.png'),
                   figsize=arch_figsize, outline_width=ARCH_OUTLINE_W)

    print("02b_pred_overlay_for_architecture.png")
    render_overlay(gray_arch, pred_masks, z, arch_bounds, legend_elems=None,
                   out_path=os.path.join(args.output_dir, '02b_pred_overlay_for_architecture.png'),
                   figsize=arch_figsize, outline_width=ARCH_OUTLINE_W)

    # ---- Version B: for paper/result figure -- full legend, more generous crop ----
    print("02a_gt_overlay_for_paper.png")
    render_overlay(gray_paper, gt_masks, z, paper_bounds, legend_elems=legend_elems_full,
                   out_path=os.path.join(args.output_dir, '02a_gt_overlay_for_paper.png'),
                   figsize=paper_figsize, legend_fontsize=7, outline_width=PAPER_OUTLINE_W)

    print("02b_pred_overlay_for_paper.png")
    render_overlay(gray_paper, pred_masks, z, paper_bounds, legend_elems=legend_elems_full,
                   out_path=os.path.join(args.output_dir, '02b_pred_overlay_for_paper.png'),
                   figsize=paper_figsize, legend_fontsize=7, outline_width=PAPER_OUTLINE_W)

    print("\n=== Done: 5/5 figures generated (1 plain CT + 2 architecture-inset + 2 paper-result) ===")


if __name__ == '__main__':
    main()
