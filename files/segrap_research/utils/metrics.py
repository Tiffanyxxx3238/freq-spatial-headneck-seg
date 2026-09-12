"""
Evaluation metrics for 3-D medical image segmentation.

Dependencies: torch, numpy, scipy, pandas
"""
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from data.segrap_dataset import THIN_WALL_OARS


# ------------------------------------------------------------------ #
# Core binary metrics (numpy arrays, values in {0, 1})               #
# ------------------------------------------------------------------ #

def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0  # both empty → perfect score
    return float((2.0 * inter + smooth) / (union + smooth))


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    if union == 0:
        return 1.0
    return float((inter + smooth) / (union + smooth))


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    verbose: bool = False,
) -> float:
    """95th-percentile Hausdorff distance in mm. Pure scipy/numpy (no SimpleITK
    dependency — a prior version silently returned NaN for every case whenever
    SimpleITK failed to import, since that path was gated by `_SITK_AVAILABLE`).
    Returns inf if either mask, or its extracted surface, is empty."""
    from scipy.ndimage import distance_transform_edt

    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)

    if pred_b.sum() == 0 or tgt_b.sum() == 0:
        if verbose:
            print(f"[compute_hd95] empty mask -> inf  "
                  f"(pred.sum()={int(pred_b.sum())}, target.sum()={int(tgt_b.sum())})")
        return float('inf')

    pred_border = _get_border(pred_b)
    tgt_border = _get_border(tgt_b)
    if pred_border.sum() == 0 or tgt_border.sum() == 0:
        if verbose:
            print(f"[compute_hd95] empty surface -> inf  "
                  f"(pred.sum()={int(pred_b.sum())}, target.sum()={int(tgt_b.sum())}, "
                  f"surf_pred.sum()={int(pred_border.sum())}, surf_target.sum()={int(tgt_border.sum())})")
        return float('inf')

    dt_pred = distance_transform_edt(~pred_b, sampling=spacing)
    dt_tgt = distance_transform_edt(~tgt_b, sampling=spacing)
    d_pred_to_tgt = dt_tgt[pred_border > 0]
    d_tgt_to_pred = dt_pred[tgt_border > 0]
    all_d = np.concatenate([d_pred_to_tgt, d_tgt_to_pred])
    result = float(np.percentile(all_d, 95))
    if verbose and np.isnan(result):
        print(f"[compute_hd95] percentile produced NaN  "
              f"(all_d.size={all_d.size}, pred.sum()={int(pred_b.sum())}, target.sum()={int(tgt_b.sum())})")
    return result


def compute_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance_mm: float = 1.0,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """Normalised surface Dice at a given tolerance in mm."""
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    from scipy.ndimage import distance_transform_edt
    pred_border = _get_border(pred)
    tgt_border = _get_border(target)

    dt_pred = distance_transform_edt(~pred.astype(bool), sampling=spacing)
    dt_tgt = distance_transform_edt(~target.astype(bool), sampling=spacing)

    pred_on_tgt = (dt_tgt[pred_border > 0] <= tolerance_mm).sum()
    tgt_on_pred = (dt_pred[tgt_border > 0] <= tolerance_mm).sum()

    denom = pred_border.sum() + tgt_border.sum()
    if denom == 0:
        return 1.0
    return float((pred_on_tgt + tgt_on_pred) / denom)


def _get_border(mask: np.ndarray) -> np.ndarray:
    """Return binary border voxels of a 3-D binary mask."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask.astype(bool))
    return (mask.astype(bool) & ~eroded).astype(np.uint8)


# ------------------------------------------------------------------ #
# Per-case evaluation                                                  #
# ------------------------------------------------------------------ #

def evaluate_case(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    oar_name: str = '',
    verbose: bool = False,
) -> dict:
    """Compute all metrics for a single binary (pred, target) pair."""
    pred_b = (pred > 0.5).astype(np.uint8)
    tgt_b = (target > 0.5).astype(np.uint8)
    result = {
        'oar_name': oar_name,
        'is_thin_wall': oar_name in THIN_WALL_OARS,
        'dice': compute_dice(pred_b, tgt_b),
        'iou': compute_iou(pred_b, tgt_b),
        'hd95': compute_hd95(pred_b, tgt_b, spacing, verbose=verbose),
        'surface_dice_1mm': compute_surface_dice(pred_b, tgt_b, 1.0, spacing),
        'surface_dice_2mm': compute_surface_dice(pred_b, tgt_b, 2.0, spacing),
    }
    return result


# ------------------------------------------------------------------ #
# Dataset-level evaluation                                             #
# ------------------------------------------------------------------ #

def evaluate_dataset(
    model,
    dataloader,
    device,
    spacing: tuple = (1.0, 1.0, 1.0),
    output_csv: str = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run model over the full dataloader, compute per-case metrics,
    and return a DataFrame with per-OAR results + thin-wall subset summary.

    `spacing` is only a fallback for batches that don't carry a 'spacing'
    field. When the dataset returns one (SegRapDataset does, as the
    effective mm/voxel of the resized ROI — native CT spacing is wrong
    once the crop has been resampled to target_size), it's used per-sample.
    """
    model.eval()
    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if isinstance(batch, dict):
                images        = batch['image'].to(device)
                labels        = batch['label']
                case_ids      = batch['case_id']
                oar_names     = batch['oar_name']
                batch_spacing = batch.get('spacing', None)  # collated: [tensor(B), tensor(B), tensor(B)]
            else:
                images, labels, case_ids, oar_names = batch[0].to(device), batch[1], batch[2], batch[3]
                batch_spacing = None
            out = model(images)
            logits = out['output'] if isinstance(out, dict) else out
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for b in range(images.shape[0]):
                pred_b = (preds[b] > 0).astype(np.uint8)
                tgt_b = (labels_np[b] > 0).astype(np.uint8)
                if batch_spacing is not None:
                    sample_spacing = tuple(float(batch_spacing[i][b]) for i in range(3))
                else:
                    sample_spacing = spacing
                rec = evaluate_case(pred_b, tgt_b, sample_spacing, oar_names[b], verbose=verbose)
                rec['case_id'] = case_ids[b]
                records.append(rec)

    df = pd.DataFrame(records)

    # Summary rows
    overall = df[['dice', 'iou', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']].mean()
    thin_df = df[df['is_thin_wall']]
    thin = thin_df[['dice', 'iou', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']].mean()

    print("\n=== Overall ===")
    print(overall.to_string())
    print("\n=== Thin-wall OAR subset ===")
    print(thin.to_string())

    per_oar = df.groupby('oar_name')[['dice', 'iou', 'hd95', 'surface_dice_1mm', 'surface_dice_2mm']].mean()
    print("\n=== Per-OAR ===")
    print(per_oar.to_string())

    if output_csv:
        df.to_csv(output_csv, index=False)
        per_oar.to_csv(output_csv.replace('.csv', '_per_oar.csv'))
        print(f"Saved to {output_csv}")

    return df


# ------------------------------------------------------------------ #
# Batch dice for training monitoring (torch tensors)                  #
# ------------------------------------------------------------------ #

def batch_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    preds = torch.argmax(logits, dim=1)
    scores = []
    for c in range(1, num_classes):
        pc = (preds == c).float()
        tc = (targets == c).float()
        inter = (pc * tc).sum()
        union = pc.sum() + tc.sum()
        scores.append(float((2.0 * inter) / (union + 1e-8)) if union > 0 else 1.0)
    return float(np.mean(scores)) if scores else 0.0
