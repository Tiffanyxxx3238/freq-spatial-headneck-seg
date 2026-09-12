"""
Shared training utilities: checkpointing, LR schedulers, training loop helpers.
"""
import os
import math
import torch
import torch.optim as optim
from pathlib import Path


# ------------------------------------------------------------------ #
# Checkpointing                                                        #
# ------------------------------------------------------------------ #

def save_checkpoint(model, optimizer, epoch, best_dice, path, extra=None):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # AFS_DSN_V4 checkpoint ≈ 3.3 GB (model + optimizer).  PyTorch's new
    # zipfile serializer has a ZIP-offset overflow bug above ~3 GB.
    # Fix 1: legacy pickle format (_use_new_zipfile_serialization=False).
    # Fix 2 (fallback if the flag was removed in this PyTorch build): split
    #         into <name>_weights.pth + <name>_meta.pth.
    try:
        torch.save(state, path, _use_new_zipfile_serialization=False)
    except TypeError:
        _save_split(state, path)

    print(f"  [ckpt] saved {path}  (epoch={epoch}, dice={best_dice:.4f})")


def _save_split(state: dict, path: str) -> None:
    """Write large checkpoints as two files to avoid ZIP-64 overflow."""
    base = path[:-4] if path.endswith('.pth') else path
    torch.save(state['model'], base + '_weights.pth')
    torch.save({k: v for k, v in state.items() if k != 'model'},
               base + '_meta.pth')


def load_checkpoint(model, optimizer, path, device='cpu'):
    # Support both single-file (legacy) and split (_weights + _meta) formats.
    base = path[:-4] if path.endswith('.pth') else path
    weights_path = base + '_weights.pth'
    meta_path    = base + '_meta.pth'

    if Path(weights_path).exists() and Path(meta_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        meta = torch.load(meta_path, map_location=device)
        if optimizer is not None and 'optimizer' in meta:
            optimizer.load_state_dict(meta['optimizer'])
        epoch     = meta.get('epoch', 0)
        best_dice = meta.get('best_dice', 0.0)
    else:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        epoch     = ckpt.get('epoch', 0)
        best_dice = ckpt.get('best_dice', 0.0)

    print(f"  [ckpt] loaded {path}  (epoch={epoch}, dice={best_dice:.4f})")
    return epoch, best_dice


# ------------------------------------------------------------------ #
# LR schedulers                                                        #
# ------------------------------------------------------------------ #

class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ------------------------------------------------------------------ #
# Training loop helpers                                                #
# ------------------------------------------------------------------ #

def _unpack(batch, device):
    """Return (images, labels) from a dict or tuple batch."""
    if isinstance(batch, dict):
        return batch['image'].to(device), batch['label'].to(device)
    # tuple/list fallback
    return batch[0].to(device), batch[1].to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    epoch=1, total_epochs=1):
    """
    NOTE on NaN/Inf protection (added after a Stage 4 Mamba-fusion incident
    where train_loss went to NaN around epoch 65 and never recovered):

    torch.nn.utils.clip_grad_norm_ alone does NOT prevent a non-finite batch
    from permanently corrupting the model. If any single gradient is NaN/Inf,
    clip_grad_norm_'s computed total_norm is itself NaN/Inf, and the clip
    coefficient (max_norm / (total_norm + eps)) then poisons EVERY parameter's
    .grad with NaN — even ones that were originally finite. In the non-AMP
    branch this NaN-poisoned gradient was previously passed straight to
    optimizer.step(), permanently corrupting the model (irrecoverable: every
    later forward pass starts from the same broken weights, so it diverges
    again immediately, explaining why the NaN never cleared).

    Fix: clip_grad_norm_'s return value (the total norm BEFORE clipping) is
    checked with torch.isfinite() and the optimizer step is skipped entirely
    when it isn't — in both the AMP and non-AMP paths. (GradScaler.step()
    already skips internally when scaler.unscale_() finds inf, but that check
    happens BEFORE our clip_grad_norm_ call, so it doesn't cover poisoning
    introduced by the clip step itself; the explicit check here does.)

    Loss aggregation also now skips non-finite batches, so a handful of bad
    batches no longer permanently pins the displayed/returned train_loss to
    NaN for the rest of the epoch once those batches' optimizer step is
    skipped (the running average instead reflects the still-valid batches).
    """
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    n_valid = 0
    n_skipped = 0
    pbar = tqdm(
        loader,
        desc=f'Epoch {epoch:03d}/{total_epochs}',
        unit='batch',
        dynamic_ncols=True,
        leave=False,
        position=1,
    )
    for i, batch in enumerate(pbar):
        images, labels = _unpack(batch, device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                n_skipped += 1
                tqdm.write(f"  [WARN] epoch {epoch} batch {i}: non-finite grad norm "
                          f"({grad_norm.item()}); skipping optimizer step")
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                n_skipped += 1
                tqdm.write(f"  [WARN] epoch {epoch} batch {i}: non-finite grad norm "
                          f"({grad_norm.item()}); skipping optimizer step")

        loss_val = loss.item()
        if math.isfinite(loss_val):
            total_loss += loss_val
            n_valid += 1
        pbar.set_postfix(loss=f'{total_loss / max(n_valid, 1):.4f}', skipped=n_skipped)

    if n_skipped > 0:
        print(f"  [WARN] epoch {epoch}: skipped {n_skipped}/{len(loader)} batches "
              f"due to non-finite gradients")
    return total_loss / max(n_valid, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    from utils.metrics import batch_dice
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for batch in loader:
        images, labels = _unpack(batch, device)
        out = model(images)
        logits = out['output'] if isinstance(out, dict) else out
        loss = criterion(out, labels)
        total_loss += loss.item()
        total_dice += batch_dice(logits, labels)
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def log_csv(log_path: str, row: dict):
    import csv
    file_exists = Path(log_path).exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
