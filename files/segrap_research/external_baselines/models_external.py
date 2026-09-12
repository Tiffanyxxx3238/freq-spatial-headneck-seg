"""
External Baseline Study — model builders.

Provides standard, widely-used 3D medical segmentation architectures as
fair external comparison points for Stage5C, trained on the EXACT SAME
thin-wall OAR binary crop pipeline (same dataset split, same crop/resize,
same metrics, same CSV columns — see external_baselines/README.md for the
full fairness contract). This is the "External Baseline Study", referred
to in any paper text as "Comparison with External Medical Segmentation
Baselines" — NOT Stage6, NOT Stage7.

NAMING DISCIPLINE (important — read before writing it up):
  "3D U-Net baseline" here means MONAI's generic UNet implementation, NOT
  the official nnU-Net framework. nnU-Net (https://github.com/MIC-DKFZ/nnUNet)
  is a full automated pipeline — its own preprocessing, architecture
  search, and training loop — and is a fundamentally different, heavier
  thing than "a U-Net-shaped network trained in our existing loop". Call
  the model in this file "3D U-Net" or "nnU-Net-style 3D U-Net"; never call
  it "nnU-Net" unqualified. See nnunet_feasibility.py for what a TRUE
  nnU-Net comparison would require and why it isn't run here.

Each builder is a thin, lazily-imported wrapper (the external package is
only imported inside the function body, not at module level) so that
`import external_baselines.models_external` always succeeds regardless of
which optional dependencies happen to be installed — only the specific
builder call fails if its package is missing, exactly like Stage 4's
mamba_ssm try/except pattern elsewhere in this project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ['unet', 'segresnet', 'mednext_s']


class _SpatialSizeWrapper(nn.Module):
    """
    Ensures output spatial size exactly matches input spatial size, and
    returns the {'output': ...} dict shape the existing train/eval pipeline
    expects (utils/train_utils.py and utils/metrics.py both already do
    `out['output'] if isinstance(out, dict) else out` — wrapping in a dict
    here means train_one_epoch / validate / evaluate_dataset all work
    completely unmodified, satisfying "use the same train loop" exactly).

    The interpolation is a safety net, not expected to trigger for the
    128^3 crop size actually used (4 strides of 2 evenly divide 128), but
    guards against any other crop size / deep-supervision multi-output
    architecture without silently producing shape-mismatched logits.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> dict:
        target_size = x.shape[2:]
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]   # deep-supervision-style models: take the finest-resolution head
        if tuple(out.shape[2:]) != tuple(target_size):
            out = F.interpolate(out, size=target_size, mode='trilinear', align_corners=False)
        return {'output': out}


def build_unet_baseline(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """
    3D U-Net baseline (MONAI UNet). NOT true nnU-Net — see module docstring.
    """
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )
    return _SpatialSizeWrapper(model)


def build_segresnet_baseline(in_channels: int = 1, num_classes: int = 2,
                             init_filters: int = 32) -> nn.Module:
    """
    MONAI SegResNet baseline. If init_filters=32 turns out too large for
    available GPU memory, the caller may pass init_filters=16 — report this
    explicitly if done (per the project's instructions), don't silently
    downgrade it here.
    """
    from monai.networks.nets import SegResNet
    model = SegResNet(
        spatial_dims=3,
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=num_classes,
        dropout_prob=0.0,
    )
    return _SpatialSizeWrapper(model)


def build_mednext_s_baseline(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """
    MedNeXt-S baseline.

    Package: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
    Pip distribution name: mednextv1 (confusingly different from its
    importable module name, which is nnunet_mednext — verified in this
    environment: `pip show mednextv1` reports Location pointing at the
    installed nnunet_mednext/ package directory).

    Raises ImportError with a clear, actionable message if unavailable;
    callers (smoke_test.py, train_external.py) catch this and report
    gracefully rather than crashing the whole External Baseline Study.
    """
    try:
        from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
            create_mednextv1_small,
        )
    except ImportError as e:
        raise ImportError(
            "MedNeXt-S requires the 'mednextv1' package "
            "(pip install git+https://github.com/MIC-DKFZ/MedNeXt.git ; "
            "importable as `nnunet_mednext`, NOT as `mednextv1` or `mednext`). "
            f"Original error: {e}"
        ) from e

    # ds=False: no deep supervision -- a single full-resolution output head,
    # matching every other baseline's (B, num_classes, D, H, W) interface.
    model = create_mednextv1_small(in_channels, num_classes, kernel_size=3, ds=False)
    return _SpatialSizeWrapper(model)


_BUILDERS = {
    'unet':      build_unet_baseline,
    'segresnet': build_segresnet_baseline,
    'mednext_s': build_mednext_s_baseline,
}


def build_external_model(model_name: str, in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """
    model_name options: 'unet', 'segresnet', 'mednext_s'.
    Raises ValueError for an unknown name, or whatever ImportError the
    underlying builder raises if its package isn't installed.
    """
    if model_name not in _BUILDERS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choices: {list(_BUILDERS.keys())}")
    return _BUILDERS[model_name](in_channels=in_channels, num_classes=num_classes)
