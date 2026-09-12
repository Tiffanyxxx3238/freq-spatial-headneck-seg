"""
Centralised hyperparameter configuration for all stages.
All stage scripts import from here; override via argparse in each train.py.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    # Absolute path to SegRap2023_Training_Set_120cases directory
    data_root: str = 'data/SegRap2023/Training_Set_120cases/SegRap2023_Training_Set_120cases'
    target_size: Tuple[int, int, int] = (128, 128, 128)
    seed: int = 42
    num_workers: int = 4


@dataclass
class ModelConfig:
    in_channels: int = 1          # ncCT only
    num_classes: int = 2          # binary (thin-wall mode default)
    base_features: int = 32
    use_freq_branch: bool = True
    use_cross_attention: bool = True
    use_router: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 2
    val_batch_size: int = 1
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    amp: bool = True              # mixed precision


@dataclass
class Stage1Config:
    lambda_ffl: float = 0.1
    ffl_alpha: float = 1.0
    ffl_patch_factor: int = 2

    # Ablation sweep
    ablation_lambdas: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)


@dataclass
class Stage2Config:
    reduction: int = 16
    freq_sel_method: str = 'top16'


@dataclass
class Stage3Config:
    use_magnitude: bool = True
    use_phase: bool = True
    target_params_m: float = 30.0  # parameter budget in millions


@dataclass
class Stage4Config:
    d_model: int = 512
    d_state: int = 16
    fusion_type: str = 'gated_mamba'  # 'cross_attention' | 'mamba' | 'gated_mamba'


@dataclass
class Stage5Config:
    n_experts: int = 4
    top_k: int = 2
    expert_types: Tuple[str, ...] = ('low_freq', 'mid_freq', 'high_freq', 'artifact_suppress')


@dataclass
class Stage6Config:
    lambda_domain: float = 0.1
    grl_alpha: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)
    stage6: Stage6Config = field(default_factory=Stage6Config)
    exp_name: str = 'experiment'
    device: str = 'cuda'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'


DEFAULT_CONFIG = Config()
