from dataclasses import dataclass, field

from obliviator.schemas import (
    ActivationType,
    MLPConfig,
    OptimConfig,
    SupervisedConfig,
    UnsupervisedConfig,
)


@dataclass
class DeepClassifier(MLPConfig):
    use_projection: bool = True
    hidden_dim: int = 128
    n_layer: int = 4


@dataclass
class ClassifierOptim(OptimConfig):
    batch_size: int = 4096
    weight_decay: float = 0.01
    lr: float = 5e-3


@dataclass
class EraserEncoder(MLPConfig):
    use_projection: bool = False
    hidden_dim: int = 512
    n_layer: int = 1
    activation: ActivationType = ActivationType.SILU


@dataclass
class EraserOptim(OptimConfig):
    batch_size: int = 32_768
    weight_decay: float = 0.01
    lr: float = 5e-4


@dataclass
class BaseUnsup(UnsupervisedConfig):
    encoder_config: MLPConfig = field(default_factory=EraserEncoder)
    optim_config: OptimConfig = field(default_factory=EraserOptim)


@dataclass
class LargeUnsup(BaseUnsup):
    drff_max: int = 6500
    drff_min: int = 1500
    sigma_min: float = 1.75
    sigma_min_x: float = 3
    sigma_min_z: float = 1.5
    evp_tau_x: float = 0.15


@dataclass
class BaseSup(SupervisedConfig):
    encoder_config: MLPConfig = field(default_factory=EraserEncoder)
    optim_config: OptimConfig = field(default_factory=EraserOptim)


@dataclass
class LargeSup(BaseSup):
    drff_max: int = 6500
    drff_min: int = 1500
    sigma_min: float = 1.75
    sigma_min_x: float = 3
    sigma_min_z: float = 1.5
    evp_tau_x: float = 0.15
    evp_tau_y: float = 2.5
