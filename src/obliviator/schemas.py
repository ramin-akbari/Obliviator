from dataclasses import dataclass, field
from enum import StrEnum, auto


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.001
    batch_size: int = 1024
    use_adaptive_momentum: bool = True
    decoupled_weight_decay: bool = True
    use_nesterov: bool = True
    beta_1: float = 0.85
    beta_2: float = 0.995
    epsilon: float = 1e-8
    momentum_decay: float = 4e-3


class ActivationType(StrEnum):
    RELU = auto()
    GELU = auto()
    SILU = auto()
    TANH = auto()
    SIGMOID = auto()


@dataclass
class MLPConfig:
    input_dim: int = -1
    out_dim: int = -1
    hidden_dim: int = 256
    n_layer: int = 2
    use_projection: bool = True
    activation: ActivationType = ActivationType.SILU


@dataclass
class UnsupervisedConfig:
    drff_min: int = 1000
    drff_max: int = 6000

    rff_scale: int = 5
    rff_scale_x: int = 4
    rff_scale_z: int = 5

    sigma_min: float = 0.05
    sigma_min_x: float = 0.05
    sigma_min_z: float = 0.05

    resample_x: bool = False
    resample_z: bool = False

    smoother_rff_factor: float = 1.5

    use_rff_s: bool = False
    sigma_min_s: float = 0.1
    drff_min_s: int = 50
    rff_scale_s: int = 5

    tau_z: float = 0.05
    tau_x: float = 0.01

    evp_tau_z: float = 0.1
    evp_tau_x: float = 0.1

    matmul_batch: int | None = None

    device: str = "cpu"
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    encoder_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class SupervisedConfig(UnsupervisedConfig):
    tau_y: float = 2.5
    evp_tau_y: float = 2.5
    use_rff_y: bool = False
    sigma_y_max: float = 1
    resamle_rff_weights_y: bool = False
