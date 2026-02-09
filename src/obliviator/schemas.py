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
class ErasureConfig:
    drff_min: int = 1000
    drff_max: int = 6000

    sigma_min: float = 0.05
    sigma_min_x: float = 0.05
    sigma_min_z: float = 0.05

    resample_x: bool = False
    resample_z: bool = False

    use_rff_s: bool = False
    sigma_min_s: float = 0.1
    drff_label_min: int = 50

    tau_z: float = 0.05
    tau_x: float = 0.01

    evp_tau_z: float = 0.001
    evp_tau_x: float = 0.1

    matmul_batch: int | None = None
    encoder_batch: int = 16384

    device: str = "cpu"
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    encoder_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class UnsupervisedConfig(ErasureConfig):
    init_erasure_epochs: int = 30
    init_erasure_steps: int = 2


@dataclass
class SupervisedConfig(ErasureConfig):
    tau_y: float = 2.5
    evp_tau_y: float = 2.5
    use_rff_y: bool = False
    sigma_y_max: float = 1
    resamle_rff_weights_y: bool = False
