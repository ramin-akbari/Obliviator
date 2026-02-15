from dataclasses import dataclass, field

import torch

from obliviator.schemas import (
    ActivationType,
    MLPConfig,
    OptimConfig,
    SupervisedConfig,
    UnsupervisedConfig,
)

from .data_loader import get_experimental_data
from .schemas import Experiment


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


def select_experiment(
    exp_config: Experiment,
) -> tuple[
    UnsupervisedConfig | SupervisedConfig,
    MLPConfig,
    OptimConfig,
    dict[str, torch.Tensor],
]:

    match exp_config.mode:
        case "sup":
            match exp_config.model:
                case "deepseek" | "llama":
                    oblv = LargeSup()
                case "gpt2" | "bert":
                    oblv = BaseSup()

        case "unsup":
            match exp_config.model:
                case "deepseek" | "llama":
                    oblv = LargeUnsup()
                case "gpt2" | "bert":
                    oblv = BaseUnsup()

    mlp_classifier = DeepClassifier()
    optim_classifier = ClassifierOptim()
    data = get_experimental_data(exp_config)

    return oblv, mlp_classifier, optim_classifier, data
