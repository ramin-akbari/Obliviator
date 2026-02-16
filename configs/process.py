from obliviator.schemas import (
    MLPConfig,
    OptimConfig,
    SupervisedConfig,
    UnsupervisedConfig,
)

from .data_loader import user_loader
from .experiment_factory import select_experiment
from .schemas import Experiment, InputConfig, RawData


def process_args(
    cfg: InputConfig,
) -> tuple[UnsupervisedConfig | SupervisedConfig, MLPConfig, OptimConfig, RawData]:

    match cfg:
        case Experiment():
            oblv, mlp_classifier, optim_classifier, data = select_experiment(cfg)
        case _:
            oblv = cfg.eraser
            mlp_classifier = cfg.probing_encoder
            optim_classifier = cfg.probing_optimizer
            data = user_loader(cfg.data_adr)

    return oblv, mlp_classifier, optim_classifier, data
