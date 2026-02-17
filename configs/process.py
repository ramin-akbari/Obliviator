from evaluation.probing import MLPCrossEntropy

from .factory import Eraser, classifier_factory, experiment_factory, obliviator_factory
from .loader import user_loader
from .schemas import Experiment, InputConfig


def process_args(
    cfg: InputConfig,
) -> tuple[Eraser, MLPCrossEntropy, MLPCrossEntropy]:
    match cfg:
        case Experiment():
            return experiment_factory(cfg)
        case _:
            oblv = cfg.eraser
            data = user_loader(cfg.data_adr)
            eraser = obliviator_factory(data, oblv)
            adversary_cls = classifier_factory(data, cfg.prob_config, is_adversary=True)
            utility_cls = classifier_factory(data, cfg.prob_config, is_adversary=False)

    return eraser, adversary_cls, utility_cls
