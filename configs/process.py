from evaluation.probing import MLPCrossEntropy

from .factory import Eraser, experiment_factory, user_factory
from .schemas import Expr, InputConfig, Tol


def process_args(
    cfg: InputConfig,
) -> tuple[Eraser, MLPCrossEntropy, MLPCrossEntropy, Tol]:
    match cfg:
        case Expr():
            return experiment_factory(cfg)
        case _:
            return user_factory(cfg)
