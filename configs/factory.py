import obliviator
from evaluation.probing import MLPCrossEntropy, ProbConfig, ProbData
from obliviator.schemas import (
    SupervisedConfig,
    SupervisedData,
    TermColor,
    UnsupervisedConfig,
    UnsupervisedData,
)
from obliviator.supervised import Supervised
from obliviator.unsupervised import Unsupervised
from obliviator.utils.misc import convert_to_onehot

from .defaults import (
    BaseSup,
    BaseUnsup,
    ClassifierOptim,
    DeepClassifier,
    LargeSup,
    LargeUnsup,
)
from .loader import load_experimental_data, user_loader
from .schemas import Expr, RawData, Sup, Tol, Unsup

type ObliviatorConfig = UnsupervisedConfig | SupervisedConfig
type Eraser = Supervised | Unsupervised


def build_classifier(
    raw_data: RawData, config: ProbConfig, is_adversary: bool
) -> MLPCrossEntropy:
    if is_adversary:
        data = ProbData(
            x=raw_data.x_train,
            x_test=raw_data.x_test,
            y=raw_data.s,
            y_test=raw_data.s_test,
        )
    else:
        data = ProbData(
            x=raw_data.x_train,
            x_test=raw_data.x_test,
            y=raw_data.y,
            y_test=raw_data.y_test,
        )
    return MLPCrossEntropy(data, config)


def build_obliviator(
    raw_data: RawData, config: ObliviatorConfig
) -> obliviator.Supervised | obliviator.Unsupervised:
    match config:
        case SupervisedConfig():
            s_onehot = convert_to_onehot(raw_data.s, is_zero_indexed=True)
            y_onehot = convert_to_onehot(raw_data.y, is_zero_indexed=True)
            data = SupervisedData(
                x=raw_data.x_train, x_test=raw_data.x_test, s=s_onehot, y=y_onehot
            )
            return obliviator.Supervised(data, config)

        case UnsupervisedConfig():
            s_onehot = convert_to_onehot(raw_data.s, is_zero_indexed=True)
            data = UnsupervisedData(
                x=raw_data.x_train, x_test=raw_data.x_test, s=s_onehot
            )
            return obliviator.Unsupervised(data, config)


def experiment_factory(
    exp_config: Expr,
) -> tuple[Eraser, MLPCrossEntropy, MLPCrossEntropy, Tol]:
    data = load_experimental_data(exp_config)

    match exp_config.mode:
        case "sup":
            tol = Tol(dim_reduction=1e-5, evp=1e-5)
            match exp_config.model:
                case "deepseek" | "llama":
                    oblv = LargeSup()
                case "gpt2" | "bert":
                    oblv = BaseSup()

        case "unsup":
            tol = Tol(dim_reduction=5e-4, evp=2e-5)
            match exp_config.model:
                case "deepseek" | "llama":
                    oblv = LargeUnsup()
                case "gpt2" | "bert":
                    oblv = BaseUnsup()

    utility_config = ProbConfig(
        device=exp_config.dev_pb,
        mlp_config=DeepClassifier(),
        optim_config=ClassifierOptim(),
        name="Utility",
        color=TermColor.BRIGHT_GREEN,
    )

    unwanted_config = ProbConfig(
        device=exp_config.dev_pb,
        mlp_config=DeepClassifier(),
        optim_config=ClassifierOptim(),
        name="Unwanted",
        color=TermColor.BRIGHT_RED,
    )

    adversary_cls = build_classifier(data, unwanted_config, is_adversary=True)
    utility_cls = build_classifier(data, utility_config, is_adversary=False)
    eraser = build_obliviator(data, oblv)

    return eraser, adversary_cls, utility_cls, tol


def user_factory(
    cfg: Unsup | Sup,
) -> tuple[Eraser, MLPCrossEntropy, MLPCrossEntropy, Tol]:
    oblv = cfg.eraser
    data = user_loader(cfg.adr)
    eraser = build_obliviator(data, oblv)
    adversary_cls = build_classifier(data, cfg.cls_un, is_adversary=True)
    utility_cls = build_classifier(data, cfg.cls_ut, is_adversary=False)
    tol = Tol(cfg.tol_dim, cfg.tol_evp)
    return eraser, adversary_cls, utility_cls, tol
