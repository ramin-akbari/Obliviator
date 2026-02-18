import obliviator
from evaluation.probing import MLPCrossEntropy, ProbConfig, ProbData
from obliviator.schemas import (
    SupervisedConfig,
    SupervisedData,
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
from .loader import load_experimental_data
from .schemas import Experiment, RawData

type ObliviatorConfig = UnsupervisedConfig | SupervisedConfig
type Eraser = Supervised | Unsupervised


def classifier_factory(
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


def obliviator_factory(
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
    exp_config: Experiment,
) -> tuple[
    Eraser,
    MLPCrossEntropy,
    MLPCrossEntropy,
]:
    data = load_experimental_data(exp_config)

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

    cls_config = ProbConfig(
        device=exp_config.probing_device,
        mlp_config=DeepClassifier(),
        optim_config=ClassifierOptim(),
        name="Utility",
    )
    utility_cls = classifier_factory(data, cls_config, is_adversary=False)
    cls_config = ProbConfig(
        device=exp_config.probing_device,
        mlp_config=DeepClassifier(),
        optim_config=ClassifierOptim(),
        name="Unwanted",
    )
    adversary_cls = classifier_factory(data, cls_config, is_adversary=True)

    eraser = obliviator_factory(data, oblv)

    return eraser, adversary_cls, utility_cls
