import obliviator
from obliviator.schemas import (
    MLPConfig,
    OptimConfig,
    SupervisedConfig,
    SupervisedData,
    UnsupervisedConfig,
    UnsupervisedData,
)
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


def obliviator_factory(
    raw_data: RawData, config: ObliviatorConfig
) -> obliviator.Supervised | obliviator.Unsupervised:
    match config:
        case SupervisedConfig():
            s_onehot = convert_to_onehot(raw_data.s, is_zero_indexed=True)
            y_onehot = convert_to_onehot(raw_data.s, is_zero_indexed=True)
            data = SupervisedData(
                x=RawData.x_train, x_test=RawData.x_test, s=s_onehot, y=y_onehot
            )
            return obliviator.Supervised(data, config)

        case UnsupervisedConfig():
            s_onehot = convert_to_onehot(raw_data.s, is_zero_indexed=True)
            data = UnsupervisedData(
                x=RawData.x_train, x_test=RawData.x_test, s=s_onehot
            )
            return obliviator.Unsupervised(data, config)


def experiment_factory(
    exp_config: Experiment,
) -> tuple[
    ObliviatorConfig,
    MLPConfig,
    OptimConfig,
    RawData,
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

    mlp_classifier = DeepClassifier()
    optim_classifier = ClassifierOptim()

    return oblv, mlp_classifier, optim_classifier, data
