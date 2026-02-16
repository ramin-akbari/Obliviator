from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from numpy import ndarray
from torch import Tensor

import obliviator.schemas as oblvsc

from .user import UserSup, UserUnsup


@dataclass(slots=True, kw_only=True)
class RawData:
    "A DataClass to standardize input data"

    x_train: Tensor | ndarray
    "Learned representation used for training"

    x_test: Tensor | ndarray
    "Learned representation used for testing"

    y: Tensor | ndarray
    "Utility label correponding to train representation"

    y_test: Tensor | ndarray
    "Utility label correponding to test representation"

    s: Tensor | ndarray
    "Unwanted label correponding to train representation"

    s_test: Tensor | ndarray
    "Unwanted label correponding to test representation"


@dataclass(slots=True)
class Experiment:
    """For Reproducing Paper's Experiments"""

    model: Literal["deepseek", "llama", "gpt2", "bert"]
    """Language Model Which We Erase From """
    data: Literal["bios", "dial-men", "dial-sen"]
    """Dataset Used For Erasure """
    mode: Literal["sup", "unsup"]
    """Erasure Mode [sup:Supervised (with y-label) , unsup:Unsupervised] """


@dataclass(slots=True)
class BasicErasureConfig:
    data_adr: Path
    """Directory of Dataset [x:Representations, y:Utility Attribute, s: Unwanted Atrribute]"""

    probing_encoder: oblvsc.MLPConfig
    """Probing Network Configuration - MLP architecture"""

    probing_optimizer: oblvsc.OptimConfig
    """ Probing Network Optimizer"""


@dataclass(slots=True)
class UnsupErasure(BasicErasureConfig):
    """Unsupervised Erasure Config"""

    eraser: oblvsc.UnsupervisedConfig = field(default_factory=UserUnsup)
    """Eraser Config"""


@dataclass(slots=True)
class SupErasure(BasicErasureConfig):
    """Supervised Erasure Config"""

    eraser: oblvsc.SupervisedConfig = field(default_factory=UserSup)
    """Eraser Config"""


InputConfig = Experiment | UnsupErasure | SupErasure
