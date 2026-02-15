from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from numpy import ndarray
from numpy.lib.npyio import NpzFile
from torch import Tensor

import obliviator.schemas as oblvsc

from .user_config import UserSup, UserUnsup


@dataclass
class Experiment:
    """For Reproducing Paper's Experiments"""

    model: Literal["deepseek", "llama", "gpt2", "bert"]
    """Language Model Which We Erase From """
    data: Literal["bios", "dial-men", "dial-sen"]
    """Dataset Used For Erasure """
    mode: Literal["sup", "unsup"]
    """Erasure Mode [sup:Supervised (with y-label) , unsup:Unsupervised] """


@dataclass
class BasicErasureConfig:
    data_adr: Path
    """Directory of Dataset [x:Representations, y:Utility Attribute, s: Unwanted Atrribute]"""

    probing_encoder: oblvsc.MLPConfig
    """Probing Network Configuration - MLP architecture"""

    probing_optimizer: oblvsc.OptimConfig
    """ Probing Network Optimizer"""


@dataclass
class UnsupErasure(BasicErasureConfig):
    """Unsupervised Erasure Config"""

    eraser: oblvsc.UnsupervisedConfig = field(default_factory=UserUnsup)
    """Eraser Config"""


@dataclass
class SupErasure(BasicErasureConfig):
    """Supervised Erasure Config"""

    eraser: oblvsc.SupervisedConfig = field(default_factory=UserSup)
    """Eraser Config"""


InputConfig = Experiment | UnsupErasure | SupErasure
type ErasureData = NpzFile | dict[str, Tensor] | dict[str, ndarray]
