from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import obliviator.schemas as oblvsc

from .user_defined import UserSupConfig, UserUnsupConfig


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

    eraser: oblvsc.UnsupervisedConfig = field(default_factory=UserUnsupConfig)
    """Eraser Config"""


@dataclass
class SupErasure(BasicErasureConfig):
    """Supervised Erasure Config"""

    eraser: oblvsc.SupervisedConfig = field(default_factory=UserSupConfig)
    """Eraser Config"""


InputConfig = Experiment | UnsupErasure | SupErasure
