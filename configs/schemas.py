from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

from numpy import ndarray
from torch import Tensor

import obliviator.schemas as oblvsc
from evaluation.probing import ProbConfig

from .user import UserProbUnwanted, UserProbUtility, UserSup, UserUnsup


class Tol(NamedTuple):
    dim_reduction: float
    evp: float


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


@dataclass(slots=True, kw_only=True)
class Expr:
    """For Reproducing Paper's Experiments"""

    model: Literal["deepseek", "llama", "gpt2", "bert"]
    """Language Model Which We Erase From """
    data: Literal["bios", "dial-men", "dial-sen"]
    """Dataset Used For Erasure """
    mode: Literal["sup", "unsup"]
    """Erasure Mode [sup:Supervised (with y-label) , unsup:Unsupervised] """
    dev_er: str = "cpu"
    """Obliviator's Device"""
    dev_pb: str = "cpu"
    """Unwanted and Utility Classifier's Device"""


@dataclass(slots=True, kw_only=True)
class BasicErasureConfig:
    adr: Path
    """Directory of Dataset [x:Representations, y:Utility Attribute, s: Unwanted Atrribute]"""

    tol_dim: float = 1e-4
    """EVP tolerance for dimensionality reduction before erasure"""

    tol_evp: float = 1e-5
    """Tolerance for Obliviator's EVP """

    cls_un: ProbConfig = field(default_factory=UserProbUnwanted)
    """Unwanted Attribute Probing Configuration [MLP + Cross-Entropy]"""

    cls_ut: ProbConfig = field(default_factory=UserProbUtility)
    """Utility Attribute Probing Configuration [MLP + Cross-Entropy]"""


@dataclass(slots=True, kw_only=True)
class Unsup(BasicErasureConfig):
    """Unsupervised Erasure Config"""

    eraser: oblvsc.UnsupervisedConfig = field(default_factory=UserUnsup)
    """Eraser Config"""


@dataclass(slots=True, kw_only=True)
class Sup(BasicErasureConfig):
    """Supervised Erasure Config"""

    eraser: oblvsc.SupervisedConfig = field(default_factory=UserSup)
    """Eraser Config"""


InputConfig = Expr | Unsup | Sup
