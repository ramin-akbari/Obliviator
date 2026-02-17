from dataclasses import dataclass, field
from enum import StrEnum, auto

import numpy as np
import torch


@dataclass(slots=True, kw_only=True)
class OptimConfig:
    """Optimizer Configuration [Adam,AdamN,AdamW,NAdamW,SGD]"""

    lr: float = 1e-3
    """Learning Rate"""
    weight_decay: float = 0.001
    """Weight Decay or Regularizer"""
    batch_size: int = 1024
    """Batch Size"""
    use_adaptive_momentum: bool = True
    """Switches to Adam Variants"""
    decoupled_weight_decay: bool = True
    """Switches to AdamW Variants"""
    use_nesterov: bool = True
    """- With Adam is Equivalent to NAdamW
       - With SGD is Equivalent to Nesterov Momentum"""
    beta_1: float = 0.9
    """Momentum Coefficient"""
    beta_2: float = 0.999
    """Squared Momentum Coefficient"""
    epsilon: float = 1e-8
    """Stablizer"""
    momentum_decay: float = 4e-3
    """Decaying momentum in NAdam or NAdamW"""


class ActivationType(StrEnum):
    RELU = auto()
    GELU = auto()
    SILU = auto()
    TANH = auto()
    SIGMOID = auto()


@dataclass(slots=True, kw_only=True)
class MLPConfig:
    """MLP Configuration used in Eraser-Encoder or Probing-Encoder"""

    input_dim: int = -1
    """Input Dim, -1 means will be determined from data"""
    out_dim: int = -1
    """Outpu Dim, -1 means will be determined from data"""
    hidden_dim: int = 256
    """Hidden Dimension"""
    n_layer: int = 2
    """Number of Hidden Layer"""
    use_projection: bool = True
    """Projection After Last Layer - Disabled in Eraser-Encoder [Obtained From EVP]"""
    activation: ActivationType = ActivationType.SILU
    """Activation Function """


@dataclass(slots=True, kw_only=True)
class UnsupervisedConfig:
    """Unsupervised Erasure Config. Erasure Process:  x->....->z->Encoder->w"""

    drff_min: int = 1000
    """min number of Random Fourier Features used to estimate HSIC and solve EVP """
    drff_max: int = 4000
    """max number of Random Fourier Features used to estimate HSIC and solve EVP """

    rff_scale: int = 6
    """Scale Factor for Random Fourier Features for Current RV [w]"""
    rff_scale_x: int = 4
    """Scale Factor for Random Fourier Features for Initial RV [x]"""
    rff_scale_z: int = 8
    """Scale Factor for Random Fourier Features for Previous RV [z]"""

    sigma_min: float = 1.5
    """Min value for sigma based on Median Heuristic for RV [w]"""
    sigma_min_x: float = 2.5
    """Min value for sigma based on Median Heuristic for RV [x]"""
    sigma_min_z: float = 1
    """Min value for sigma based on Median Heuristic for RV [z]"""

    resample_x: bool = False
    """Resampling Random Fourier Features After each epoch [use if x requires high dimensional RFF]"""
    resample_z: bool = False
    """Resampling Random Fourier Features After each epoch [use if z requires high dimensional RFF]"""

    smoother_rff_factor: float = 1.5
    """Make sigma [w] smoother during encoder training"""

    use_rff_s: bool = False
    """Use RFF for S, if S is binary not required"""
    sigma_min_s: float = 0.1
    """Min value for sigma based on Median Heuristic for RV [s]"""
    drff_min_s: int = 100
    """min number of Random Fourier Features for RV [s]"""
    rff_scale_s: int = 15
    """Scale Factor for Random Fourier Features for RV [s]"""

    tau_z: float = 0.05
    """coefficient of HSIC(z,w)"""
    tau_x: float = 0.01
    """coefficient of HSIC(x,w)"""

    evp_tau_z: float = 1
    """coefficient of Cov(z,w) in EVP"""
    evp_tau_x: float = 0.2
    """coefficient of Cov(x,w) in EVP"""

    matmul_batch: int | None = None
    """Batch size dimension for matrix-multiplication [used for covariance calculation and EVP]"""

    device: str = "cpu"
    """ Device : [cpu] or cuda or [cuda:0(device id)]"""
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    encoder_config: MLPConfig = field(default_factory=MLPConfig)


@dataclass(slots=True, kw_only=True)
class SupervisedConfig(UnsupervisedConfig):
    tau_y: float = 2
    """coefficient of HSIC(y,w)"""
    evp_tau_y: float = 2
    """coefficient of Cov(y,w) in EVP"""
    use_rff_y: bool = True
    """Use RFF for Y, if Y is binary is not required"""
    sigma_min_y: float = 1
    """Min value for sigma based on Median Heuristic for RV [y]"""
    drff_min_y: int = 100
    """min number of Random Fourier Features for RV [y]"""
    rff_scale_y: int = 15
    """Scale Factor for Random Fourier Features for RV [y]"""


@dataclass(slots=True)
class UnsupervisedData:
    x: torch.Tensor
    s: torch.Tensor
    x_test: torch.Tensor

    def __init__(
        self,
        *,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.x_test = torch.as_tensor(x_test, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)


@dataclass(slots=True)
class SupervisedData(UnsupervisedData):
    y: torch.Tensor

    def __init__(
        self,
        *,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ):
        UnsupervisedData.__init__(self, x=x, s=s, x_test=x_test, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
