import numpy as np
import torch

from .base import Obliviator
from .schemas import SupervisedConfig


class Supervised(Obliviator):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        encoder: torch.nn.Module,
        config: SupervisedConfig,
        device: torch.device,
    ) -> None:
        super().__init__(x, s, x_test, encoder, config, device)
