import numpy as np
import torch

from .base import Obliviator


class Supervised(Obliviator):
    def __init__(
        self, x: torch.Tensor | np.ndarray, s: torch.Tensor | np.ndarray
    ) -> None:
        super().__init__(x, s)
