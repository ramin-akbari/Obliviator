from abc import ABC

import numpy as np
import torch


class Obliviator(ABC):
    def __init__(
        self, x: torch.Tensor | np.ndarray, s: torch.Tensor | np.ndarray
    ) -> None:
        pass
