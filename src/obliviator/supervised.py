import numpy as np
import torch

from .schemas import SupervisedConfig
from .unsupervised import Unsupervised
from .utils.kernel import RandomFourierFeature, median_sigma


class Supervised(Unsupervised):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        config: SupervisedConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(x, s, x_test, config, dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
        self.tau_y = config.tau_y
        self.evptau_y = config.evp_tau_y

        if config.use_rff_y:
            phi_y = RandomFourierFeature(
                y.shape[1],
                config.rff_scale_y,
                config.drff_max,
                config.drff_min_y,
                median_sigma(y, config.sigma_min_y),
                False,
                self.device,
            )
            self.y = phi_y(self.y, self.matmul_batch)
