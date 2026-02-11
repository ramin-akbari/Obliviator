from typing import override

import numpy as np
import torch

from .schemas import SupervisedConfig
from .unsupervised import Unsupervised
from .utils.kernel import RandomFourierFeature, median_sigma
from .utils.linalg import null_supervised_pca


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

    @override
    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        data_list = [x, self.y]
        tau_list = [self.tau_x, self.tau_y]
        return null_supervised_pca(
            x, data_list, tau_list, self.s, self.device, self.matmul_batch, rtol=tol
        )

    @override
    def _cache_rff(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
    ) -> tuple[
        list[torch.Tensor],
        list[float],
        list[torch.Tensor],
        list[float],
        list[RandomFourierFeature],
    ]:
        # we just need to add y to the cached RVs
        cached, cached_taus, not_cached, not_cached_taus, not_cached_phi = (
            super()._cache_rff(data_list, phi_list, tau_list)
        )
        cached.append(self.y)
        return cached, cached_taus, not_cached, not_cached_taus, not_cached_phi
