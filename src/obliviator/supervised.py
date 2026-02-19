from typing import override

import torch

from .schemas import SupervisedConfig, SupervisedData
from .unsupervised import DataSplit, Unsupervised
from .utils.kernel import RandomFourierFeature, median_sigma
from .utils.linalg import null_supervised_pca


class Supervised(Unsupervised):
    def __init__(
        self,
        data: SupervisedData,
        config: SupervisedConfig,
    ) -> None:
        super().__init__(data, config)
        self.y = data.y
        self.tau_y = config.tau_y
        self.evptau_y = config.evp_tau_y
        if config.use_rff_y:
            phi_y = RandomFourierFeature(
                self.y.shape[1],
                config.rff_scale_y,
                config.drff_max,
                config.drff_min_y,
                median_sigma(self.y, config.sigma_min_y, alpha=0.9),
                False,
                self.device,
            )
            self.y = phi_y(self.y, self.mm_batch)

    @override
    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        data_list = [x, self.y]
        tau_list = [self.evptau_x, self.evptau_y]
        return null_supervised_pca(
            x, data_list, tau_list, self.s, self.device, self.mm_batch, rtol=tol
        )

    def _cache_rff(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        evptau_list: list[float],
    ) -> DataSplit:
        data = super()._cache_rff(data_list, phi_list, tau_list, evptau_list)
        data.static_features.append(self.y)
        data.static_evptaus.append(self.evptau_y)
        data.static_taus.append(self.tau_y)
        return data
