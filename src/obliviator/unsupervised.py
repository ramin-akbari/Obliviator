from math import tau

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import trange
from typing_extensions import override

from .base import Obliviator
from .schemas import UnsupervisedConfig
from .utils.linalg import (
    RandomFourierFeature,
    cross_cov,
    median_sigma,
    null_pca,
    null_supervised_pca,
)


class Unsupervised(Obliviator):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        config: UnsupervisedConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(x, s, x_test, config, dtype)

    @override
    def null_dim_reduction(self, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.phi_x(self.x, self.matmul_batch).cpu()
        self.x_test = self.phi_x(self.x_test, self.matmul_batch).cpu()

        f = null_pca(x, self.s, self.device, self.matmul_batch, rtol=tol)

        self.x = self.update_and_project(f, x, normalize=True)

        self.phi_x.change_params(
            d_in=f.shape[1], sigma=median_sigma(self.x, self.sigma_min_x)
        )
        self.update_encoder(f.shape[1], self.encoder_config.hidden_dim)
        return self.x, self.x_test

    def _init_evp(self, tol: float) -> torch.Tensor:
        w = self.get_embeddings(self.x, self.encoder_batch)
        self.x_test = self.get_embeddings(self.x_test, self.encoder_batch)

        self.phi.change_params(sigma=median_sigma(w, self.sigma_min))

        w = self.phi(w, self.matmul_batch).cpu()
        self.x_test = self.phi(self.x_test, self.matmul_batch).cpu()
        x = self.phi_x(self.x, self.matmul_batch).cpu()

        f = null_supervised_pca(
            w, (x,), (self.tau_x,), self.s, self.device, self.matmul_batch, rtol=tol
        )

        return self.update_and_project(f, w, True)

    def init_erasure(
        self, tol: float, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().init_erasure(tol, epochs)

    def _train_encoder(
        self,
        data_list: list[torch.Tensor],
        map_list: list[RandomFourierFeature],
        taus: list[float],
        epochs: int,
    ) -> None:
        z = data_list[0]
        cached, cached_taus, not_cached, taus, map_list = self._cache_rff(
            data_list, map_list, taus
        )
        n_cached = len(cached)
        data_list = cached + not_cached
        data_list.append(z)
        data_list.append(self.s)

        self._train(data_list, map_list, cached_taus, taus, n_cached, epochs)
