import numpy as np
import torch
from typing_extensions import override

from .base import Obliviator
from .schemas import UnsupervisedConfig
from .utils.linalg import (
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
    def init_dim_reduction(self, tol: float) -> None:
        x = self.phi_x(self.x, self.matmul_batch)
        self.x_test = self.phi_x(self.x_test, self.matmul_batch)

        f = null_pca(x, self.s, self.device, self.matmul_batch, rtol=tol)

        self.x = self.update_and_project(f, self.x, True)

        self.phi_x.change_params(
            d_in=f.shape[1], sigma=median_sigma(self.x, self.sigma_min_x)
        )
        self.update_encoder(f.shape[1], self.encoder_config.hidden_dim)

    def _init_evp(self, tol: float):
        w = self.get_embeddings(self.x, self.encoder_batch)
        self.x_test = self.get_embeddings(self.x_test, self.encoder_batch)

        self.phi.change_params(sigma=median_sigma(w, self.sigma_min))

        w = self.phi(w, self.matmul_batch)
        self.x_test = self.phi(self.x_test, self.matmul_batch)
        x = self.phi_x(self.x, self.matmul_batch)

        f = null_supervised_pca(
            w, (x,), (self.tau_x,), self.s, self.device, self.matmul_batch, rtol=tol
        )

        self.z = self.update_and_project(f, w, True)
