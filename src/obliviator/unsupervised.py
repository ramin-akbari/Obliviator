import torch
from typing_extensions import override

from .base import Obliviator
from .utils.linalg import (
    null_pca,
)


class Unsupervised(Obliviator):
    @override
    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        return null_pca(x, self.s, self.device, self.matmul_batch, rtol=tol)

    @override
    def init_erasure(
        self, tol: float, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_list = [self.x]
        map_list = [self.phi_x]
        tau_list = [self.tau_x]
        self._train_encoder(data_list, map_list, tau_list, epochs)
        z = self._solve_evp(data_list, map_list, tau_list, tol)
        return z, self.x_test
