import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from typing_extensions import override

from .base import Obliviator
from .schemas import UnsupervisedConfig
from .utils.dataloader import InitDataset
from .utils.linalg import (
    batched_matmul,
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

        self.init_erasure_epochs = config.init_erasure_epochs
        self.init_erasure_steps = config.init_erasure_steps

    @override
    def _init_dim_reduction(self, tol: float) -> None:
        x = self.phi_x(self.x, self.matmul_batch)
        self.x_test = self.phi_x(self.x_test, self.matmul_batch)

        f = null_pca(x, self.s, self.device, self.matmul_batch, rtol=tol)

        self.x = self.eval_function(f, self.x, True)

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

        self.z = self.eval_function(f, w, True)

    def init_erasure(self, tol: float) -> None:
        self._init_dim_reduction(tol)
        # self._init_encoder(self.init_erasure_epochs)
        self._init_evp(tol)

    def _init_encoder(self, data: DataLoader, taus: list[float], epochs: int) -> None:
        pbar = trange(epochs)
        optimizer = self.optim_factory(self.encoder.parameters())
        for _ in pbar:
            for *rvs, s in data:
                optimizer.zero_grad()
                s = s.to(self.device, non_blocking=True)
                rvs = [rv.to(self.device, non_blocking=True) for rv in rvs]
                w = self._loss_embeddings(rvs[0])

                sc_s = torch.cov(s.T).norm("fro").sqrt()
                hs_s = cross_cov(w, s).norm("fro").div(sc_s)

                hs_y = torch.tensor(0.0, device=self.device)
                for tau, rv in zip(taus, rvs):
                    sc = torch.cov(rv.T).norm("fro").sqrt()
                    hs_y = hs_y + cross_cov(w, rv).norm("fro").mul(tau / sc)

                loss = hs_s - hs_y
                loss.backward()
                optimizer.step()
