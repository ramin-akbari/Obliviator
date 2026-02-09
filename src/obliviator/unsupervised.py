import numpy as np
import torch
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

        mu = x.mean(dim=0)
        x.sub_(mu)
        self.x = batched_matmul(x, f, self.matmul_batch, self.device)

        self.x_test.sub_(mu)
        self.x_test = batched_matmul(self.x_test, f, self.matmul_batch, self.device)

        self.x.div_(self.x.norm(dim=1, keepdim=True))
        self.x_test.div_(self.x_test.norm(dim=1, keepdim=True))

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

        mu = w.mean(dim=0)
        w.sub_(mu)
        self.x_test.sub_(mu)

        self.z = batched_matmul(w, f, self.matmul_batch, self.device)
        self.x_test = batched_matmul(self.x_test, f, self.matmul_batch, self.device)

        self.z.div_(self.x.norm(dim=1, keepdim=True))
        self.x_test.div_(self.x_test.norm(dim=1, keepdim=True))

    def init_erasure(self, tol: float) -> None:
        self._init_dim_reduction(tol)
        self._init_encoder(self.init_erasure_epochs)
        self._init_evp(tol)

    def _init_encoder(self, epochs: int) -> None:
        data = self.loader(InitDataset(self.x, self.s))
        pbar = trange(epochs)
        optimizer = self.optim_factory(self.encoder.parameters())
        for _ in pbar:
            for z, s in data:
                optimizer.zero_grad()
                z = z.to(self.device, non_blocking=True)
                s = s.to(self.device, non_blocking=True)
                sc_s = torch.cov(s.T).norm("fro").sqrt()
                sc_z = torch.cov(z.T).norm("fro").sqrt()
                w = self._loss_embeddings(z)
                hs_s = cross_cov(w, s).norm("fro").div_(sc_s)
                hs_z = cross_cov(w, z).norm("fro").div_(sc_z)
                loss = hs_s - hs_z.mul_(self.tau_z)
                loss.backward()
                optimizer.step()
