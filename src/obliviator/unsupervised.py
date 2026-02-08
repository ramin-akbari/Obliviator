import numpy as np
import torch
from tqdm import trange
from typing_extensions import override

from .base import Obliviator
from .schemas import UnsupervisedConfig
from .utils.dataloader import InitDataset
from .utils.linalg import cross_cov, null_pca


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

    def init_erasure(self, tol: float) -> None:
        x = self.phi_x(self.x)
        f = null_pca(x, self.s, tol)
        mu = x.mean(dim=0)
        self.x = x.sub_(mu).mm(f).cpu()
        self.x_test = self.x_test.sub_(mu).mm(f)
        self.update_encoder(f.shape[1], self.encoder_config.hidden_dim)
        self._init_encoder(self.init_erasure_epochs)
        self.z = self.get_embeddings(self.x, self.update_batch)
        self.x_test = self.get_embeddings(self.x_test, self.update_batch)

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
                hs_z = cross_cov(z, s).norm("fro").div_(sc_z)
                loss = hs_z.mul_(self.tau_z) - hs_s
                loss.backward()
                optimizer.step()

    @override
    def solve_evp(self, tol: float) -> None:
        return super().solve_evp(tol)
