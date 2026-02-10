from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .schemas import UnsupervisedConfig
from .utils.linalg import RandomFourierFeature, batched_matmul, cross_cov, median_sigma
from .utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8
DRFF_SCALE: int = 4


class Obliviator(ABC):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        config: UnsupervisedConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.x = torch.as_tensor(x, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)
        self.x_test = torch.as_tensor(x_test, dtype=dtype)

        self.sigma_min = config.sigma_min
        self.sigma_min_z = config.sigma_min_z
        self.sigma_min_x = config.sigma_min_x
        self.smooth_sigma_factor = config.smoother_rff_factor

        self.encoder_batch = config.optim_config.batch_size
        self.matmul_batch = config.matmul_batch

        self.tau_x = config.tau_x
        self.tau_z = config.tau_z

        self.encoder_config = config.encoder_config
        self.encoder = torch.nn.Identity()
        self.optim_factory = optim_factory(config.optim_config)
        self.device = torch.device(config.device)
        self.loader = partial(
            DataLoader,
            batch_size=config.optim_config.batch_size,
            num_workers=NUM_THREADS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        self.phi_x = RandomFourierFeature(
            x.shape[1],
            config.rff_scale_x,
            config.drff_max,
            config.drff_min,
            median_sigma(x, config.sigma_min_x),
            config.resample_x,
            self.device,
        )

        self.phi_z = RandomFourierFeature(
            config.encoder_config.out_dim,
            config.rff_scale_z,
            config.drff_max,
            config.drff_min,
            median_sigma(x, config.sigma_min_z),
            config.resample_z,
            self.device,
        )

        self.phi = RandomFourierFeature(
            config.encoder_config.out_dim,
            config.rff_scale,
            config.drff_max,
            config.drff_min,
            median_sigma(x, config.sigma_min),
            False,
            self.device,
        )

        if config.use_rff_s:
            phi_s = RandomFourierFeature(
                s.shape[1],
                config.rff_scale_s,
                config.drff_max,
                config.drff_min_s,
                median_sigma(x, config.sigma_min_s),
                False,
                self.device,
            )
            s = phi_s(self.s, self.matmul_batch)

    def _rff_encoder_embeddings(self, z_batch: torch.Tensor) -> torch.Tensor:
        w = self.encoder(z_batch)
        w = w.div(w.norm(dim=1, keepdim=True))
        self.phi.change_params(
            sigma=median_sigma(w, self.sigma_min, alpha=self.smooth_sigma_factor)
        )
        return self.phi(w)

    @torch.no_grad()
    def get_embeddings(self, z: torch.Tensor, batch: int) -> torch.Tensor:
        def helper(x: torch.Tensor) -> torch.Tensor:
            x = self.encoder(x.to(device=self.device, non_blocking=True))
            return x.div_(x.norm(dim=1, keepdim=True)).cpu()

        return torch.cat(
            [helper(z_batched) for z_batched in torch.split(z, batch, dim=0)],
            dim=0,
        )

    def update_encoder(self, in_dim: int, out_dim: int) -> None:
        self.encoder_config.input_dim = in_dim

        if out_dim != self.encoder_config.out_dim:
            self.phi.change_params(d_in=out_dim)
            self.encoder_config.out_dim = out_dim
            self.encoder_config.hidden_dim = out_dim

        self.encoder = mlp_factory(self.encoder_config)

    def update_interim_rvs(
        self, rv: torch.Tensor, update_x: bool = False, normalize: bool = True
    ):
        z = self.get_embeddings(rv, self.encoder_batch)
        self.x_test = self.get_embeddings(self.x_test, self.encoder_batch)
        self.phi_z.change_params(
            d_in=z.shape[1], sigma=median_sigma(z, self.sigma_min_z)
        )

    def update_and_project(
        self, f: torch.Tensor, x: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        mu = x.mean(dim=0)
        x.sub_(mu)
        x = batched_matmul(x, f, self.matmul_batch, self.device)

        self.x_test.sub_(mu)
        self.x_test = batched_matmul(self.x_test, f, self.matmul_batch, self.device)

        if normalize:
            x.div_(self.x.norm(dim=1, keepdim=True))
            self.x_test.div_(self.x_test.norm(dim=1, keepdim=True))
        return x

    def train_encoder(
        self,
        data_list: list[torch.Tensor],
        map_list: list[RandomFourierFeature],
        taus: list[float],
        epochs: int,
    ) -> None:
        pbar = trange(epochs)
        data = self.loader(TensorDataset(*data_list))
        optimizer = self.optim_factory(self.encoder.parameters())
        for _ in pbar:
            for *rvs, s in data:
                optimizer.zero_grad()
                s = s.to(self.device, non_blocking=True)
                rvs = [rv.to(self.device, non_blocking=True) for rv in rvs]
                w = self._rff_encoder_embeddings(rvs[0])

                hs_s = cross_cov(w, s).square().mean().sqrt()
                hs_y = torch.tensor(0.0, device=self.device)
                for tau, rff, rv in zip(taus, map_list, rvs):
                    hs_y = hs_y + cross_cov(w, rff(rv)).square().mean().sqrt().mul(tau)

                loss = hs_s - hs_y
                loss.backward()
                optimizer.step()

    @abstractmethod
    def init_dim_reduction(self, tol: float) -> None:
        pass

    @abstractmethod
    def init_erasure(self, tol: float, epochs: int) -> None:
        pass

    @abstractmethod
    def solve_evp(self, tol: float) -> None:
        pass
