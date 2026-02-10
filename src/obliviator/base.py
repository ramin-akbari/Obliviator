from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .schemas import UnsupervisedConfig
from .utils.kernel import RandomFourierFeature, median_sigma
from .utils.linalg import batched_matmul, cross_cov
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
            return x.div_(x.norm(dim=1, keepdim=True)).to(device=x.device)

        return torch.cat(
            [helper(z_batched) for z_batched in torch.split(z, batch, dim=0)],
            dim=0,
        )

    # update encoder for iterative erasure
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

    # equivalent to f(x) - mu_f
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

    def _train(
        self,
        data_list: list[torch.Tensor],
        map_list: list[RandomFourierFeature],
        cached_taus: list[float],
        taus: list[float],
        n_cached: int,
        epochs: int,
    ):
        data = self.loader(TensorDataset(*data_list))
        optimizer = self.optim_factory(self.encoder.parameters())
        pbar = trange(epochs)

        for _ in pbar:
            for *rvs, z, s in data:
                optimizer.zero_grad()
                s = s.to(self.device, non_blocking=True)
                z = z.to(self.device, non_blocking=True)
                rvs = [rv.to(self.device, non_blocking=True) for rv in rvs]
                w = self._rff_encoder_embeddings(z)

                hs_s = cross_cov(w, s).square().mean().sqrt()

                hs_p = torch.tensor(0.0, device=self.device)

                # HSIC for cached inputs, we normalize HSIC based on the dimension of RVs
                for tau, rv in zip(cached_taus, rvs[:n_cached]):
                    hs_p = hs_p + cross_cov(w, rv).square().mean().sqrt().mul(tau)

                # HSIC for not_cached inputs
                for tau, rff, rv in zip(taus, map_list, rvs[n_cached:]):
                    hs_p = hs_p + cross_cov(w, rff(rv)).square().mean().sqrt().mul(tau)

                loss = hs_s - hs_p
                loss.backward()
                optimizer.step()

            # resample RFF weights
            for map in map_list:
                if map.resample:
                    map.sample_weights()

        return

    def _cache_rff(
        self,
        data_list: list[torch.Tensor],
        map_list: list[RandomFourierFeature],
        tau_list: list[float],
    ) -> tuple[
        list[torch.Tensor],
        list[float],
        list[torch.Tensor],
        list[float],
        list[RandomFourierFeature],
    ]:
        cached = []
        cached_tau = []
        not_cached = []
        not_cached_map = []
        not_cached_tau = []

        for map, data, tau in zip(map_list, data_list, tau_list):
            if not map.resample:
                cached.append(map(data, self.matmul_batch))
                cached_tau.append(tau)
            else:
                not_cached.append(data)
                not_cached_map.append(map)
                not_cached_tau.append(tau)

        return cached, cached_tau, not_cached, not_cached_tau, not_cached_map

    @abstractmethod
    def null_dim_reduction(self, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def init_erasure(
        self, tol: float, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass
