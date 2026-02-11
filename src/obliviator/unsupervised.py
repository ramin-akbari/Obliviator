from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from .schemas import MLPConfig, UnsupervisedConfig
from .utils.kernel import RandomFourierFeature, median_sigma
from .utils.linalg import batched_matmul, cross_cov, null_pca, null_supervised_pca
from .utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8
DRFF_SCALE: int = 4


class Unsupervised:
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

    def null_dim_reduction(self, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
        # map input with RFF
        x = self.phi_x(self.x, self.matmul_batch)
        self.x_test = self.phi_x(self.x_test, self.matmul_batch)

        # perfoming KPCA/SKPCA [depending on erasure scheme] in the null space of Csx
        f = self._dim_reduction(x, tol)

        # update input and test
        self.x = self._update_and_project(f, x, normalize=True)

        # update RFF map
        self.phi_x.change_params(
            d_in=f.shape[1], sigma=median_sigma(self.x, self.sigma_min_x)
        )

        # update Encoder Parameters
        self._update_encoder(f.shape[1])
        return self.x, self.x_test

    def init_erasure(
        self, tol: float, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_list = [self.x]
        phi_list = [self.phi_x]
        tau_list = [self.tau_x]
        z = self._obliviator_step(data_list, phi_list, tau_list, epochs, tol)
        return z, self.x_test

    def erasure_step(
        self, z: torch.Tensor, tol: float, epochs: int, update_x: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # prepare data for erasure
        data_list = [z, self.x]
        phi_list = [self.phi, self.phi_x]
        tau_list = [self.tau_z, self.tau_x]

        # perform erasure : encoder + evp
        z_new = self._obliviator_step(data_list, phi_list, tau_list, epochs, tol)

        # if we want to update x with the current RV
        if update_x:
            self.sigma_min_x = self.sigma_min_z
            self.x = z.to(self.x.device)
            self.phi_x.change_params(
                self.x.shape[1], median_sigma(self.x, self.sigma_min_x)
            )
        return z_new, self.x_test

    def _obliviator_step(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        epochs: int,
        tol: float,
    ) -> torch.Tensor:
        self._train_encoder(data_list, phi_list, tau_list, epochs)
        z = self._solve_evp(data_list, phi_list, tau_list, tol)
        self._update_encoder(z.shape[1])
        return z

    def _solve_evp(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        tol: float,
    ) -> torch.Tensor:
        # first element should always be the input to the encoder
        # map input and test data
        w = self._get_embeddings(data_list[0], self.encoder_batch)
        self.x_test = self._get_embeddings(self.x_test, self.encoder_batch)

        # update RFF map (not necessary)
        self.phi.change_params(sigma=median_sigma(w, self.sigma_min))

        # map encoder's output using RFF
        w = self.phi(w, self.matmul_batch)
        self.x_test = self.phi(self.x_test, self.matmul_batch)

        # map data_list
        evp_data = [
            phi(var, self.matmul_batch) for var, phi in zip(data_list, phi_list)
        ]

        # solve the SKPCA (EVP in the paper) in the nullspace of Csx
        f = null_supervised_pca(
            w, evp_data, tau_list, self.s, self.device, self.matmul_batch, rtol=tol
        )

        return self._update_and_project(f, w, True)

    # update encoder for iterative erasure
    def _update_encoder(self, in_dim: int) -> None:
        self.encoder_config.input_dim = in_dim
        self.encoder = mlp_factory(self.encoder_config)

    def change_encoder(self, config: MLPConfig) -> None:
        if config != self.encoder_config:
            self.encoder_config = config
            self.encoder = mlp_factory(config)
            self.phi.change_params(d_in=config.out_dim, do_resample=False)

    # equivalent to f(x) - mu_f
    def _update_and_project(
        self, f: torch.Tensor, x: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        mu = x.mean(dim=0)
        x.sub_(mu)

        # update input
        x = batched_matmul(x, f, self.matmul_batch, self.device)

        # update test
        self.x_test.sub_(mu)
        self.x_test = batched_matmul(self.x_test, f, self.matmul_batch, self.device)

        if normalize:
            x.div_(self.x.norm(dim=1, keepdim=True))
            self.x_test.div_(self.x_test.norm(dim=1, keepdim=True))
        return x

    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        return null_pca(x, self.s, self.device, self.matmul_batch, rtol=tol)

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
        cached = []
        cached_tau = []
        not_cached = []
        not_cached_phi = []
        not_cached_tau = []

        for map, data, tau in zip(phi_list, data_list, tau_list):
            if not map.resample:
                cached.append(map(data, self.matmul_batch))
                cached_tau.append(tau)
            else:
                not_cached.append(data)
                not_cached_phi.append(map)
                not_cached_tau.append(tau)

        return cached, cached_tau, not_cached, not_cached_tau, not_cached_phi

    def _prepare_data(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        taus: list[float],
    ) -> tuple[
        list[torch.Tensor], list[RandomFourierFeature], list[float], list[float], int
    ]:
        # first element is the input to the encoder
        z = data_list[0]

        # cache rff map if resampling is false for faster training
        cached, cached_taus, not_cached, not_cached_taus, not_cached_phi = (
            self._cache_rff(data_list, phi_list, taus)
        )

        # reordering inputs [cached, uncached, z, s]
        n_cached = len(cached)
        processed = cached + not_cached
        processed.append(z)
        processed.append(self.s)

        return processed, not_cached_phi, cached_taus, not_cached_taus, n_cached

    def _rff_encoder_embeddings(self, z_batch: torch.Tensor) -> torch.Tensor:
        w = self.encoder(z_batch)
        w = w.div(w.norm(dim=1, keepdim=True))
        self.phi.change_params(
            sigma=median_sigma(w, self.sigma_min, alpha=self.smooth_sigma_factor)
        )
        return self.phi(w)

    @torch.no_grad()
    def _get_embeddings(self, z: torch.Tensor, batch: int) -> torch.Tensor:
        def helper(x: torch.Tensor) -> torch.Tensor:
            x = self.encoder(x.to(device=self.device, non_blocking=True))
            return x.div_(x.norm(dim=1, keepdim=True)).to(device=x.device)

        return torch.cat(
            [helper(z_batched) for z_batched in torch.split(z, batch, dim=0)],
            dim=0,
        )

    def _train_encoder(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        epochs: int,
    ):

        processed, not_cached_phi, cached_taus, not_cached_taus, n_cached = (
            self._prepare_data(data_list, phi_list, tau_list)
        )
        data = self.loader(TensorDataset(*processed))
        optimizer = self.optim_factory(self.encoder.parameters())
        pbar = trange(epochs)

        for _ in pbar:
            for *rvs, z, s in data:
                optimizer.zero_grad()
                s = s.to(self.device, non_blocking=True)
                z = z.to(self.device, non_blocking=True)
                rvs = [rv.to(self.device, non_blocking=True) for rv in rvs]
                w = self._rff_encoder_embeddings(z)

                # s is already cached
                hs_s = cross_cov(w, s).square().mean().sqrt()

                hs_p = torch.tensor(0.0, device=self.device)

                # HSIC for cached inputs, we normalize HSIC based on the dimension of RVs
                for tau, rv in zip(cached_taus, rvs[:n_cached]):
                    hs_p = hs_p + cross_cov(w, rv).square().mean().sqrt().mul(tau)

                # HSIC for not_cached inputs
                for tau, phi, rv in zip(
                    not_cached_taus, not_cached_phi, rvs[n_cached:]
                ):
                    hs_p = hs_p + cross_cov(w, phi(rv)).square().mean().sqrt().mul(tau)

                loss = hs_s - hs_p
                loss.backward()
                optimizer.step()

            # resample RFF weights
            for phi in phi_list:
                if phi.resample:
                    phi.sample_weights()
