from copy import copy
from typing import NamedTuple

import torch
from tqdm import trange

from .schemas import MLPConfig, UnsupervisedConfig, UnsupervisedData
from .utils.kernel import RandomFourierFeature, median_sigma
from .utils.linalg import (
    _cross_cov,
    null_pca,
    null_supervised_pca,
)
from .utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8


class DataSplit(NamedTuple):
    # Fixed features (cached RFFs or Linear Features)
    static_features: list[torch.Tensor]
    static_taus: list[float]
    static_evptaus: list[float]

    # Dynamic Feature (RFF with Resampling)
    dynamic_features: list[torch.Tensor]
    dynamic_phis: list[RandomFourierFeature]
    dynamic_taus: list[float]
    dynamic_evptaus: list[float]


class Unsupervised:
    def __init__(
        self,
        data: UnsupervisedData,
        config: UnsupervisedConfig,
    ) -> None:
        self.x = data.x
        self.s = data.s
        self.x_test = data.x_test

        self.sigma_min = config.sigma_min
        self.sigma_min_z = config.sigma_min_z
        self.sigma_min_x = config.sigma_min_x
        self.smooth_sigma_factor = config.smoother_rff_factor

        self.mm_batch = config.matmul_batch

        self.tau_x = config.tau_x
        self.tau_z = config.tau_z
        self.evptau_x = config.evp_tau_x
        self.evptau_z = config.evp_tau_z

        self._encoder_config = copy(config.encoder_config)

        self.optim_factory = optim_factory(config.optim_config)
        self.batch = config.optim_config.batch_size
        self.device = torch.device(config.device)
        self._encoder = torch.nn.Identity()  # dummy encoder

        self._phi_x = RandomFourierFeature(
            self.x.shape[1],
            config.rff_scale_x,
            config.drff_max,
            config.drff_min,
            median_sigma(self.x, config.sigma_min_x),
            config.resample_x,
            self.device,
        )

        self._phi_z = RandomFourierFeature(
            config.encoder_config.out_dim,
            config.rff_scale_z,
            config.drff_max,
            config.drff_min,
            sigma_rff=1.0,
            resample=config.resample_z,
            device=self.device,
        )

        self._phi = RandomFourierFeature(
            config.encoder_config.out_dim,
            config.rff_scale,
            config.drff_max,
            config.drff_min,
            sigma_rff=1.0,
            resample=False,
            device=self.device,
        )

        if config.use_rff_s:
            phi_s = RandomFourierFeature(
                self.s.shape[1],
                config.rff_scale_s,
                config.drff_max,
                config.drff_min_s,
                median_sigma(self.s, config.sigma_min_s),
                False,
                self.device,
            )
            self.s = phi_s(self.s, self.mm_batch)

    def null_dim_reduction(self, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
        # map input with RFF
        x = self._phi_x(self.x, self.mm_batch)
        self.x_test = self._phi_x(self.x_test, self.mm_batch)

        # perfoming KPCA/SKPCA [depending on erasure scheme] in the null space of Csx
        f = self._dim_reduction(x, tol)

        # update input and test
        self.x = self._update_and_project(f, x, normalize=True)

        # update RFF map
        self._phi_x.change_params(
            d_in=f.shape[1], sigma=median_sigma(self.x, self.sigma_min_x)
        )

        # update Encoder Parameters
        self._encoder_config.input_dim = f.shape[1]
        return self.x, self.x_test

    def init_erasure(
        self, tol: float, epochs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The very first step for erasure
        data_list = [self.x]
        phi_list = [self._phi_x]
        tau_list = [self.tau_x]
        evptau_list = [self.evptau_x]
        z = self._obliviator_step(
            self.x, data_list, phi_list, tau_list, evptau_list, epochs, tol
        )
        return z, self.x_test

    def erasure_step(
        self, z: torch.Tensor, tol: float, epochs: int, update_x: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normal step (intermediate) for erasure
        data_list = [z, self.x]
        phi_list = [self._phi_z, self._phi_x]
        tau_list = [self.tau_z, self.tau_x]
        evptau_list = [self.evptau_z, self.evptau_x]

        # Obliviator iterative erasure : encoder + evp
        z_new = self._obliviator_step(
            z, data_list, phi_list, tau_list, evptau_list, epochs, tol
        )

        # if we want to update x with the current RV
        if update_x:
            self.sigma_min_x = self.sigma_min_z
            self.x = z.to(self.x.device)
            self._phi_x.change_params(
                self.x.shape[1], median_sigma(self.x, self.sigma_min_x)
            )
        return z_new, self.x_test

    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        return null_pca(x, self.s, self.device, self.mm_batch, rtol=tol)

    def _obliviator_step(
        self,
        input_rv: torch.Tensor,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        evptau_list: list[float],
        epochs: int,
        tol: float,
    ) -> torch.Tensor:
        data_split = self._cache_rff(data_list, phi_list, tau_list, evptau_list)

        self._train_encoder(input_rv, data_split, epochs)
        rv = self._solve_evp(input_rv, data_split, tol)

        # intermediate rv is updated, so we update encoder and RFF
        self._encoder_config.input_dim = rv.shape[1]
        self._phi_z.change_params(
            d_in=rv.shape[1], sigma=median_sigma(rv, self.sigma_min_z)
        )
        return rv

    def _train_encoder(
        self,
        input_rv: torch.Tensor,
        data: DataSplit,
        epochs: int,
    ):
        def scaled_cross_cov_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return _cross_cov(x, y, batch=None, device=x.device).square().mean().sqrt()

        self._encoder = mlp_factory(self._encoder_config).to(device=self.device)
        optimizer = self.optim_factory(self._encoder.parameters())
        pbar = trange(epochs)
        N = input_rv.shape[0] - (input_rv.shape[0] % self.batch)

        s_buf = torch.empty_like(self.s).pin_memory()
        rv_buf = torch.empty_like(input_rv).pin_memory()
        static_buf = [torch.empty_like(t).pin_memory() for t in data.static_features]
        dynamic_buf = [torch.empty_like(t).pin_memory() for t in data.dynamic_features]

        def shuffle():
            idx = torch.randperm(input_rv.shape[0])
            s_buf.copy_(self.s[idx])
            rv_buf.copy_(input_rv[idx])
            for dst, src in zip(static_buf, data.static_features):
                dst.copy_(src[idx])
            for dst, src in zip(dynamic_buf, data.dynamic_features):
                dst.copy_(src[idx])

        for _ in pbar:
            shuffle()
            for i in range(0, N, self.batch):
                optimizer.zero_grad()
                s = s_buf[i : i + self.batch].to(self.device, non_blocking=True)
                z = rv_buf[i : i + self.batch].to(self.device, non_blocking=True)
                static_rv = [
                    rv[i : i + self.batch].to(self.device, non_blocking=True)
                    for rv in static_buf
                ]
                dynamic_rv = [
                    rv[i : i + self.batch].to(self.device, non_blocking=True)
                    for rv in dynamic_buf
                ]

                w = self._rff_encoder_embeddings(z)

                # s is already cached
                hs_s = scaled_cross_cov_norm(w, s)

                hs_p = torch.tensor(0.0, device=self.device)

                # HSIC for cached inputs, we normalize HSIC based on the dimension of RVs
                for tau, rv in zip(data.static_taus, static_rv):
                    hs_p = hs_p + scaled_cross_cov_norm(w, rv).mul(tau)

                # HSIC for the not_cached inputs
                for tau, phi, rv in zip(
                    data.dynamic_taus, data.dynamic_phis, dynamic_rv
                ):
                    hs_p = hs_p + scaled_cross_cov_norm(w, phi(rv)).mul(tau)

                loss = hs_s - hs_p
                loss.backward()
                optimizer.step()

            # resample active RFF weights for the next epoch
            for phi in data.dynamic_phis:
                phi.sample_weights()

    def _solve_evp(
        self,
        input_rv: torch.Tensor,
        data_split: DataSplit,
        tol: float,
    ) -> torch.Tensor:
        # for the processed data after caching we put z as the last element
        w = self.get_embeddings(input_rv, self.batch)
        self.x_test = self.get_embeddings(self.x_test, self.batch)

        # update RFF map (not really necessary, it is only one iteration behind from encoder training)
        self._phi.change_params(
            sigma=median_sigma(w, self.sigma_min, alpha=self.smooth_sigma_factor)
        )

        # map encoder's output using RFF
        w = self._phi(w, self.mm_batch)
        self.x_test = self._phi(self.x_test, self.mm_batch)

        # map data_list
        evp_data = [
            phi(var, self.mm_batch)
            for var, phi in zip(data_split.dynamic_features, data_split.dynamic_phis)
        ]
        evp_data.extend(data_split.static_features)

        evptau_list = data_split.dynamic_evptaus + data_split.static_evptaus

        # solve the SKPCA (EVP in the paper) in the nullspace of Csx
        f = null_supervised_pca(
            w, evp_data, evptau_list, self.s, self.device, self.mm_batch, rtol=tol
        )

        return self._update_and_project(f, w, True)

    # to change encoder architecture
    def change_encoder(self, config: MLPConfig) -> None:
        if config != self._encoder_config:
            self._encoder_config = copy(config)
            self._encoder = mlp_factory(config)
            self._phi.change_params(d_in=config.out_dim, do_resample=False)

    # equivalent to f(x) - mu_f
    def _update_and_project(
        self, f: torch.Tensor, x: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        # we assume f is already on the device [since it is the solution of evp]
        def project(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
            if self.mm_batch is None:
                xd = x.to(device=self.device)
                return xd.mm(mat).to(device=x.device)

            def helper(bx: torch.Tensor):
                bx = bx.to(device=self.device)
                return bx.mm(mat).to(device=bx.device)

            return torch.cat(
                [helper(bx) for bx in torch.split(x, self.mm_batch)], dim=0
            )

        mu = x.mean(dim=0)

        # update input
        x.sub_(mu)
        x = project(x=x, mat=f)

        # update test
        self.x_test.sub_(mu)
        self.x_test = project(x=self.x_test, mat=f)

        if normalize:
            x.div_(x.norm(dim=1, keepdim=True))
            self.x_test.div_(self.x_test.norm(dim=1, keepdim=True))
        return x

    def _cache_rff(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        evptau_list: list[float],
    ) -> DataSplit:
        # cache RFF for those RVs that doesn't require
        # resampling for more efficient training
        cached: list[torch.Tensor] = []
        not_cached: list[torch.Tensor] = []
        cached_tau: list[float] = []
        not_cached_tau: list[float] = []
        cached_evptau: list[float] = []
        not_cached_evptau: list[float] = []
        not_cached_phi: list[RandomFourierFeature] = []

        for rff, rv, tau, evptau in zip(phi_list, data_list, tau_list, evptau_list):
            if not rff.resample:
                cached.append(rff(rv, self.mm_batch))
                cached_tau.append(tau)
                cached_evptau.append(evptau)
            else:
                not_cached.append(rv)
                not_cached_phi.append(rff)
                not_cached_tau.append(tau)
                not_cached_evptau.append(evptau)

        return DataSplit(
            static_features=cached,
            static_taus=cached_tau,
            static_evptaus=cached_evptau,
            dynamic_features=not_cached,
            dynamic_phis=not_cached_phi,
            dynamic_taus=not_cached_tau,
            dynamic_evptaus=not_cached_evptau,
        )

    def _rff_encoder_embeddings(self, z_batch: torch.Tensor) -> torch.Tensor:
        w = self._encoder(z_batch)
        w = w.div(w.norm(dim=1, keepdim=True))
        self._phi.change_params(
            sigma=median_sigma(w, self.sigma_min, alpha=self.smooth_sigma_factor)
        )
        return self._phi(w)

    @torch.no_grad()
    def get_embeddings(self, z: torch.Tensor, batch: int) -> torch.Tensor:
        def helper(x: torch.Tensor) -> torch.Tensor:
            x = self._encoder(x.to(device=self.device, non_blocking=True))
            return x.div_(x.norm(dim=1, keepdim=True)).to(device=x.device)

        return torch.cat(
            [helper(z_batched) for z_batched in torch.split(z, batch, dim=0)],
            dim=0,
        )
