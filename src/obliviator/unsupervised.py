from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset
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

        self.encoder_batch = config.optim_config.batch_size
        self.mm_batch = config.matmul_batch

        self.tau_x = config.tau_x
        self.tau_z = config.tau_z
        self.evptau_x = config.evp_tau_x
        self.evptau_z = config.evp_tau_z

        self.encoder_config = config.encoder_config
        self._update_encoder(self.x.shape[1])

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
        print(median_sigma(self.x, config.sigma_min_x,alpha=1.25))
        self.phi_x = RandomFourierFeature(
            self.x.shape[1],
            config.rff_scale_x,
            config.drff_max,
            config.drff_min,
            median_sigma(self.x, config.sigma_min_x,alpha=1.25),
            config.resample_x,
            self.device,
        )

        self.phi_z = RandomFourierFeature(
            config.encoder_config.out_dim,
            config.rff_scale_z,
            config.drff_max,
            config.drff_min,
            sigma_rff=1.0,
            resample=config.resample_z,
            device=self.device,
        )

        self.phi = RandomFourierFeature(
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
        x = self.phi_x(self.x, self.mm_batch)
        self.x_test = self.phi_x(self.x_test, self.mm_batch)

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
        # The very first step for erasure
        data_list = [self.x]
        phi_list = [self.phi_x]
        tau_list = [self.tau_x]
        evptau_list = [self.evptau_x]
        z = self._obliviator_step(
            data_list, phi_list, tau_list, evptau_list, epochs, tol
        )
        return z, self.x_test

    def erasure_step(
        self, z: torch.Tensor, tol: float, epochs: int, update_x: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normal step (intermediate) for erasure
        data_list = [z, self.x]
        phi_list = [self.phi_z, self.phi_x]
        tau_list = [self.tau_z, self.tau_x]
        evptau_list = [self.evptau_z, self.evptau_x]

        # Obliviator iterative erasure : encoder + evp
        z_new = self._obliviator_step(
            data_list, phi_list, tau_list, evptau_list, epochs, tol
        )

        # if we want to update x with the current RV
        if update_x:
            self.sigma_min_x = self.sigma_min_z
            self.x = z.to(self.x.device)
            self.phi_x.change_params(
                self.x.shape[1], median_sigma(self.x, self.sigma_min_x)
            )
        return z_new, self.x_test

    def _dim_reduction(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        return null_pca(x, self.s, self.device, self.mm_batch, rtol=tol)

    def _obliviator_step(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        evptau_list: list[float],
        epochs: int,
        tol: float,
    ) -> torch.Tensor:
        out_tuple = self._prepare_data(data_list, phi_list, tau_list, evptau_list)
        self._train_encoder(*out_tuple[:-2], epochs)
        evptau_list_processed = out_tuple[-1] + out_tuple[-2]
        z = self._solve_evp(*out_tuple[:3], evptau_list_processed, tol)

        # intermediate rv is updated, so we update encoder and RFF
        self._update_encoder(z.shape[1])
        self.phi_z.change_params(
            d_in=z.shape[1], sigma=median_sigma(z, self.sigma_min_z)
        )
        return z

    def _train_encoder(
        self,
        processed_data: list[torch.Tensor],
        n_cached: int,
        active_phi: list[RandomFourierFeature],
        cached_taus: list[float],
        not_cached_taus: list[float],
        epochs: int,
    ):
        def scaled_cross_cov_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return _cross_cov(x, y, batch=None, device=x.device).square().mean().sqrt()

        data = self.loader(TensorDataset(*processed_data, self.s))
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
                hs_s = scaled_cross_cov_norm(w, s)

                hs_p = torch.tensor(0.0, device=self.device)

                # HSIC for cached inputs, we normalize HSIC based on the dimension of RVs
                for tau, rv in zip(cached_taus, rvs[:n_cached]):
                    hs_p = hs_p + scaled_cross_cov_norm(w, rv).mul(tau)

                # HSIC for the not_cached inputs
                for tau, phi, rv in zip(not_cached_taus, active_phi, rvs[n_cached:]):
                    hs_p = hs_p + scaled_cross_cov_norm(w, phi(rv)).mul(tau)

                loss = hs_s - hs_p
                loss.backward()
                optimizer.step()

            # resample active RFF weights for the next epoch
            for phi in active_phi:
                phi.sample_weights()

    def _solve_evp(
        self,
        processed_data: list[torch.Tensor],
        n_cached: int,
        active_phi: list[RandomFourierFeature],
        evptau_list: list[float],
        tol: float,
    ) -> torch.Tensor:
        # for the processed data after caching we put z as the last element
        w = self._get_embeddings(processed_data[-1], self.encoder_batch)
        self.x_test = self._get_embeddings(self.x_test, self.encoder_batch)

        # update RFF map (not really necessary)
        self.phi.change_params(sigma=median_sigma(w, self.sigma_min))

        # map encoder's output using RFF
        w = self.phi(w, self.mm_batch)
        self.x_test = self.phi(self.x_test, self.mm_batch)

        # map data_list
        evp_data = [
            phi(var, self.mm_batch)
            for var, phi in zip(processed_data[n_cached:], active_phi)
        ]

        evp_data.extend(processed_data[:n_cached])

        # solve the SKPCA (EVP in the paper) in the nullspace of Csx
        f = null_supervised_pca(
            w, evp_data, evptau_list, self.s, self.device, self.mm_batch, rtol=tol
        )

        return self._update_and_project(f, w, True)

    # update encoder after each iteration of erasure
    def _update_encoder(self, in_dim: int) -> None:
        self.encoder_config.input_dim = in_dim
        self.encoder = mlp_factory(self.encoder_config)

    # to change encoder architecture
    def change_encoder(self, config: MLPConfig) -> None:
        if config != self.encoder_config:
            self.encoder_config = config
            self.encoder = mlp_factory(config)
            self.phi.change_params(d_in=config.out_dim, do_resample=False)

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
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[RandomFourierFeature],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        # cache RFF for those RVs that doesn't require
        # resampling for more efficient training
        cached: list[torch.Tensor] = []
        not_cached: list[torch.Tensor] = []
        cached_tau: list[float] = []
        not_cached_tau: list[float] = []
        cached_evptau: list[float] = []
        not_cached_evptau: list[float] = []
        not_cached_phi: list[RandomFourierFeature] = []

        for map, data, tau, evptau in zip(phi_list, data_list, tau_list, evptau_list):
            if not map.resample:
                cached.append(map(data, self.mm_batch))
                cached_tau.append(tau)
                cached_evptau.append(evptau)
            else:
                not_cached.append(data)
                not_cached_phi.append(map)
                not_cached_tau.append(tau)
                not_cached_evptau.append(evptau)

        return (
            cached,
            not_cached,
            not_cached_phi,
            cached_tau,
            not_cached_tau,
            cached_evptau,
            not_cached_evptau,
        )

    def _prepare_data(
        self,
        data_list: list[torch.Tensor],
        phi_list: list[RandomFourierFeature],
        tau_list: list[float],
        evptau_list: list[float],
    ) -> tuple[
        list[torch.Tensor],
        int,
        list[RandomFourierFeature],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        # first element is the input to the encoder
        z = data_list[0]

        # cache rff map if resampling is false for faster training
        out_tuple = self._cache_rff(data_list, phi_list, tau_list, evptau_list)

        # reordering inputs [cached[,y], uncached, z]
        # y is present in supervised
        n_cached = len(out_tuple[0])
        processed = out_tuple[0] + out_tuple[1]
        processed.append(z)

        return processed, n_cached, *out_tuple[2:]

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
