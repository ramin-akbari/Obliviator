from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .schemas import ErasureConfig
from .utils.linalg import RandomFourierFeature, batched_matmul, median_sigma
from .utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8
DRFF_SCALE: int = 4


def _rff_helper(
    d_in: int,
    d_min: int,
    d_max: int,
    device: torch.device,
    resample: bool,
    sigma: float = 1,
) -> RandomFourierFeature:
    drrf = max(min(DRFF_SCALE * d_in, d_max), d_min)
    return RandomFourierFeature(d_in, drrf, resample, sigma, device)


class Obliviator(ABC):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        config: ErasureConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.x = torch.as_tensor(x, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)
        self.x_test = torch.as_tensor(x_test, dtype=dtype)

        self.sigma_min = config.sigma_min
        self.sigma_min_z = config.sigma_min_z
        self.sigma_min_x = config.sigma_min_x

        self.encoder_batch = config.encoder_batch
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

        self.phi_x = _rff_helper(
            self.x.shape[1],
            config.drff_min,
            config.drff_max,
            self.device,
            config.resample_x,
            median_sigma(self.x, config.sigma_min_x),
        )

        self.phi_z = _rff_helper(
            config.encoder_config.out_dim,
            config.drff_min,
            config.drff_max,
            self.device,
            config.resample_z,
        )

        self.phi = _rff_helper(
            config.encoder_config.out_dim,
            config.drff_min,
            config.drff_max,
            self.device,
            True,
        )

        if config.use_rff_s:
            phi_s = _rff_helper(
                self.s.shape[1],
                config.drff_label_min,
                config.drff_max,
                self.device,
                False,
                median_sigma(s, config.sigma_min_s),
            )
            self.s = phi_s(self.s, self.matmul_batch)

    def _loss_embeddings(self, z_batch: torch.Tensor) -> torch.Tensor:
        w = self.encoder(z_batch)
        w = w.div(w.norm(dim=1, keepdim=True))
        self.phi.change_params(sigma=median_sigma(w, self.sigma_min))
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

    def eval_function(
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

    @abstractmethod
    def _init_dim_reduction(self, tol: float) -> None:
        pass

    @abstractmethod
    def init_erasure(self, tol: float) -> None:
        pass

    @abstractmethod
    def train_encoder(self, epochs: int) -> None:
        pass

    @abstractmethod
    def solve_evp(self, tol: float) -> None:
        pass
