from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .schemas import UnsupervisedConfig
from .utils.linalg import RandomFourierFeature, median_heuristic_sigma
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
        config: UnsupervisedConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.x = torch.as_tensor(x, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)
        self.x_test = torch.as_tensor(x_test, dtype=dtype, device=config.device)
        self.sigma_min = config.sigma_min

        self.tau_x = config.tau_x
        self.tau_z = config.tau_z

        self.encoder_config = config.encoder_config
        self.encoder = mlp_factory(config.encoder_config)
        self.optim = optim_factory(config.optim_config)
        self.device = device
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
            device,
            config.resamle_rff_weights_x,
            median_heuristic_sigma(self.x, config.sigma_min_x),
        )

        self.phi = _rff_helper(
            config.encoder_config.out_dim,
            config.drff_min,
            config.drff_max,
            device,
            True,
        )

        if config.use_rff_s:
            phi_s = _rff_helper(
                self.s.shape[1],
                config.drff_min,
                config.drff_max,
                device,
                False,
                median_heuristic_sigma(s, config.sigma_min_s),
            )
            self.s = phi_s(self.s)

    def _loss_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        w = self.encoder(z)
        w = w.div(w.norm(dim=1, keepdim=True))
        self.phi.sigma = median_heuristic_sigma(w, self.sigma_min)
        return self.phi(w)

    @torch.no_grad()
    def get_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        embd = self.encoder(z)
        return embd.div_(embd.norm(dim=1, keepdim=True))

    def update_encoder(self, in_dim: int, out_dim: int) -> None:
        self.encoder_config.input_dim = in_dim
        self.encoder_config.hidden_dim = out_dim
        self.encoder = mlp_factory(self.encoder_config)

    @abstractmethod
    def _init_encoder(self, epochs: int) -> None:
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
