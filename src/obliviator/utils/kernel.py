from math import sqrt as pysqrt

import torch
from numpy import ndarray


class RandomFourierFeature:
    def __init__(
        self,
        d_in: int,
        scale: int,
        drff_max: int,
        drff_min: int,
        sigma_rff: float,
        resample: bool,
        device: torch.device,
    ) -> None:
        def helper_rff(d_in: int):
            return max(min(scale * d_in, drff_max), drff_min) // 2

        self._get_drff = helper_rff
        self._drff = self._get_drff(d_in)
        self._sigma = sigma_rff
        self._d_in = d_in

        def helper_weight_sampler() -> torch.Tensor:
            return torch.randn(self._d_in, self._drff, device=device).div_(self._sigma)

        self.w = helper_weight_sampler()
        self.sampler = helper_weight_sampler
        self.c = pysqrt(1.0 / self._drff)
        self.resample = resample

    def map(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.to(device=self.w.device)
        rff = xd @ self.w
        return torch.cat([rff.sin(), rff.cos()], dim=1).mul(self.c).to(device=x.device)

    def batched_map(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        feature = []
        for xb in torch.split(x, batch):
            xb = xb.to(device=self.w.device)
            rff = xb @ self.w
            feature.append(torch.cat([rff.sin(), rff.cos()], dim=1).to(device=x.device))
        return torch.cat(feature, dim=0)

    def change_params(
        self,
        d_in: int | None = None,
        sigma: float | None = None,
        do_resample: bool = True,
    ) -> None:
        is_changed = False
        if d_in is not None and d_in != self._d_in:
            self._d_in = d_in
            self._drff = self._get_drff(d_in)
            self.c = pysqrt(1.0 / self._drff)
            is_changed = True

        if sigma is not None:
            self._sigma = sigma
            self.sample_weights()
            return

        if do_resample and is_changed:
            self.sample_weights()

    def __call__(self, x: torch.Tensor, batch: int | None = None) -> torch.Tensor:
        if batch is None:
            return self.map(x)

        return self.batched_map(x, batch)

    def sample_weights(self):
        self.w = self.sampler()


def median_sigma(
    x: torch.Tensor | ndarray,
    sigma_min: float,
    alpha: float = 1.0,
    max_sample: int = 5_000,
) -> float:
    x = torch.as_tensor(x)
    if x.shape[0] > max_sample:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_sample]]

    n = x.shape[0]
    sigma = (
        torch.cdist(x, x)[*torch.triu_indices(n, n, offset=1, device=x.device)]
        .median()
        .item()
    )
    return max(alpha * sigma, sigma_min)
