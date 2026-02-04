import torch


class RandomFourierFeature:
    def __init__(
        self, dim_x: int, dim_rff: int, sigma: float, device: torch.device
    ) -> None:
        dim_rff = dim_rff // 2

        def helper_weight_sampler() -> torch.Tensor:
            return torch.randn(dim_x, dim_rff, device=device).div_(sigma)

        self.sampler = helper_weight_sampler
        self.sample_weights()
        self.c = (1.0 / torch.tensor(dim_rff, device=device)).sqrt()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.w.device)
        rff = x @ self.w
        return torch.cat([rff.sin().mul_(self.c), rff.cos().mul_(self.c)], dim=1)

    def sample_weights(self) -> None:
        self.w = self.sampler()


def cross_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    # technically not needed, only because of floating-point error in x_mean
    y = y - y.mean(dim=0)
    return (x.T @ y).mul_(1 / x.shape[0])


def median_heuristic_sigma(x: torch.Tensor, max_sample: int = 5_000) -> torch.Tensor:
    if x.shape[0] > max_sample:
        x = x[torch.randperm(x.shape[0])[:max_sample]]

    n = x.shape[0]
    return torch.cdist(x, x)[*torch.triu_indices(n, n, offset=1)].median()
