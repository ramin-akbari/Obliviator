import torch
from numpy import ndarray


def cross_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    # technically not needed, only because of floating-point error in x_mean
    y = y - y.mean(dim=0)
    return (x.T @ y).mul_(1 / x.shape[0])


def select_top_k_eigvec(x: torch.Tensor, rtol: float, atol: float) -> torch.Tensor:
    eigval, eigvec = torch.linalg.eigh(x)
    tol = max(eigval[-1] * rtol, atol)
    return eigvec[:, eigval > tol]


def find_null_xs(
    x: torch.Tensor, s: torch.Tensor, rtol: float, atol: float = 1e-6
) -> torch.Tensor:
    Csx = cross_cov(s, x)
    Csx = Csx.mm(Csx.T)
    eigval, eigvec = torch.linalg.eigh(Csx)
    tol = max(rtol * eigval[-1], atol)
    return eigvec[:, eigval < tol]


def null_pca(
    x: torch.Tensor, s: torch.Tensor, rtol: float, atol: float = 1e-6
) -> torch.Tensor:
    U_null = find_null_xs(x, s, rtol, atol)
    x_null = (x - x.mean(dim=0)).mm(U_null)
    pcs = select_top_k_eigvec(torch.cov(x_null), rtol, atol)
    return U_null.mm(pcs)


class RandomFourierFeature:
    def __init__(
        self,
        dim_x: int,
        dim_rff: int,
        resample: bool,
        sigma_rff: float,
        device: torch.device,
    ) -> None:
        dim_rff = dim_rff // 2
        self.sigma = sigma_rff

        def helper_weight_sampler() -> torch.Tensor:
            return torch.randn(dim_x, dim_rff, device=device).div_(self.sigma)

        self.w = helper_weight_sampler()
        self.sampler = helper_weight_sampler
        self.c = (1.0 / torch.tensor(dim_rff, device=device)).sqrt()
        if resample:
            self.map_ptr = self.map
        else:
            self.map_ptr = self.resample_map

    def map(self, x: torch.Tensor | ndarray) -> torch.Tensor:
        x = torch.as_tensor(x).to(device=self.w.device, dtype=self.w.dtype)
        rff = x @ self.w
        return torch.cat([rff.sin(), rff.cos()], dim=1).mul_(self.c)

    def resample_map(self, x: torch.Tensor | ndarray) -> torch.Tensor:
        self.w = self.sampler()
        return self.map(x)

    def __call__(self, x: torch.Tensor | ndarray) -> torch.Tensor:
        return self.map_ptr(x)


def median_heuristic_sigma(
    x: torch.Tensor | ndarray, sigma_min: float, max_sample: int = 5_000
) -> float:
    x = torch.as_tensor(x)
    if x.shape[0] > max_sample:
        x = x[torch.randperm(x.shape[0])[:max_sample]]

    n = x.shape[0]
    sigma = (
        torch.cdist(x, x)[*torch.triu_indices(n, n, offset=1, device=x.device)]
        .median()
        .item()
    )
    return max(sigma, sigma_min)
