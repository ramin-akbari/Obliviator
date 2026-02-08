import torch
from numpy import ndarray


def cross_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    # technically not needed, only because of floating-point error in x_mean
    y = y - y.mean(dim=0)
    return (x.T @ y).mul(1 / x.shape[0])


def batched_cross_cov(
    x: torch.Tensor, y: torch.Tensor, batch: int, device: torch.device
) -> torch.Tensor:
    mu_x = x.mean(dim=0).to(device=device)
    mu_y = y.mean(dim=0).to(device=device)
    batched_x = torch.split(x, batch)
    batched_y = torch.split(y, batch)
    Cxy = torch.zeros(x.shape[1], y.shape[1], device=device)

    for x, y in zip(batched_x, batched_y):
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        x = x.sub(mu_x)
        y = y.sub(mu_y)
        Cxy = Cxy.addmm(x.T, y)

    return Cxy.div(x.shape[0])


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
        d_in: int,
        dim_rff: int,
        resample: bool,
        sigma_rff: float,
        device: torch.device,
    ) -> None:
        dim_rff = dim_rff // 2
        self._sigma = sigma_rff
        self._d_in = d_in

        def helper_weight_sampler() -> torch.Tensor:
            return torch.randn(self._d_in, dim_rff, device=device).div_(self._sigma)

        self.w = helper_weight_sampler()
        self.sampler = helper_weight_sampler
        self.c = (1.0 / torch.tensor(dim_rff, device=device)).sqrt()
        self.resample = resample
        if resample:
            self.map_ptr = self.resample_map
        else:
            self.map_ptr = self.map

    def map(self, x: torch.Tensor | ndarray) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.w.dtype).to(device=self.w.device)
        rff = x @ self.w
        return torch.cat([rff.sin(), rff.cos()], dim=1).mul_(self.c)

    def change_input_dim(self, d_in: int) -> None:
        self._d_in = d_in
        if not self.resample:
            self.w = self.sampler()

    def change_sigma(self, sigma: float) -> None:
        self._sigma = sigma
        if not self.resample:
            self.w = self.sampler()

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
