from functools import partial
from math import sqrt as pysqrt

import torch
from numpy import ndarray


def cross_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    # technically not needed, only because of floating-point error in x_mean
    y = y - y.mean(dim=0)
    return (x.T @ y).mul(1 / x.shape[0])


def batched_cov(x: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    mu_x = x.mean(dim=0).to(device=device)
    batched_x = torch.split(x, batch)
    Cxx = torch.zeros(x.shape[1], x.shape[1], device=device)

    for xb in batched_x:
        xb = xb.to(device=device, non_blocking=True)
        xb = xb.sub(mu_x)
        Cxx = Cxx.addmm(xb.T, xb)

    return Cxx.div(x.shape[0])


def batched_cross_cov(
    x: torch.Tensor, y: torch.Tensor, batch: int, device: torch.device
) -> torch.Tensor:
    mu_x = x.mean(dim=0).to(device=device)
    mu_y = y.mean(dim=0).to(device=device)
    batched_x = torch.split(x, batch)
    batched_y = torch.split(y, batch)
    Cxy = torch.zeros(x.shape[1], y.shape[1], device=device)

    for xb, yb in zip(batched_x, batched_y):
        xb = xb.to(device=device, non_blocking=True)
        yb = yb.to(device=device, non_blocking=True)
        xb = xb.sub(mu_x)
        yb = yb.sub(mu_y)
        Cxy = Cxy.addmm(xb.T, yb)

    return Cxy.div(x.shape[0])


def batched_matmul(
    x: torch.Tensor, y_fixed: torch.Tensor, batch: int | None, device: torch.device
) -> torch.Tensor:
    y_fixed = y_fixed.to(device=device)

    if batch is None:
        return x.mm(y_fixed).cpu()

    def helper(bx: torch.Tensor):
        bx = bx.to(device=device)
        return bx.mm(y_fixed).cpu()

    return torch.cat([helper(bx) for bx in torch.split(x, batch)], dim=0)


def select_top_k_eigvec(x: torch.Tensor, rtol: float, atol: float) -> torch.Tensor:
    eigval, eigvec = torch.linalg.eigh(x)
    tol = max(eigval[-1] * rtol, atol)
    return eigvec[:, eigval > tol]


def find_null_sx(
    x: torch.Tensor,
    s: torch.Tensor,
    device: torch.device,
    batch: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.Tensor:
    if batch is None:
        Csx = cross_cov(s, x)
    else:
        Csx = batched_cross_cov(s, x, batch, device)

    _, sigmas, v = torch.linalg.svd(Csx, full_matrices=False)
    tol = max(rtol * sigmas[0], atol)
    v = v[sigmas > tol].T
    full_v = torch.linalg.qr(v, mode="complete")[0]
    return full_v[:, v.shape[1] :]


def null_supervised_pca(
    x: torch.Tensor,
    rvs: tuple[torch.Tensor, ...],
    taus: tuple[float, ...],
    s: torch.Tensor,
    device: torch.device,
    batch: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.Tensor:
    u_null = find_null_sx(x, s, device, batch, rtol, atol)
    mat = torch.zeros(u_null.shape[1], u_null.shape[1], device=device)
    if batch is None:
        helper = partial(cross_cov, y=x)
    else:
        helper = partial(batched_cross_cov, y=x, device=device, batch=batch)

    for tau, rv in zip(taus, rvs):
        C = helper(rv).mm(u_null)
        mat.addmm_(C.T, C, alpha=tau)

    pcs = select_top_k_eigvec(mat, rtol, atol)
    return u_null.mm(pcs)


def null_pca(
    x: torch.Tensor,
    s: torch.Tensor,
    device: torch.device,
    batch: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> torch.Tensor:
    u_null = find_null_sx(x, s, device, batch, rtol, atol)
    if batch is None:
        C = torch.cov(x.T)
    else:
        C = batched_cov(x, batch, device)
    C = C.mm(u_null)
    C = u_null.T.mm(C)
    pcs = select_top_k_eigvec(C, rtol, atol)
    return u_null.mm(pcs)


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
        rff = x @ self.w
        return torch.cat([rff.sin(), rff.cos()], dim=1).mul(self.c)

    def batched_map(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        feature = []
        for xb in torch.split(x, batch):
            xb = xb.to(device=self.w.device)
            rff = xb @ self.w
            feature.append(torch.cat([rff.sin(), rff.cos()], dim=1).cpu())
        return torch.cat(feature, dim=0)

    def change_params(
        self,
        d_in: int | None = None,
        sigma: float | None = None,
    ) -> None:

        if d_in is not None:
            self._d_in = d_in
            self._drff = self._get_drff(d_in)
            self.c = pysqrt(1.0 / self._drff)

        if sigma is not None:
            self._sigma = sigma

        self.w = self.sampler()

    def __call__(self, x: torch.Tensor, batch: int | None = None) -> torch.Tensor:
        if batch is None:
            x = x.to(device=self.w.device)
            return self.map(x).cpu()

        return self.batched_map(x, batch)


def median_sigma(
    x: torch.Tensor | ndarray,
    sigma_min: float,
    alpha: float = 1,
    max_sample: int = 5_000,
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
    return max(alpha * sigma, sigma_min)
