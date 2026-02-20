from functools import partial

import torch


def _cov_mat(x: torch.Tensor, batch: int | None, device: torch.device) -> torch.Tensor:
    if batch is None:
        return torch.cov(x.to(device=device).T)

    mu_x = x.mean(dim=0).to(device=device)
    batched_x = torch.split(x, batch)
    Cxx = torch.zeros(x.shape[1], x.shape[1], device=device)

    for xb in batched_x:
        xb = xb.to(device=device, non_blocking=True)
        xb = xb.sub(mu_x)
        Cxx = Cxx.addmm(xb.T, xb)

    return Cxx.div(x.shape[0])


def _cross_cov(
    x: torch.Tensor, y: torch.Tensor, batch: int | None, device: torch.device
) -> torch.Tensor:

    if batch is None:
        xd = x.to(device=device)
        yd = y.to(device=device)
        xd = xd - xd.mean(dim=0)
        yd = yd - yd.mean(dim=0)
        Cxy = xd.T.mm(yd)
        return Cxy.div(x.shape[0])

    batched_x = torch.split(x, batch)
    batched_y = torch.split(y, batch)
    Cxy = torch.zeros(x.shape[1], y.shape[1], device=device)
    mu_x = x.mean(dim=0).to(device=device)
    # technically not needed, only because of floating-point error in x_mean
    mu_y = y.mean(dim=0).to(device=device)

    for xb, yb in zip(batched_x, batched_y):
        xb = xb.to(device=device, non_blocking=True)
        yb = yb.to(device=device, non_blocking=True)
        xb = xb.sub(mu_x)
        yb = yb.sub(mu_y)
        Cxy = Cxy.addmm(xb.T, yb)

    return Cxy.div(x.shape[0])


def _select_top_k_eigvec(
    x: torch.Tensor, rtol: float, atol: float, display_eigs: bool
) -> torch.Tensor:
    eigval, eigvec = torch.linalg.eigh(x)
    tol = max(eigval[-1] * rtol, atol)
    eigvec = eigvec[:, eigval > tol]
    if display_eigs:
        eigs = (eigval[-8:].clone().flip(dims=0) / eigval[-1]).cpu().tolist()
        print(
            f"Normalized eigs: {''.join([f'{e:<4.2e}' for e in eigs])}   Dimension:f{eigvec.shape[1]}"
        )

    return eigvec


def _find_null(
    C: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 2e-7,
) -> torch.Tensor:

    _, sigmas, v = torch.linalg.svd(C, full_matrices=False)
    tol = max(rtol * sigmas[0], atol)
    v = v[sigmas > tol].T
    full_v = torch.linalg.qr(v, mode="complete")[0]
    return full_v[:, v.shape[1] :]


def null_supervised_pca(
    target_rv: torch.Tensor,
    align_rvs: list[torch.Tensor],
    align_taus: list[float],
    null_rv: torch.Tensor,
    device: torch.device,
    batch: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    display_eigs: bool = False,
) -> torch.Tensor:

    Csx = _cross_cov(null_rv, target_rv, batch, device)
    u_null = _find_null(Csx, rtol, atol)

    mat = torch.zeros(u_null.shape[1], u_null.shape[1], device=device)
    cov_ix = partial(_cross_cov, y=target_rv, device=device, batch=batch)

    taus = torch.as_tensor(align_taus)
    taus.div_(taus.sum())

    for tau, rv in zip(taus, align_rvs):
        C = cov_ix(rv).mm(u_null)
        C = C.T.mm(C)
        C.div_(fast_sym_spectral_norm(C) + 1e-7)
        mat.add_(C.mul_(tau))
    pcs = _select_top_k_eigvec(mat, rtol, atol, display_eigs)

    return u_null.mm(pcs)


def fast_sym_spectral_norm(C: torch.Tensor, num_iter: int = 5) -> float:
    v = torch.randn(C.shape[1], device=C.device, dtype=C.dtype)
    for _ in range(num_iter):
        v.copy_(C.mv(v))
        v.div_(v.norm())
    return C.mv(v).norm().item()


def null_pca(
    target_rv: torch.Tensor,
    null_rv: torch.Tensor,
    device: torch.device,
    batch: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    display_eigs: bool = False,
) -> torch.Tensor:
    Csx = _cross_cov(null_rv, target_rv, batch, device)
    u_null = _find_null(Csx, rtol, atol)
    C = _cov_mat(target_rv, batch, device)
    C = C.mm(u_null)
    C = u_null.T.mm(C)
    pcs = _select_top_k_eigvec(C, rtol, atol, display_eigs)
    return u_null.mm(pcs)
