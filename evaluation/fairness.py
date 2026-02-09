import torch


def DP(y: torch.Tensor, y_prd: torch.Tensor, s: torch.Tensor) -> float:
    n_group = torch.unique(y)
    group_true = y_prd[s == 1]
    group_false = y_prd[s == 0]
    n_true = len(group_true)
    n_false = len(group_false)
    dp = [
        (group_true == i).sum() / n_true - (group_false == i).sum() / n_false
        for i in n_group
    ]
    return torch.tensor(dp).abs_().mean().item()


def TPR(y: torch.Tensor, y_prd: torch.Tensor, s: torch.Tensor) -> float:
    n_group = torch.unique(y)
    mask_s_true = s == 1
    mask_s_false = s == 0
    tpr = []
    for i in n_group:
        mask_false = (y == i) & mask_s_false
        mask_true = (y == i) & mask_s_true
        mask_y = y_prd == i
        tpr.append(
            (mask_y & mask_false).sum() / mask_false.sum()
            - (mask_y & mask_true).sum() / mask_true.sum()
        )

    return torch.tensor(tpr).square_().mean().sqrt().item()
