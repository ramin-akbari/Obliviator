import numpy as np
import torch
from torch.utils.data import Dataset


class UnsupervisedDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(x) == len(s), "Inputs x and s must have the same length!"
        self.x = torch.as_tensor(x, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.s[idx]


class SupervisedDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(x) == len(y) == len(s), "Inputs x, y, s must have the same length!"
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.s[idx]
