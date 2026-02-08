import numpy as np
import torch
from torch.utils.data import Dataset


class _BaseDataset(Dataset):
    def __init__(
        self,
        z: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(z) == len(s), "Inputs must have the same length!"
        self.z = torch.as_tensor(z, dtype=dtype)
        self.s = torch.as_tensor(s, dtype=dtype)

    def __len__(self) -> int:
        return self.z.shape[0]


class InitDataset(_BaseDataset):
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z[idx], self.s[idx]


class UnsupervisedDataset(_BaseDataset):
    def __init__(
        self,
        z: torch.Tensor | np.ndarray,
        x: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(z, s, dtype)
        assert len(x) == len(s), "Inputs must have the same length!"
        self.x = torch.as_tensor(x, dtype=dtype)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z[idx], self.x[idx], self.s[idx]


class SupervisedDataset(_BaseDataset):
    def __init__(
        self,
        z: torch.Tensor | np.ndarray,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        s: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(z, s, dtype)
        assert len(x) == len(y) == len(z), "Inputs must have the same length!"
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z[idx], self.x[idx], self.y[idx], self.s[idx]
