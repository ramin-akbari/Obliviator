import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from obliviator.utils.misc import ActFactory, mlp_factory


class ClassficationDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert len(x) == len(y), "Inputs x and labels y must have the same length!"
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class MLPCrossEntropy:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device,
        dim_hidden: int,
        n_layer: int,
        activation: ActFactory = lambda: tnn.SiLU(inplace=True),
        lr: float = 5e-3,
        batch_size: int = 4096,
        weight_decay: float = 1e-2,
        update_accuracy_interval: int = 40,
    ) -> None:
        self.x_test = x_test.to(device=device)
        self.y_test = y_test.to(device=device)
        data = ClassficationDataset(x, y)
        self.loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        n_label = int(y.max().item()) + 1
        self.net = mlp_factory(
            x.shape[1], dim_hidden, n_layer, True, n_label, activation
        )
        self.loss = tnn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.acc_interval = update_accuracy_interval
        self.net.to(device=device)
        self.loss.to(device=device)
        self.device = device
        self.max_acc = 0

    def train(self, epoch: int = 100) -> None:
        pbar = trange(epoch)
        for _ in pbar:
            for idx, (x, y) in enumerate(self.loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                self.optimizer.zero_grad()
                logit = self.net(x)
                loss = self.loss(logit, y)
                loss.backward()
                self.optimizer.step()
                if not (idx + 1) % self.acc_interval:
                    self.max_acc = max(self.accuracy(), self.max_acc)
                    pbar.set_description(f"Best_Accuracy :{self.max_acc * 100:<5.2f}")

    @torch.no_grad()
    def accuracy(self) -> float:
        prd = self.net(self.x_test).argmax(dim=1)
        return prd.eq(self.y_test).float().mean().item()
