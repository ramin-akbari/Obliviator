from dataclasses import field

import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from obliviator.schemas import MLPConfig, OptimConfig
from obliviator.utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8


class MLPCrossEntropy:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device,
        update_accuracy_interval: int = 40,
        mlp_config: MLPConfig = field(default_factory=MLPConfig),
        optim_config: OptimConfig = field(default_factory=OptimConfig),
    ) -> None:
        self.x_test = x_test.to(device=device)
        self.y_test = y_test.to(device=device)
        data = TensorDataset(x, y)
        self.loader = DataLoader(
            data,
            batch_size=OptimConfig.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_THREADS,
            pin_memory=True,
        )

        mlp_config.input_dim = x.shape[0]
        mlp_config.out_dim = int(y.max().item()) + 1

        self.net = mlp_factory(mlp_config)
        self.loss = tnn.CrossEntropyLoss()
        self.optimizer = optim_factory(optim_config)(self.net.parameters())
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
