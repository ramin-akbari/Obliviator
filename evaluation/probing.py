from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from obliviator.schemas import MLPConfig, OptimConfig
from obliviator.utils.misc import mlp_factory, optim_factory

NUM_THREADS = 16
MIN_TEST_BATCH = 2048


@dataclass(slots=True)
class ProbData:
    x: torch.Tensor
    y: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    def __init__(
        self,
        *,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
        y_test: torch.Tensor | np.ndarray,
        dtype: torch.dtype = torch.float32,
    ):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.x_test = torch.as_tensor(x_test, dtype=dtype)

        self.y = torch.as_tensor(y, dtype=torch.long)
        self.y_test = torch.as_tensor(y_test, dtype=torch.long)


@dataclass
class ProbConfig:
    device: str = "cpu"
    mlp_config: MLPConfig = field(default_factory=MLPConfig)
    optim_config: OptimConfig = field(default_factory=OptimConfig)


class MLPCrossEntropy:
    def __init__(self, data: ProbData, config: ProbConfig) -> None:
        self.x = data.x
        self.y = data.y
        self.x_test = data.x_test
        self.y_test = data.y_test
        self.device = torch.device(config.device)
        self.max_acc = 0
        self.mlp_config = config.mlp_config

        self.test_batch = max(config.optim_config.batch_size, MIN_TEST_BATCH)
        config.mlp_config.input_dim = data.x.shape[1]
        config.mlp_config.out_dim = int(data.y.max().item()) + 1

        self.net = mlp_factory(config.mlp_config)
        self.loss = tnn.CrossEntropyLoss()
        self.net.to(device=self.device)
        self.loss.to(device=self.device)
        self.optimizer = optim_factory(config.optim_config)
        self.train_batch = config.optim_config.batch_size

        self.loader = partial(
            DataLoader,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_THREADS,
            pin_memory=True,
        )

    def train(self, epoch: int = 100) -> None:
        optim = self.optimizer(self.net.parameters())
        pbar = trange(epoch)
        data = TensorDataset(self.x, self.y)

        for _ in pbar:
            for x, y in tqdm(self.loader(data, batch_size=self.train_batch)):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                optim.zero_grad()
                logit = self.net(x)
                loss = self.loss(logit, y)
                loss.backward()
                optim.step()

            self.max_acc = max(self.accuracy(), self.max_acc)
            pbar.set_description(f"Best_Accuracy :{self.max_acc * 100:<5.2f}")

    @torch.no_grad()
    def accuracy(self) -> float:
        # not needed, but for safety if net architecture has changed
        self.net.eval()
        prd = torch.cat(
            [
                self.net(x.to(self.device)).argmax(dim=1).to(self.y_test.device)
                for x in torch.split(self.x_test, self.test_batch)
            ]
        )
        self.net.train()
        return prd.eq(self.y_test).float().mean().item()

    @torch.no_grad()
    def reset_net(self) -> None:
        def helper(module: tnn.Module) -> None:
            reset_param = getattr(module, "reset_parameters", None)
            if callable(reset_param):
                reset_param()

        self.net.apply(helper)

    def update_input(self, *, x: torch.Tensor, x_test: torch.Tensor) -> None:
        self.x = x.to(device=self.x.device, dtype=self.x.dtype)
        self.x_test = x_test.to(device=self.x_test.device, dtype=self.x_test.dtype)

        self.max_acc = 0
        if self.mlp_config.input_dim != x.shape[1]:
            self.mlp_config.input_dim = x.shape[1]
            self.net = mlp_factory(self.mlp_config)
        else:
            self.reset_net()

        self.net.to(self.device)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.net.to(device=self.device)
        self.loss.to(device=self.device)
