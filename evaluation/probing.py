from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as tnn
from tqdm import tqdm

from obliviator.schemas import MLPConfig, OptimConfig, TermColor
from obliviator.utils.misc import mlp_factory, optim_factory


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
    name: str = "Classifier"
    color: TermColor = TermColor.WHITE


class MLPCrossEntropy:
    def __init__(self, data: ProbData, config: ProbConfig) -> None:
        self.device = torch.device(config.device)
        self.x = data.x
        self.y = data.y
        self.x_test = data.x_test
        self.y_test = data.y_test

        self.max_acc = 0
        self.mlp_config = config.mlp_config
        self.name = config.name
        self.color = config.color

        config.mlp_config.input_dim = data.x.shape[1]
        config.mlp_config.out_dim = int(data.y.max().item()) + 1

        self.net = mlp_factory(config.mlp_config)
        self.loss = tnn.CrossEntropyLoss()
        self.net.to(device=self.device)
        self.loss.to(device=self.device)
        self.optimizer = optim_factory(config.optim_config)
        self.train_batch = config.optim_config.batch_size

    def train(self, epochs: int = 100) -> None:
        optim = self.optimizer(self.net.parameters())

        x_buf = torch.empty_like(self.x).pin_memory()
        y_buf = torch.empty_like(self.y).pin_memory()
        N = self.x.shape[0] - (self.x.shape[0] % self.train_batch)

        pbar = tqdm(
            total=epochs * (N // self.train_batch),
            dynamic_ncols=True,
        )

        for _ in range(epochs):
            idx = torch.randperm(self.x.shape[0])
            x_buf.copy_(self.x[idx])
            y_buf.copy_(self.y[idx])
            for i in range(0, N, self.train_batch):
                x = x_buf[i : i + self.train_batch].to(
                    device=self.device, non_blocking=True
                )
                y = y_buf[i : i + self.train_batch].to(
                    device=self.device, non_blocking=True
                )
                optim.zero_grad()
                logit = self.net(x)
                loss = self.loss(logit, y)
                loss.backward()
                optim.step()
                pbar.update()
                pbar_str = f"Best Accuracy:{self.max_acc * 100:<5.2f}   Loss:{loss.item():<6.3f}"
                pbar.bar_format = f"{self.color}{self.name}|{{bar}}| {{n_fmt}}/{{total_fmt}} [{pbar_str}]{TermColor.RESET}"

            self.max_acc = max(self.accuracy(), self.max_acc)

        pbar.close()

    @torch.no_grad()
    def accuracy(self) -> float:
        # not needed, but for safety if net architecture has changed
        self.net.eval()
        prd = torch.cat(
            [
                self.net(x.to(self.device)).argmax(dim=1).to(self.y_test.device)
                for x in torch.split(self.x_test, self.train_batch * 2)
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
