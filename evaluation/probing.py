from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from obliviator.schemas import MLPConfig, OptimConfig
from obliviator.utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8
MIN_TEST_BATCH = 1024


@dataclass(slots=True)
class ProbingData:
    x: torch.Tensor
    y: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    dtype: torch.dtype = torch.float32
    label_dtype: torch.dtype = torch.long

    def __post_init__(self):
        def helper(data: Any, data_type: torch.dtype) -> torch.Tensor:
            if not isinstance(data, (torch.Tensor, np.ndarray, list)):
                raise ValueError(
                    "Input must either a torch tensor, numpy array or python list"
                )
            return torch.as_tensor(data, dtype=data_type)

        self.x = helper(self.x, self.dtype)
        self.x_test = helper(self.x_test, self.dtype)

        self.y = helper(self.y, self.label_dtype)
        self.y_test = helper(self.y_test, self.label_dtype)


class MLPCrossEntropy:
    def __init__(
        self,
        data: ProbingData,
        device: torch.device,
        mlp_config: MLPConfig | None = None,
        optim_config: OptimConfig | None = None,
    ) -> None:
        if mlp_config is None:
            mlp_config = MLPConfig()

        if optim_config is None:
            optim_config = OptimConfig()

        self.x = data.x
        self.y = data.y
        self.x_test = data.x_test
        self.y_test = data.y_test
        self.device = device
        self.max_acc = 0
        self.mlp_config = mlp_config

        self.test_batch = max(optim_config.batch_size * 2, MIN_TEST_BATCH)
        mlp_config.input_dim = data.x.shape[1]
        mlp_config.out_dim = int(data.y.max().item()) + 1

        self.net = mlp_factory(mlp_config)
        self.loss = tnn.CrossEntropyLoss()
        self.net.to(device=device)
        self.loss.to(device=device)
        self.optimizer = optim_factory(optim_config)

        self.loader = partial(
            DataLoader,
            batch_size=optim_config.batch_size,
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
            for x, y in self.loader(data):
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
