import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from obliviator.schemas import MLPConfig, OptimConfig
from obliviator.utils.misc import mlp_factory, optim_factory

NUM_THREADS = 8
MIN_TEST_BATCH = 1024


class MLPCrossEntropy:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device,
        mlp_config: MLPConfig | None = None,
        optim_config: OptimConfig | None = None,
    ) -> None:
        if mlp_config is None:
            mlp_config = MLPConfig()

        if optim_config is None:
            optim_config = OptimConfig()

        self.x_test = x_test
        self.y_test = y_test
        self.device = device
        self.max_acc = 0

        self.test_batch = max(optim_config.batch_size * 2, MIN_TEST_BATCH)
        mlp_config.input_dim = x.shape[1]
        mlp_config.out_dim = int(y.max().item()) + 1

        self.net = mlp_factory(mlp_config)
        self.loss = tnn.CrossEntropyLoss()
        self.net.to(device=device)
        self.loss.to(device=device)
        self.optimizer = optim_factory(optim_config)(self.net.parameters())

        data = TensorDataset(x, y)
        self.loader = DataLoader(
            data,
            batch_size=optim_config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_THREADS,
            pin_memory=True,
        )

    def train(self, epoch: int = 100) -> None:
        pbar = trange(epoch)
        for _ in pbar:
            for x, y in self.loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                self.optimizer.zero_grad()
                logit = self.net(x)
                loss = self.loss(logit, y)
                loss.backward()
                self.optimizer.step()

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
