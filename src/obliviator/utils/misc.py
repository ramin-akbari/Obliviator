from typing import Callable

import torch.nn as tnn

type ActFactory = Callable[[], tnn.Module]


def mlp_factory(
    dim_x: int,
    dim_hidden: int,
    n_layer: int,
    use_projection: bool,
    dim_out: int,
    activation: ActFactory = lambda: tnn.SiLU(inplace=True),
) -> tnn.Module:
    net = [tnn.Linear(dim_x, dim_hidden), activation()]
    for _ in range(n_layer - 1):
        net.append(tnn.Linear(dim_hidden, dim_hidden))
        net.append(activation())

    if use_projection:
        net.append(tnn.Linear(dim_hidden, dim_out))

    return tnn.Sequential(*net)
