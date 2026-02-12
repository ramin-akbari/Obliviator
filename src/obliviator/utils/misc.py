from functools import partial

import torch.nn as tnn
import torch.optim as topt

from obliviator.schemas import (
    ActivationType,
    MLPConfig,
    OptimConfig,
)


def get_activation(act_type: ActivationType) -> tnn.Module:
    match act_type:
        case ActivationType.RELU:
            return tnn.ReLU(inplace=True)
        case ActivationType.GELU:
            return tnn.GELU()
        case ActivationType.SILU:
            return tnn.SiLU(inplace=True)
        case ActivationType.TANH:
            return tnn.Tanh()
        case ActivationType.SIGMOID:
            return tnn.Sigmoid()


def optim_factory(config: OptimConfig) -> partial[topt.Optimizer]:
    if not config.use_adaptive_momentum:
        return partial(
            topt.SGD,
            lr=config.lr,
            momentum=config.beta_1,
            dampening=config.beta_2,
            weight_decay=config.weight_decay,
            nesterov=config.use_nesterov,
        )
    if config.use_nesterov:
        return partial(
            topt.NAdam,
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            weight_decay=config.weight_decay,
            eps=config.epsilon,
            momentum_decay=config.momentum_decay,
            decoupled_weight_decay=config.decoupled_weight_decay,
        )

    return partial(
        topt.AdamW,
        lr=config.lr,
        betas=(config.beta_1, config.beta_2),
        weight_decay=config.weight_decay,
        eps=config.epsilon,
    )


def mlp_factory(
    config: MLPConfig,
) -> tnn.Module:
    net = [
        tnn.Linear(config.input_dim, config.hidden_dim),
        get_activation(config.activation),
    ]
    for _ in range(config.n_layer - 1):
        net.append(tnn.Linear(config.hidden_dim, config.hidden_dim))
        net.append(get_activation(config.activation))

    if config.use_projection:
        net.append(tnn.Linear(config.hidden_dim, config.out_dim))

    return tnn.Sequential(*net)
