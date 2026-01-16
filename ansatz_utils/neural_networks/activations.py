from typing import Optional
import torch
from torch.nn import GELU, SELU, SiLU, Identity, Sigmoid, Softplus, ReLU, LeakyReLU, ELU, Mish, Tanh


class Sin(torch.nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_activation(activation: str, constant: Optional[float] = 0.01) -> nn.Module:
    """ The function generates an instance of an activation function to the user's choice.
    Args:
        activation (str):
            Either one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', 'identity' or 'tanh'.
        constant (float, Optional):
            A tunable hyperparameter that some of the mentioned activation functions use. Neglected when irrelevant.
            Default: 0.01.

    Returns:
        Torch instance of the selected activation function.

    Raises:
        NotImplementedError: If none of the above is chosen.

    """
    if activation.lower() == 'identity':
        return Identity()
    elif activation.lower() == 'leakyrelu':
        return LeakyReLU(negative_slope=constant)
    elif activation.lower() == 'elu':
        return ELU(alpha=constant)
    elif activation.lower() == 'relu':
        return ReLU()
    elif activation.lower() == 'silu':
        return SiLU()
    elif activation.lower() == 'sigmoid':
        return Sigmoid()
    elif activation.lower() == 'softplus':
        return Softplus(beta=constant)
    elif activation.lower() == 'mish':
        return Mish()
    elif activation.lower() == 'tanh':
        return Tanh()
    elif activation.lower() == 'selu':
        return SELU()
    elif activation.lower() == 'gelu':
        return GELU()
    elif activation.lower() == 'sin':
        return Sin()

    raise NotImplementedError("Select one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', "
                              "'identity', 'selu', 'gelu', 'sin' or 'tanh'")
