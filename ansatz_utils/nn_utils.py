from typing import Optional

from torch import nn


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
        return nn.Identity()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=constant)
    elif activation.lower() == 'elu':
        return nn.ELU(alpha=constant)
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'silu':
        return nn.SiLU()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'softplus':
        return nn.Softplus(beta=constant)
    elif activation.lower() == 'mish':
        return nn.Mish()
    elif activation.lower() == 'tanh':
        return nn.Tanh()

    raise NotImplementedError("Select one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', "
                              "'identity' or 'tanh'")


if __name__ == '__main__':
    pass

    # b, k, n = 10, 8, 13
    # temp = torch.rand(b, k, n, device='cpu', dtype=torch.float64, requires_grad=True) * 100 + 7
