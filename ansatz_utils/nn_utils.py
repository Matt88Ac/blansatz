from typing import Optional

import torch
from torch import autograd
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


class SelfDiff(autograd.Function):
    """ This class implements both a function that maps a tensor X to [X_j - X_i: 1 <= i < j <= len(X.T)]
        and it's gradient.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx: Neglect.
            inp (torch.Tensor):
                A real tensor X of shape [b, d_1, ..., d_k, n]

        Returns:
            A real tensor, of shape [b, d_1, ..., d_k, n(n-1)/2], for [X_j - X_i: 1 <= i < j <= n]
        """
        *b, n = inp.shape
        differences = []
        for i in range(n - 1):
            differences.append(
                inp[..., i + 1:] - inp[..., i:i + 1]
            )
        ctx.save_for_backward(inp)
        ctx.n = n
        ctx.size = n * (n - 1) // 2
        ctx.b = b
        return torch.cat(differences, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        n = ctx.n
        b = ctx.b
        size = ctx.size

        grad_input = torch.zeros(*b, n, size, device=X.device, dtype=X.dtype)
        last_sub_matrix_size = 0
        sub_matrix_size = n - 1
        to_add = n - 2
        for i in range(n):
            grad_input[..., i, last_sub_matrix_size:sub_matrix_size] += -1
            grad_input[..., i + 1:, last_sub_matrix_size:sub_matrix_size] += torch.eye(to_add + 1, dtype=X.dtype,
                                                                                       device=X.device)
            last_sub_matrix_size = sub_matrix_size
            sub_matrix_size += to_add
            to_add -= 1

        grad_input = (grad_input * grad_output.unsqueeze(len(b))).sum(dim=-1)

        return grad_input


if __name__ == '__main__':
    pass

    # b, k, n = 10, 8, 13
    # temp = torch.rand(b, k, n, device='cpu', dtype=torch.float64, requires_grad=True) * 100 + 7
