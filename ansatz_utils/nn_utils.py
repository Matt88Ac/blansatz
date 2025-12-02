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


class AllDifferences(autograd.Function):
    """ This class implements both a function that maps a tensor X to [X_j - X_i: 1 <= i < j <= len(X.T)]
        and it's gradient.
    """

    @staticmethod
    @torch.jit.export
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
    @torch.jit.export
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


@torch.jit.script
def vandermonde_determinant(X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        X (torch.Tensor):
            A real tensor X of shape [b, d_1, ..., d_k, n]

    Returns:
        The Vandermonde Determinant of X along the last axis.
    """
    return AllDifferences.apply(X).prod(-1, True)


def uniform_sphere_point(n_samples: int, dim: int) -> torch.Tensor:
    samples = torch.randn(n_samples, dim)
    return samples / torch.norm(samples, dim=-1, keepdim=True)


@torch.jit.script
def prod_all_but_one(X: torch.Tensor) -> torch.Tensor:
    return torch.cat([X[..., :i].prod(-1, True) * X[..., i + 1:].prod(-1, True) for i in range(X.size(-1))], -1)


def kk(X: torch.Tensor, offset=1, scale=0.5) -> torch.Tensor:
    pabo = prod_all_but_one(X)
    scales = torch.full_like(X, scale, requires_grad=True, device=X.device, dtype=X.dtype,
                             pin_memory=X.is_pinned(X.device), layout=X.layout)
    out = scales * pabo / (offset * X.prod(-1, True) + pabo.clone().sum(-1, True))
    return out


if __name__ == '__main__':
    import numpy as np
    from permutation_actions import all_transpositions


    def make_transpositions(_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        *size, d, n = _x.shape
        at = all_transpositions(_x.shape[-2], _x.device).unsqueeze(0).unsqueeze(-1)
        for i in range(len(size) - 1):
            at = at.unsqueeze(2)

        shaped_input = _x.clone().unsqueeze(1)
        trans_input = torch.take_along_dim(shaped_input, at, -2)
        # print(shaped_input.shape, trans_input.shape, _x.shape)
        return shaped_input, trans_input


    x = torch.rand(1, 5, 30)
    x[:, :, 3] = x[:, :, 2]

    xs, xt = make_transpositions(x.transpose(-2, -1))

    xx = (xs - xt).norm(dim=(-2, -1))

    print(kk(xx))

    # b, k, n = 10, 8, 13
    # temp = torch.rand(b, k, n, device='cpu', dtype=torch.float64, requires_grad=True) * 100 + 7
