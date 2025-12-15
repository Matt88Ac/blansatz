from typing import Optional, Union, Iterable

import torch
from torch import nn
from .activations import get_activation
from .aggregations import get_aggregation
from .mlp import MLP


class DeepSets(nn.Module):
    """ DeepSets class, paper available at
        https://proceedings.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf

         Attributes:
            mlp (nn.Module):
                Size of each input sample.
            out_dim (int):
                Size of each output sample.
            layer_dims (list[int]):
                Each input/output dimension.
            swap_last_axes (bool):
                Whether to swap between the last two axes, upon the forward pass. Default: True.
     """

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Optional[list[int]] = None,
                 biases: Optional[Union[bool, Iterable[bool]]] = True,
                 new_dim: Optional[bool] = False,
                 swap_last_axes: Optional[bool] = True,
                 aggregation: Optional[str] = None,
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 device=torch.device('cpu'), dtype=torch.float64):
        """
        Args:
            in_dim (int):
                Size of each input sample.
            out_dim (int):
                Desired output dimension.
            hidden_layers (list[int], Optional):
                The width of each hidden layer. If None, then the MLP is equivalent to a linear layer. Default: None.
            biases (bool, Iterable[bool], Optional):
                When set to True, each linear layer learns an additive bias, and doesn't when set to False. Another
                option is to specify for each hidden layer, by setting a boolean iterable. Default: True.
            activation: (str, Optional):
                Either one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', 'identity' or 'tanh'.
                Default: 'leakyrelu'.
            activation_constant: (float, Optional):
                A tunable hyperparameter that some of the mentioned activation functions use. Neglected when irrelevant.
                Default: 0.01.
            device (str, torch.device, Optional): The device. Default: 'cpu'.
            dtype (str, torch.dtype, Optional): The dtype. Default: torch.float64.

        Examples:
            >>> ds = DeepSets(5, 7, [3, 10, 2], True)
            >>> ds
            DeepSets(
              (layers): Sequential(
                (0): Linear(in_features=5, out_features=3, bias=True)
                (1): LeakyReLU(negative_slope=0.01)
                (2): Linear(in_features=3, out_features=10, bias=True)
                (3): LeakyReLU(negative_slope=0.01)
                (4): Linear(in_features=10, out_features=2, bias=True)
                (5): LeakyReLU(negative_slope=0.01)
                (6): Linear(in_features=2, out_features=7, bias=True)
              )
            )
        """
        super(DeepSets, self).__init__()
        if new_dim:
            assert in_dim == 1
            swap_last_axes = False
        self.mlp = MLP(in_dim, out_dim, hidden_layers, biases, activation, activation_constant, device, dtype)
        self.new_dim = new_dim
        self.swap_last_axes = swap_last_axes
        if aggregation is None:
            self.agg = nn.Identity()
        else:
            self.agg = get_aggregation(
                agg_name=aggregation,
                dim=-2,
                keepdims=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.new_dim:
            fx = x.unsqueeze(-1)
        else:
            if self.swap_last_axes:
                assert x.size(-2) == self.mlp.in_dim
                fx = x.swapaxes(-1, -2)
            else:
                assert x.size(-1) == self.mlp.in_dim
                fx = x

        return self.agg(self.mlp(fx))




