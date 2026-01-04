from typing import Optional, Union, Iterable

import torch
from torch import nn

from .aggregations import get_aggregation, partial
from .mlp import MLP


class DeepSets(nn.Module):
    """ 
    DeepSets class, paper available at https://proceedings.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf

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
        new_dim (bool, Optional):
            Whether the input dimension is a new dimension. If True, the input is expected to have shape
            (..., n_elements, 1), and will be unsqueezed upon the forward pass. Default: False.
        swap_last_axes (bool, Optional):
            Whether to swap between the last two axes, upon the forward pass. Default: True.
        aggregation (str, Optional):
            Either one of 'mean', 'sum', 'max', 'min' or 'prod'. Default: 'mean'.
        activation: (str, Optional):
            Either one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', 'identity' or 'tanh'.
            Default: 'leakyrelu'.
        activation_constant: (float, Optional):
            A tunable hyperparameter that some of the mentioned activation functions use. Neglected when irrelevant.
            Default: 0.01.
        layer_norm (bool, Iterable[bool], str, Optional):
            When set to True, layer normalization is applied after each linear layer. Another option is to specify for
            each hidden layer, by setting a boolean iterable. Default: False. If set to 'all_but_last', layer normalization is applied
            to all layers except the last one.
         elementwise_affine (bool, Optional):
            Whether the layer normalization includes learnable affine parameters. Default: True.
        device (str, torch.device, Optional): The device. Default: 'cpu'.
        dtype (str, torch.dtype, Optional): The dtype. Default: torch.float64.
        kwargs: Additional arguments (not used).
    
    Attributes:
       mlp (nn.Module):
           An MLP model that defines the architecture.
       agg (partial):
              An aggregation function.
       swap_last_axes (bool):
           Whether to swap between the last two axes, upon the forward pass. Default: True.

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

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Optional[list[int]] = None,
                 biases: Optional[Union[bool, Iterable[bool], str]] = True,
                 new_dim: Optional[bool] = False,
                 swap_last_axes: Optional[bool] = True,
                 aggregation: Optional[str] = 'mean',
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[Union[bool, Iterable[bool], str]] = False,
                 elementwise_affine: Optional[bool] = True,
                 device=torch.device('cpu'), dtype=torch.float64, **kwargs):
        super(DeepSets, self).__init__()
        if new_dim:
            assert in_dim == 1
            swap_last_axes = False
        self.mlp = MLP(in_dim, out_dim, hidden_layers, biases, activation, activation_constant,
                       layer_norm, elementwise_affine, device, dtype)
        self.new_dim = new_dim
        self.swap_last_axes = swap_last_axes
        self.agg = get_aggregation(
            agg_name=aggregation,
            dim=-2,
            keepdims=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.new_dim:
            fx = self.mlp(x.unsqueeze(-1))
        else:
            if self.swap_last_axes:
                assert x.size(-2) == self.mlp.in_dim
                fx = self.mlp(x.swapaxes(-1, -2))
            else:
                assert x.size(-1) == self.mlp.in_dim
                fx = self.mlp(x)
        return self.agg(fx)




