from typing import Optional, Union, Iterable

import torch
from torch import nn

from .aggregations import get_aggregation, Aggregation
from .activations import get_activation
from .dropout import DropoutEquivariant


class MeanResAffineBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: Optional[bool] = True,
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[bool] = False,
                 elementwise_affine: Optional[bool] = True,
                 dropout_p: Optional[float] = 0.0,
                 mean_residual: Optional[bool] = False,
                 device=torch.device('cpu'), dtype=torch.float64):
        super(MeanResAffineBlock, self).__init__()

        if layer_norm:
            layer = [
                nn.Linear(in_dim, out_dim, bias),
                nn.LayerNorm(out_dim, bias=bias, elementwise_affine=elementwise_affine),
                get_activation(activation, activation_constant),
            ]
        else:
            layer = [
                nn.Linear(in_dim, out_dim, bias),
                get_activation(activation, activation_constant),
            ]
        self.layer = nn.Sequential(*layer).to(device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.dropout = None
        self.mean_residual = mean_residual

        if dropout_p > 0:
            self.dropout = DropoutEquivariant(out_dim, dropout_p).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mean_residual:
            if self.dropout is None:
                return self.layer(x)
            return self.dropout(self.layer(x))
        else:
            if self.dropout is None:
                return self.layer(x + x.mean(dim=-2, keepdim=True))
            return self.dropout(self.layer(x + x.mean(dim=-2, keepdim=True)))

    def reset_dropout(self):
        if self.dropout is None:
            return
        self.dropout.reset_parameters()


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
       agg (Aggregation):
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

    def __init__(self, in_dim: int, out_dim: int, in_channels: int,
                 hidden_layers: Optional[list[int]] = None,
                 biases: Optional[Union[bool, Iterable[bool], str]] = True,
                 new_dim: Optional[bool] = False,
                 swap_last_axes: Optional[bool] = True,
                 aggregation: Optional[str] = 'mean',
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[Union[bool, Iterable[bool], str]] = False,
                 elementwise_affine: Optional[bool] = True,
                 dropout_p: Optional[float] = 0.0,
                 res: Optional[Union[bool, Iterable[bool], str]] = False,
                 device=torch.device('cpu'), dtype=torch.float64, **kwargs):
        super(DeepSets, self).__init__()
        if new_dim:
            assert in_dim == 1
            swap_last_axes = False

        if hidden_layers is None:
            hidden_layers = []

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = [in_dim] + hidden_layers + [out_dim]
        self.depth = len(self.layer_dims) - 1

        if isinstance(biases, bool):
            biases = [biases] * self.depth
        elif biases == 'all_but_last':
            biases = [True] * self.depth
            biases[-1] = False
        else:
            biases = list(biases)
            assert len(biases) == self.depth

        if isinstance(layer_norm, bool):
            layer_norm = [layer_norm] * self.depth
        elif layer_norm == 'all_but_last':
            layer_norm = [True] * self.depth
            layer_norm[-1] = False
        else:
            layer_norm = list(layer_norm)
            assert len(layer_norm) == self.depth

        if isinstance(res, bool):
            res = [res] * self.depth
        elif res == 'all_but_last':
            res = [True] * self.depth
            res[-1] = False
        else:
            res = list(res)
            assert len(res) == self.depth

        activations = [activation] * self.depth
        activations[-1] = 'identity'

        self.dropout_needed = dropout_p > 0

        dropout_p = [dropout_p] * self.depth
        dropout_p[-1] = 0.0

        self.layers = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.layers.append(
                MeanResAffineBlock(in_dim, out_dim, biases[i], activations[i], activation_constant,
                                   layer_norm[i], elementwise_affine, dropout_p[i], res[i], device, dtype)
            )

        self.new_dim = new_dim
        self.swap_last_axes = swap_last_axes

        if 'linear+' in aggregation.lower():
            aggregation = aggregation.lower().split('+')[1]
            self.agg = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False, device=device, dtype=dtype),
                get_aggregation(
                    agg_name=aggregation,
                    dim=-1,
                    keepdims=False
                )
            )

        else:
            self.agg = get_aggregation(
                agg_name=aggregation,
                dim=-1,
                keepdims=False
            )

    def sub_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = layer(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.new_dim:
            fx = self.sub_forward(x.unsqueeze(-1))
        else:
            if self.swap_last_axes:
                assert x.size(-2) == self.in_dim
                fx = self.sub_forward(x.contiguous().transpose(-1, -2))
            else:
                assert x.size(-1) == self.in_dim
                fx = self.sub_forward(x)

        fx = self.agg(fx.contiguous().transpose(-1, -2))
        return fx

    def reset_dropout(self):
        if self.dropout_needed:
            for i in range(self.depth):
                self.layers[i].reset_dropout()
