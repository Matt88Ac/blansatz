from typing import Optional, Union, Iterable

import torch
from torch import nn

from .activations import get_activation
from .dropout import DropoutEquivariant


class AffineBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: Optional[bool] = True,
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[bool] = False,
                 elementwise_affine: Optional[bool] = True,
                 dropout_p: Optional[float] = 0.0,
                 device=torch.device('cpu'), dtype=torch.float64):
        super(AffineBlock, self).__init__()

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

        if dropout_p > 0:
            self.dropout = DropoutEquivariant(out_dim, dropout_p).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is None:
            return self.layer(x)
        return self.dropout(self.layer(x))

    def reset_dropout(self):
        if self.dropout is None:
            return
        self.dropout.reset_parameters()


class MLP(nn.Module):
    """ 
    MLP class.

    Args:
        in_dim (int):
            Size of each input sample.
        out_dim (int):
            Desired output dimension.
        hidden_layers (list[int], Optional):
            The width of each hidden layer. If None, then the MLP is equivalent to a linear layer. Default: None.
        biases (bool, Iterable[bool], str, Optional):
            When set to True, each linear layer learns an additive bias, and doesn't when set to False. Another
            option is to specify for each hidden layer, by setting a boolean iterable. Default: True.
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
        in_dim (int):
            Size of each input sample.
        out_dim (int):
            Size of each output sample.
        layer_dims (list[int]):
            Each input/output dimension.

    Examples:
        >>> mlp = MLP(5, 7, [3, 10, 2], True)
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Optional[list[int]] = None,
                 biases: Optional[Union[bool, Iterable[bool], str]] = True,
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[Union[bool, Iterable[bool], str]] = False,
                 elementwise_affine: Optional[bool] = True,
                 dropout_p: Optional[float] = 0.0,
                 device=torch.device('cpu'), dtype=torch.float64, **kwargs):
        super(MLP, self).__init__()
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

        activations = [activation] * self.depth
        activations[-1] = 'identity'

        dropout_p = [dropout_p] * self.depth
        dropout_p[-1] = 0.0

        self.layers = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.layers.append(
                AffineBlock(in_dim, out_dim, biases[i], activations[i], activation_constant,
                            layer_norm[i], elementwise_affine, dropout_p[i], device, dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = layer(y)
        return y

    def reset_dropout(self):
        for i in range(self.depth):
            self.layers[i].reset_dropout()


if __name__ == '__main__':
    pass

    X = torch.rand(10, 3, 2, 5, dtype=torch.float64)
    mlp = MLP(5, 7, [3, 10, 2], 'all_but_last', layer_norm='all_but_last',
              dtype=torch.float64, dropout_p=0.1)
    print(mlp)

    X = torch.rand(10, 3, 2, 5, dtype=torch.float64)

    print(mlp(X).shape)
