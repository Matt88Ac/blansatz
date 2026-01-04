from typing import Optional, Union, Iterable

import torch
from torch import nn
from .activations import get_activation


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
        >>> mlp
        MLP(
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
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 layer_norm: Optional[Union[bool, Iterable[bool], str]] = False,
                 elementwise_affine: Optional[bool] = True,
                 device=torch.device('cpu'), dtype=torch.float64, **kwargs):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = []

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = [in_dim] + hidden_layers + [out_dim]

        if isinstance(biases, bool):
            biases = [biases] * (len(self.layer_dims) - 1)
        elif biases == 'all_but_last':
            biases = [True] * (len(self.layer_dims) - 1)
            biases[-1] = False
        else:
            biases = list(biases)
            assert len(biases) == len(self.layer_dims) - 1

        if isinstance(layer_norm, bool):
            layer_norm = [layer_norm] * (len(self.layer_dims) - 1)
        elif layer_norm == 'all_but_last':
            layer_norm = [True] * (len(self.layer_dims) - 1)
            layer_norm[-1] = False
        else:
            layer_norm = list(layer_norm)
            assert len(layer_norm) == len(self.layer_dims) - 1

        layers = [nn.Linear(self.layer_dims[0], self.layer_dims[1], bias=biases[0])]
        if layer_norm[0]:
            layers.append(nn.LayerNorm(self.layer_dims[1], bias=biases[0], elementwise_affine=elementwise_affine))

        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[1:-1], self.layer_dims[2:])):
            layers.append(get_activation(activation, activation_constant))
            layers.append(nn.Linear(in_dim, out_dim, bias=biases[i + 1]))
            if layer_norm[i+1]:
                layers.append(nn.LayerNorm(out_dim, bias=biases[i+1], elementwise_affine=elementwise_affine))

        self.layers = nn.Sequential(*layers).to(device=device, dtype=dtype)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    pass

    mlp = MLP(5, 7, [3, 10, 2], True, layer_norm=True)
    X = torch.rand(10, 3, 2, 5, dtype=torch.float64)

    print(mlp(X).shape)
