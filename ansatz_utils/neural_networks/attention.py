from typing import Optional, Union, Iterable

import torch
from torch import nn
from .activations import get_activation
from .aggregations import get_aggregation


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """ 
    Compute the scaled dot-product attention.
    Args:
        Q (torch.Tensor): Query tensor of shape (..., n, d_k)
        K (torch.Tensor): Key tensor of shape (..., n, d_k)
        V (torch.Tensor): Value tensor of shape (..., n, d_k)

    Returns:
        torch.Tensor: Output tensor of shape (..., n, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.einsum('...ab,...ca->...bc', Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    scores = nn.functional.softmax(scores, dim=-1)
    output = torch.einsum('...bb,...nb->...nb', scores, V)
    return output


class SnEquivariantAttentionBlock(nn.Module):
    """
    An Sn-equivariant attention block.

    Args:
        in_dim (int):
            Size of each input sample.
        out_dim (int):
            Desired output dimension.
        num_heads (int, optional):
            Number of attention heads. Must be divisible by in_dim, and >= 1. Default: 1.
        bias (bool, optional):
            When set to True, each linear layer learns an additive bias, and doesn't when set to False. Default: True.
        device (str, torch.device, optional): The device. Default: 'cpu'.
        dtype (str, torch.dtype, optional): The dtype. Default: torch.float64.

    Attributes:
        qkv (nn.Linear):
            Linear layer to compute queries, keys, and values.
        out_layer (nn.Linear):
            Linear layer to compute the output.
        num_heads (int):
            Number of attention heads.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: Optional[int] = 1,
                 bias: Optional[bool] = True, device=torch.device('cpu'), dtype=torch.float64):
        super(SnEquivariantAttentionBlock, self).__init__()
        assert num_heads >= 1

        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=bias, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.out_layer = nn.Linear(in_dim, out_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x.swapaxes(-1, -2)).chunk(3, dim=-1)

        if self.num_heads > 1:
            *size, n, m = q.shape
            q, k, v = map(lambda t: t.view(*size, n, m // self.num_heads, self.num_heads).transpose(-2, -3), (q, k, v))

        out = scaled_dot_product_attention(q, k, v)

        if self.num_heads > 1:
            out = out.transpose(-2, -3).contiguous().view(*size, n, m)

        return self.out_layer(out).transpose(-1, -2)

    @property
    def out_dim(self):
        return self.out_layer.out_features


class Transformer(nn.Module):
    """
    Attention Transformer class.

    Args:
        in_dim (int):
            Size of each input sample.
        out_dim (int):
            Desired output dimension.
        in_channels (int, optional):
            Number of input channels. If None, then the model will be Sn-equivariant.
        num_heads (int, optional):
            Number of attention heads. Must be divisible by in_dim, and >= 1. Default: 1.
        hidden_layers (list[int], optional):
            The width of each hidden layer. If None, then the MLP is equivalent to a linear layer. Default: None.
        biases (bool, Iterable[bool], str, optional):
            When set to True, each linear layer learns an additive bias, and doesn't when set to False. Another
            option is to specify for each hidden layer, by setting a boolean iterable. Default: True. If set to 'all_but_last',
            then all layers but the last will have biases. 
        aggregation (str, optional):
            One of 'linear', 'mean', 'sum', 'max'. Specifies the aggregation method to use to obtain the final output from the
            output of the last layer. Default: 'linear'. If 'linear' is chosen, then the model will not be Sn-equivariant.
        activation: (str, optional):
            Either one of 'leakyrelu', 'elu', 'relu', 'silu', 'sigmoid', 'softplus', 'mish', 'identity' or 'tanh'.
            Default: 'leakyrelu'.
        activation_constant: (float, optional):
            A tunable hyperparameter that some of the mentioned activation functions use. Neglected when irrelevant.
            Default: 0.01.
        device (str, torch.device, Optional): The device. Default: 'cpu'.
        dtype (str, torch.dtype, Optional): The dtype. Default: torch.float64.
        **kwargs: Placeholder.

    Attributes:
        in_dim (int):
            Size of each input sample.
        out_dim (int):
            Size of each output sample.
        layer_dims (list[int]):
            Each input/output dimension.
        layers (nn.Sequential):
            The sequence of attention blocks and activation functions.
        agg (nn.Module):
            The aggregation module to obtain the final output.
    """

    def __init__(self, in_dim: int, out_dim: int, in_channels: Optional[int] = None,
                 num_heads: Optional[int] = 1,
                 hidden_layers: Optional[list[int]] = None,
                 biases: Optional[Union[bool, Iterable[bool], str]] = True,
                 aggregation: Optional[str] = 'linear',
                 activation: Optional[str] = 'leakyrelu', activation_constant: Optional[float] = 0.01,
                 device=torch.device('cpu'), dtype=torch.float64, **kwargs):
        super(Transformer, self).__init__()

        if in_channels is None:
            assert aggregation != 'linear'

        if hidden_layers is None:
            hidden_layers = []

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = [in_dim] + hidden_layers + [out_dim]

        if isinstance(biases, bool):
            biases = [biases] * (len(self.layer_dims) - 1 + int(aggregation.lower() == 'linear'))
        elif biases == 'all_but_last':
            biases = [True] * (len(self.layer_dims) - 1 + int(aggregation.lower() == 'linear'))
            biases[-1] = False
        else:
            biases = list(biases)
            assert len(biases) == len(self.layer_dims) - 1 + int(aggregation.lower() == 'linear')

        layers = [SnEquivariantAttentionBlock(self.layer_dims[0], self.layer_dims[1], num_heads, bias=biases[0])]

        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[1:-1], self.layer_dims[2:])):
            layers.append(get_activation(activation, activation_constant))
            layers.append(SnEquivariantAttentionBlock(in_dim, out_dim, num_heads, bias=biases[i + 1]))

        self.layers = nn.Sequential(*layers).to(device=device, dtype=dtype)

        if aggregation.lower() == 'linear':
            self.agg = nn.Sequential(
                get_activation(activation, activation_constant),
                nn.Linear(in_channels, 1, bias=biases[-1], device=device, dtype=dtype))

        else:
            self.agg = get_aggregation(aggregation.lower(), -1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.agg(self.layers(x)).mean(dim=-1)

        return fx


if __name__ == '__main__':
    b, d, n = 10, 12, 15
    model = Transformer(d, 2, 4, hidden_layers=[12, 100, 32], device='cpu', aggregation='linear')

    X = torch.rand(b, d, n, dtype=torch.float64, device='cpu') * 10
    # for i in range(100):
    #     X = torch.rand(b, d, n, dtype=torch.float64, device='cpu') * 10
    #     rand_x = X[..., torch.randperm(n)]
    #     assert torch.allclose(model(X), model(rand_x))
    #     print(i)

    print(model(X).shape)
