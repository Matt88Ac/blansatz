from typing import Optional, Union

import torch
from torch import nn

from ansatz_utils import MLP, SelfDiff, random_negative_permutation  # , permutation_sign, vectorized_permutation_sign


class BiLipschitzPsi(nn.Module):
    """ The class constructs a layer of Alternation-Invariant functions of the form ψ(X; a, b) = [sort(a*X), Q(a*X)]*b
        where Q(X*a) = prod{sign(a*(X_j - X_i)): i < j} *  min{a*(X_j - X_i): i < j}.
        The input is (batched) d x n matrices, and the output is (a batch of) m scalars.

        Attributes:
            spatial_projector (Parameter):
                a matrix of m d-dimensional parameters.
            channel_projector (Parameter):
                a matrix of m (n+1)-dimensional parameters.
            max_pool:
                torch AdaptiveMaxPool1d.
    """
    def __init__(self, in_dim: int, in_channels: int, out_dim: int):
        """
        Args:
            in_dim (int):
                spatial dimension (denoted above by d).
            in_channels (int):
                 number of particles in a system (denoted above by n).
            out_dim (int):
                number of functions ψ to define (denoted above by m).

        Examples:
            >>> b, d, n = 10, 3, 5
            >>> X = torch.randn(b, d, n)
            >>> m = 2 * d * n + 1
            >>> psi = BiLipschitzPsi(d, n, m)
            >>> psi(X).shape[0], psi(X).shape[1]
            (10, 31)
            >>> positive_permutation = torch.tensor([1, 0, 2, 4, 3], dtype=torch.int)
            >>> alt_X = X[..., positive_permutation]
            >>> torch.all(psi(X) == psi(alt_X)).item()
            True
            >>> negative_permutation = random_negative_permutation(n)
            >>> permuted_X = X[..., negative_permutation]
            >>> torch.all(psi(X) == psi(permuted_X)).item()
            False
        """
        super(BiLipschitzPsi, self).__init__()

        self.spatial_projector = nn.Parameter(torch.empty(out_dim, in_dim))
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.channel_projector = nn.Parameter(torch.empty(out_dim, in_channels + 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spatial_projector)
        nn.init.xavier_uniform_(self.channel_projector)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: input batch of tensors y = [x_1, ..., x_n]; such that x_i is a real [..., d] tensor.

        Returns:
            Ψ(X; A, B) as described above.
        """
        ax = torch.einsum('md,...dn->...mn', self.spatial_projector, X)  # [a * x_1, ..., a * x_n]
        diff_ax = SelfDiff.apply(ax.clone())  # [ a * (x_j - x_i): 1 <= i < j <= n ]
        sort_ax, argsort_ax = torch.sort(ax, dim=-1)
        fx = torch.cat(
            [sort_ax, - torch.sign(diff_ax).prod(dim=-1, keepdim=True) * self.max_pool(-diff_ax.abs())], dim=-1
        )
        # The above line computes [sort(x_1 * a_l, ..., x_n * a_l), prod(sign(a * (x_j - x_i)) * min{|a * (x_j - x_i)|}]
        # for all 1 <= l <= m, that is F(x; a_l).
        # Todo: Implement the computation using the sign of the argsort. Currently unavailable due to complexities in batched computations.
        psi = torch.einsum('...mn,mn->...m', fx, self.channel_projector)  # b_l * F(x; a_l) for all 1 <= l <= m
        return psi


class BiLipschitzAntiSymmetricModel(nn.Module):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, psi_dim: Optional[int] = None,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **mlp_kwargs):
        """
        Args:
            in_dim:
            in_channels:
            out_dim:
            psi_dim:
            device:
            dtype:
            **mlp_kwargs:
        """
        super(BiLipschitzAntiSymmetricModel, self).__init__()
        if psi_dim is None:
            psi_dim = (2 * in_dim * in_channels) + 1

        mlp_kwargs['in_dim'] = psi_dim
        mlp_kwargs['out_dim'] = out_dim
        mlp_kwargs['device'] = device
        mlp_kwargs['dtype'] = dtype
        self.tau = random_negative_permutation(in_channels, device)
        self.psi = BiLipschitzPsi(in_dim, in_channels, psi_dim).to(device=device, dtype=dtype)
        self.mlp = MLP(**mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.mlp(self.psi(x)) - self.mlp(self.psi(x[..., self.tau])))


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = BiLipschitzAntiSymmetricModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu')

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu')*1000

    print(model(temp).shape)
