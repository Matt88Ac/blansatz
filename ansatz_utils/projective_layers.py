from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from ansatz_utils import vectorized_permutation_sign, AllDifferences


# @torch.jit.script
def alternation_separation(sorted_x: Tensor) -> Tensor:
    """
    Args:
        sorted_x (torch.Tensor): A torch tensor such that i<j => x_i <= x_j along the last axis.

    Returns:
        min{(x_j - x_i): i < j} along the last axis of the input.
    """

    return torch.diff(sorted_x, dim=-1).min(dim=-1, keepdim=True)[0]


class ProjectiveSorting(nn.Module):
    """ 
    The class constructs a layer of Alternation-Invariant functions of the form [sort(a*X), Q(a*X)]
    where Q(X*a) = prod{sign(a*(X_j - X_i)): i < j} *  min{a*(X_j - X_i): i < j}.
    The input is (batched) d x n matrices, and the output is (a batch of) m x n+1 matrices.

    Args:
        in_dim (int):
            spatial dimension (denoted above by d).
        projection_dim (int):
            number of projecting vectors to define (denoted above by m).

    Attributes:
        spatial_projector (Parameter):
            a matrix of m d-dimensional parameters.

    Examples:
        >>> b, d, n = 10, 3, 5
        >>> X = torch.randn(b, d, n)
        >>> m = 2 * d * n + 1
        >>> ps = ProjectiveSorting(d, m)
        >>> diff_proj_x, sorted_proj_x, sorted_x = ps(X)
    """

    def __init__(self, in_dim: int, projection_dim: int):
        super(ProjectiveSorting, self).__init__()
        self.spatial_projector = nn.Parameter(torch.empty(projection_dim, in_dim), requires_grad=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spatial_projector, gain=nn.init.calculate_gain('relu'))
        while torch.linalg.matrix_rank(self.spatial_projector) != self.in_dim:
            nn.init.xavier_uniform_(self.spatial_projector, gain=nn.init.calculate_gain('relu'))


    def forward(self, x: Tensor, sorted_not_proj_needed: bool) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): a tensor of shape [b, d, n] or [d, n].
            sorted_not_proj_needed (bool): If true,

        Returns: the following three tensors:
            - diff_proj_x: tensor of shape [b, m, n*(n-1)/2] or [m, n*(n-1)/2] containing all differences a*(X_j - X_i): i < j for each projecting vector a.
            - sorted_proj_x: tensor of shape [b, m, n] or [m, n] of the projections a*X sorted along the last axis.
            - out_x: tensor of shape [b, m, n] or [m, n] containing the input x permuted according to the sorting of the projections a*X.
        """
        projected_x = nn.functional.linear(x.swapaxes(-2, -1), self.spatial_projector).swapaxes(-2, -1)
        diff_proj_x = AllDifferences.apply(projected_x)
        if sorted_not_proj_needed:
            return diff_proj_x, torch.sort(projected_x, stable=True, dim=-1)[0]
        else:
            permutations = torch.argsort(projected_x, stable=True, dim=-1)
            # permutations = vectorized_permutation_sign(permutations)
            return diff_proj_x, permutations

    @property
    def weights(self):
        return self.spatial_projector

    @property
    def embedding_dim(self):
        return self.spatial_projector.shape[0]

    @property
    def in_dim(self):
        return self.spatial_projector.shape[-1]


class AnInvariantEmbedding(nn.Module):
    """ 
    The class constructs a layer of Alternation-Invariant functions of the form ψ(X; a, b) = [sort(a*X), Q(a*X)]*b
    where Q(X*a) = prod{sign(a*(X_j - X_i)): i < j} *  min{a*(X_j - X_i): i < j}.
    The input is (batched) d x n matrices, and the output is (a batch of) m scalars.

    Args:
        in_dim (int):
            spatial dimension (denoted above by d).
        in_channels (int):
             number of points to consider (denoted above by n).
        embedding_dim (int):
            number of projecting vectors to define (denoted above by m).

    Attributes:
        channel_projection (Parameter):
            a matrix of m (n+1)-dimensional parameters.

    Examples:
        >>> b, d, n = 10, 3, 5
        >>> X = torch.randn(b, d, n)
        >>> m = 2 * d * n + 1
        >>> an_embedding = AnInvariantEmbedding(d, n, m)
        >>> projected_x = an_embedding(X)
    """

    def __init__(self, in_dim: int, in_channels: int, embedding_dim: int):
        super(AnInvariantEmbedding, self).__init__()

        self.in_channels = in_channels
        self.channel_projection = nn.Parameter(torch.empty(embedding_dim, in_channels + 1))
        self.projection_sorting = ProjectiveSorting(in_dim, embedding_dim)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.channel_projection)

    def forward(self, x: Tensor) -> Tensor:
        # projected_x, signs, sorted_x = self.projection_sorting(x)
        diff_proj_x, sorted_proj_x = self.projection_sorting(x, True)
        out_x = torch.cat([sorted_proj_x,
                           diff_proj_x.sign().prod(dim=-1, keepdim=True) * diff_proj_x.abs().min(dim=-1, keepdim=True)[
                               0]], dim=-1)

        size = [1] * (out_x.dim() - 2)
        return (self.channel_projection.contiguous().view(*size,
                                                          self.projection_sorting.embedding_dim,
                                                          self.in_channels + 1) * out_x).sum(dim=-1)


if __name__ == '__main__':
    from permutation_actions import random_positive_permutation, random_negative_permutation

    b, n, d = 4, 10, 3
    X = torch.rand(b, d, n)

    p_l = AnInvariantEmbedding(d, n, 120)
    
    for i in range(100):
        rand_x = X[..., random_positive_permutation(n)]
        assert torch.allclose(p_l(X), p_l(rand_x))
        rand_x = X[..., random_negative_permutation(n)]
        assert torch.all(torch.any(p_l(X) != p_l(rand_x), dim=-1))
