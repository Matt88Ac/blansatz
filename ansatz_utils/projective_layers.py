from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from .permutation_actions import vectorized_permutation_sign


def alternation_separation(sorted_x: Tensor) -> Tensor:
    return torch.diff(sorted_x, dim=-1).min(dim=-1, keepdim=True)[0]


class ProjectiveSorting(nn.Module):
    """ The class constructs a layer of Alternation-Invariant functions of the form ψ(X; a, b) = [sort(a*X), Q(a*X)]*b
        where Q(X*a) = prod{sign(a*(X_j - X_i)): i < j} *  min{a*(X_j - X_i): i < j}.
        The input is (batched) d x n matrices, and the output is (a batch of) m scalars.

        Attributes:
            spatial_projector (Parameter):
                a matrix of m d-dimensional parameters.
    """

    def __init__(self, in_dim: int, projection_dim: int):
        """
        Args:
            in_dim (int):
                spatial dimension (denoted above by d).
            projection_dim (int):
                number of projecting vectors to define (denoted above by m).

        Examples:
            >>> b, d, n = 10, 3, 5
            >>> X = torch.randn(b, d, n)
            >>> m = 2 * d * n + 1
            >>> ps = ProjectiveSorting(d, m)
            >>> projected_x, signs, sorted_x = ps(X)
        """
        super(ProjectiveSorting, self).__init__()
        self.spatial_projector = nn.Parameter(torch.empty(projection_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spatial_projector, gain=self.spatial_projector.shape[-1])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: a tensor of shape [b, d, n] or [d, n].

        Returns: the following three tensors:
            - sorted projection of x.
            - the signs of the permutations which the projection of x is sorted according to.
            - the permuted input x, according to each of the permutations mentioned in the latter.
        """
        projected_x, permutations = torch.sort(torch.einsum('md,...dn->...mn', self.spatial_projector, x),
                                               stable=True, dim=-1)

        out_x = x.clone().unsqueeze(1)
        permutations: Tensor = permutations.unsqueeze(-2)
        out_x = torch.take_along_dim(out_x.requires_grad_(True), permutations, -1)

        permutations = vectorized_permutation_sign(permutations)

        """
        for i in range(len(signs)):
            for j in range(signs.size(1)):
                assert signs[i, j] == permutation_sign(permutations[i, j])
        """

        return projected_x, permutations, out_x

    @property
    def weights(self):
        return self.spatial_projector


class AnInvariantEmbedding(nn.Module):
    """ The class constructs a layer of Alternation-Invariant functions of the form ψ(X; a, b) = [sort(a*X), Q(a*X)]*b
        where Q(X*a) = prod{sign(a*(X_j - X_i)): i < j} *  min{a*(X_j - X_i): i < j}.
        The input is (batched) d x n matrices, and the output is (a batch of) m scalars.

        Attributes:
            channel_projection (Parameter):
                a matrix of m (n+1)-dimensional parameters.
    """

    def __init__(self, in_dim: int, in_channels: int, embedding_dim: int):
        """
        Args:
            in_dim (int):
                spatial dimension (denoted above by d).
            in_channels (int):
                 number of points to consider (denoted above by n).
            embedding_dim (int):
                number of projecting vectors to define (denoted above by m).

        Examples:
            >>> b, d, n = 10, 3, 5
            >>> X = torch.randn(b, d, n)
            >>> m = 2 * d * n + 1
            >>> an_embedding = AnInvariantEmbedding(d, n, m)
            >>> projected_x = an_embedding(X)
        """
        super(AnInvariantEmbedding, self).__init__()
        self.channel_projection = nn.Parameter(torch.empty(embedding_dim, in_channels + 1))
        self.projection_sorting = ProjectiveSorting(in_dim, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.channel_projection)

    def forward(self, x: Tensor) -> Tensor:
        projected_x, signs, sorted_x = self.projection_sorting(x)
        del sorted_x
        projected_x = torch.cat([projected_x, signs * alternation_separation(projected_x)], dim=-1)
        return torch.einsum('...mn,mn->...m', projected_x, self.channel_projection)


class AsFrameWeights(nn.Module):
    def __init__(self, in_dim: int, n_frames: int):
        super(AsFrameWeights, self).__init__()
        self.projection_sorting = ProjectiveSorting(in_dim, n_frames)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        projected_x, signs, sorted_x = self.projection_sorting(x)

        weights = alternation_separation(projected_x)
        s_weights = weights.clone().sum(dim=1, keepdim=True)
        s_weights = torch.where(s_weights > 0, 1 / s_weights, s_weights)

        return signs * s_weights * weights, sorted_x

    @property
    def weights(self):
        return self.projection_sorting.weights


if __name__ == '__main__':
    from permutation_actions import random_positive_permutation, random_negative_permutation

    b, n, d = 4, 10, 3
    X = torch.rand(b, d, n)

    _x1 = X[1, :, 1]

    X[1, :, 6] = _x1

    fw = AsFrameWeights(d, 120)
    w, _x = fw(X)
    print(w.abs().sum(dim=1))
    exit(0)

    p_l = AnInvariantEmbedding(d, n, 120)

    for i in range(100):
        rand_x = X[..., random_positive_permutation(n)]
        assert torch.allclose(p_l(X), p_l(rand_x))
        rand_x = X[..., random_negative_permutation(n)]
        assert torch.all(torch.any(p_l(X) != p_l(rand_x), dim=-1))
