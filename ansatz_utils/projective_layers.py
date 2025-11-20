from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from permutation_actions import vectorized_permutation_sign


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
    def weight(self):
        return self.spatial_projector





if __name__ == '__main__':
    b, n, d = 4, 10, 3
    X = torch.rand(b, d, n)
    p_l = ProjectiveSorting(d, 120)

    ax, s, nx = p_l(X)
    print(torch.diff(ax, dim=-1))
