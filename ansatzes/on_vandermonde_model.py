from typing import Optional

import torch
from torch import nn

from ansatz_utils import get_model, vandermonde_determinant, uniform_sphere_point


class OnVandermondeModel(nn.Module):
    """
        Implementation of the discontinuous ansatz from https://arxiv.org/html/2402.15167v2
    """
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 trainable_weights: Optional[bool] = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **model_kwargs):
        """
        Args:
            in_dim:
            in_channels:
            out_dim:
            embedding_dim:
            device:
            dtype:
            **model_kwargs:
        """
        super(OnVandermondeModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = (in_dim * in_channels) + 1

        self.spatial_projector = nn.Parameter(uniform_sphere_point(embedding_dim, in_dim).to(device, dtype),
                                              requires_grad=trainable_weights)

        model_kwargs['in_dim'] = in_dim
        model_kwargs['out_dim'] = out_dim * embedding_dim
        model_kwargs['device'] = device
        model_kwargs['dtype'] = dtype

        model_kwargs['new_dim'] = False
        model_kwargs['swap_last_axes'] = True

        self.out_dim = out_dim

        self.model = get_model('ds', **model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *shape, d, n = x.shape
        N = n * (n - 1) / 2

        proj_x = torch.einsum('md,...dn->...mn', self.spatial_projector, x.clone())
        weights = torch.float_power(
            torch.norm(vandermonde_determinant(proj_x.clone()), p=2, dim=-2, keepdim=True).clone(),
            1 / N)
        proj_x = proj_x / (weights + (weights == 0).to(dtype=weights.dtype))

        fx = (self.model(x).reshape(*shape, self.embedding_dim, self.out_dim) * vandermonde_determinant(proj_x)).sum(
            dim=-2)

        return fx

    @property
    def embedding_dim(self) -> int:
        return self.spatial_projector.shape[0]


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = OnVandermondeModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu', aggregation='max')

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu') * 10

    print(model(temp).shape)
