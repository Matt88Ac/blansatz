from typing import Optional

import torch
from torch import nn

from ansatz_utils import get_model, vandermonde_determinant, uniform_sphere_sampling


class OnVandermondeModel(nn.Module):
    """
    Implementation of the discontinuous ansatz from https://arxiv.org/html/2402.15167v2. 
    In our implementation, we favour numerical stability over runtime performance.

    Args:
        in_dim (int):
            The input dimension.
        in_channels (int):
            The number of input channels.
        out_dim (int):
            The output dimension.
        embedding_dim (int, optional):
            The dimension of the embedding. If None, it is computed as (in_dim * in_channels) + 1.
        model_name (str, optional):
            The name of the model to use. Must be an Sn-invariant model. Default is 'ds'.
        trainable_weights (bool, optional):
            Whether the spatial projector weights are trainable. Default is False.
        single_model (bool, optional):
            Whether to use a single Sn-invariant model or multiple Sn-invariant models, one for each weight. Default is False.
        device (Optional):
            The device to use. Default is 'cpu'.
        dtype (Optional):
            The data type to use. Default is torch.float64.
        model_kwargs:
            Additional keyword arguments to pass to the Sn-invariant model.

    Attributes:
        spatial_projector (nn.Parameter):
            The spatial projector used to project the input data.
        out_dim (int):
            The output dimension.
        single_model (bool):
            Whether to use a single Sn-invariant model or multiple Sn-invariant models, one for each weight.
        model (Optional[nn.Module]):
            The Sn-invariant model, used when single_model is True.
        model_list (Optional[nn.ModuleList]):
            The list of Sn-invariant models, used when single_model is False.

    Example:
        >>> model = OnVandermondeModel(3, 5, 2, hidden_layers=[10, 20], device='cpu', aggregation='max', single_model=True)
        >>> x = torch.rand(4, 3, 5)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([4, 2])
    """
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'ds',
                 trainable_weights: Optional[bool] = False,
                 single_model: Optional[bool] = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **model_kwargs):
        super(OnVandermondeModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = (in_dim * in_channels) + 1

        self.spatial_projector = nn.Parameter(uniform_sphere_sampling(embedding_dim, in_dim).to(device, dtype),
                                              requires_grad=trainable_weights)

        model_kwargs['in_dim'] = in_dim
        model_kwargs['device'] = device
        model_kwargs['dtype'] = dtype

        model_kwargs['new_dim'] = False
        model_kwargs['swap_last_axes'] = True

        self.out_dim = out_dim
        self.single_model = single_model
        if single_model:
            model_kwargs['out_dim'] = out_dim * embedding_dim
            self.model = get_model('ds', **model_kwargs)
        else:
            model_kwargs['out_dim'] = out_dim
            self.model_list = nn.ModuleList([get_model('ds', **model_kwargs) for i in range(embedding_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *shape, d, n = x.shape
        N = n * (n - 1) / 2
        proj_x = nn.functional.linear(x.clone().swapaxes(-2, -1), self.spatial_projector).swapaxes(-2, -1)

        weights = torch.float_power(
            torch.norm(vandermonde_determinant(proj_x), p=2, dim=-2, keepdim=True),
            1 / N).clone()
        proj_x = proj_x / (weights + (weights == 0).to(dtype=weights.dtype))

        if self.single_model:
            fx = self.model(x).reshape(*shape, self.embedding_dim, self.out_dim)
        else:
            fx = torch.cat([model(x).unsqueeze(-2) for model in self.model_list], dim=-2)
        fx = (fx * vandermonde_determinant(proj_x)).sum(dim=-2)
        return fx

    @property
    def embedding_dim(self) -> int:
        return self.spatial_projector.shape[0]


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = OnVandermondeModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu', aggregation='max', single_model=False)

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu') * 10

    print(model(temp).shape)
