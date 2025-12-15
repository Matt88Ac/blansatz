from typing import Optional

import torch
from torch import nn

from ansatz_utils import get_model, random_negative_permutation, AnInvariantEmbedding


class BiLipschitzAntiSymmetricModel(nn.Module):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
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
        super(BiLipschitzAntiSymmetricModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = (2 * in_dim * in_channels) + 1

        model_kwargs['in_dim'] = embedding_dim
        model_kwargs['out_dim'] = out_dim
        model_kwargs['device'] = device
        model_kwargs['dtype'] = dtype
        if model_name in ['ds', 'deepsets', 'deepset']:
            model_kwargs['new_dim'] = True
            model_kwargs['in_dim'] = 1

        if 'biases' in model_kwargs.keys():
            if isinstance(model_kwargs['biases'], bool):
                if model_kwargs['biases']:
                    model_kwargs['biases'] = 'all_but_last'
        self.tau = random_negative_permutation(in_channels, device)
        self.mu = AnInvariantEmbedding(in_dim, in_channels, embedding_dim).to(device=device, dtype=dtype)
        self.model = get_model(model_name, **model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model(self.mu(x)) - self.model(self.mu(x[..., self.tau]))) / 2


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = BiLipschitzAntiSymmetricModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu', model_name='ds',
                                          aggregation='max')

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu')*1000

    print(model(temp))
