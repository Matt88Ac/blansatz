from typing import Optional, Union

import torch
from torch import nn

from ansatz_utils import MLP, random_negative_permutation, AnInvariantEmbedding


class BiLipschitzAntiSymmetricModel(nn.Module):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **mlp_kwargs):
        """
        Args:
            in_dim:
            in_channels:
            out_dim:
            embedding_dim:
            device:
            dtype:
            **mlp_kwargs:
        """
        super(BiLipschitzAntiSymmetricModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = (2 * in_dim * in_channels) + 1

        mlp_kwargs['in_dim'] = embedding_dim
        mlp_kwargs['out_dim'] = out_dim
        mlp_kwargs['device'] = device
        mlp_kwargs['dtype'] = dtype
        self.tau = random_negative_permutation(in_channels, device)
        self.psi = AnInvariantEmbedding(in_dim, in_channels, embedding_dim).to(device=device, dtype=dtype)
        self.mlp = MLP(**mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.mlp(self.psi(x)) - self.mlp(self.psi(x[..., self.tau])))


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = BiLipschitzAntiSymmetricModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu')

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu')*1000

    print(model(temp).shape)
