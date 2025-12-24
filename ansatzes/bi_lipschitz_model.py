from typing import Optional

import torch
from torch import nn

from ansatz_utils import get_model, random_negative_permutation, AnInvariantEmbedding


class BiLipschitzAntiSymmetricModel(nn.Module):
    """
    Anti-Symmetric model, that relies on a bi-Lipschitz An-invariant embedding followed by a standard model, as described in our work.
    
    Attributes:
        tau (Tensor):
            A random negative permutation.
        mu (AnInvariantEmbedding):
            A bi-Lipschitz, An-invariant embedding module.
        model (nn.Module):
            A standard model (e.g., MLP, DeepSets) that processes the embedded input.
    """
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **model_kwargs):
        """
        Args:
            in_dim (int):
                The input dimension.
            in_channels (int):
                The number of input channels.
            out_dim (int):
                The output dimension.
            embedding_dim (Optional[int]):
                The dimension of the embedding. If None, it is computed as (2 * in_dim * in_channels) + 1.
            model_name (Optional[str]):
                The name of the model to use. Default is 'mlp'.
            device (Optional):
                The device to use. Default is CPU.
            dtype (Optional):
                The data type to use. Default is torch.float64.
            model_kwargs:
                Additional keyword arguments for the model.
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
