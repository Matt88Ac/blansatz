from typing import Optional

import torch
from torch import nn

from ansatzes import BiLipschitzAntiSymmetricModel
from ansatz_utils import get_model, LinearWeightedFrame, NonLinearWeightedFrame


class AfaNetModel(nn.Module):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 frame_name: Optional[str] = 'nonlinear',
                 model_name: Optional[str] = 'mlp',
                 an_invariant: Optional[bool] = False,
                 flatten: Optional[bool] = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **model_kwargs):
        """
        Args:
            in_dim:
            in_channels:
            out_dim:
            embedding_dim:
            frame_name:
            model_name:
            an_invariant:
            flatten:
            device:
            dtype:
            **model_kwargs:
        """
        super(AfaNetModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = (2 * in_dim * in_channels) + 1

        if an_invariant:
            unstable_model = BiLipschitzAntiSymmetricModel(in_dim, in_channels, out_dim, embedding_dim, model_name,
                                                           device, dtype, **model_kwargs)

        else:
            model_kwargs['in_dim'] = in_dim
            model_kwargs['out_dim'] = out_dim
            model_kwargs['device'] = device
            model_kwargs['dtype'] = dtype
            if flatten:
                model_kwargs['in_dim'] *= in_channels
                unstable_model = nn.Sequential(nn.Flatten(-2, -1), get_model(model_name, **model_kwargs))
            else:
                unstable_model = get_model(model_name, **model_kwargs)

        if frame_name == 'linear':
            self.frames = LinearWeightedFrame(unstable_model, in_dim, in_channels, embedding_dim, an_invariant,
                                              device, dtype)

        elif frame_name == 'nonlinear':
            self.frames = NonLinearWeightedFrame(unstable_model, in_dim, in_channels, embedding_dim, an_invariant,
                                                 device, dtype)

        else:
            raise NotImplementedError("Only 'linear' and 'nonlinear' are accepted for 'frame_name' parameter.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frames(x)


if __name__ == '__main__':
    b, d, n = 10, 3, 15
    model = AfaNetModel(d, n, 4, hidden_layers=[5, 100, 3], device='cpu', model_name='mlp', an_invariant=True)

    print(model)
    exit(0)

    temp = torch.rand(b, d, n, dtype=torch.float64, device='cpu') * 1000

    print(model(temp))
