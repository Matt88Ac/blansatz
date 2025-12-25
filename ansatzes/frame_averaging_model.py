from typing import Optional

import torch
from torch import nn

from ansatzes import BiLipschitzAntiSymmetricModel
from ansatz_utils import get_model, LinearWeightedFrame, NonLinearWeightedFrame


class AfaNetModel(nn.Module):
    """
    AfaNet model that uses frame averaging with either linear or nonlinear WS-Operator, over an unstable model, as described in our work.

    Args:
        in_dim (int):
            The input dimension.
        in_channels (int):
            The number of input channels.
        out_dim (int):
            The output dimension.
        embedding_dim (int, optional):
            The dimension of the embedding. If None, it is computed as (2 * in_dim * in_channels) + 1.
        frame_name (str, optional):
            The name of the frame averaging module to use. Must be either 'linear' or 'nonlinear'. Default is 'nonlinear'.
        model_name (str, optional):
            The name of the unstable model to use. Default is 'mlp'.
        an_invariant (bool, optional):
            Whether to use an An-invariant unstable model. Default is False.
        flatten (bool, optional):
            Whether to flatten the input before passing it to the unstable model (only if an_invariant is False). Default is False.
        device (Optional):
            The device to use. Default is 'cpu'.
        dtype (Optional):
            The data type to use. Default is torch.float64.
        model_kwargs:
            Additional keyword arguments for the unstable model.

    Attributes:
        frames (nn.Module):
            A weakly-stabilizing frame averaging module (either linear or nonlinear) that processes the input using the unstable model.
    """
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 frame_name: Optional[str] = 'nonlinear',
                 model_name: Optional[str] = 'mlp',
                 an_invariant: Optional[bool] = False,
                 flatten: Optional[bool] = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, **model_kwargs):
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
            model_kwargs['in_channels'] = in_channels
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
