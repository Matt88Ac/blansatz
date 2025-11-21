from functools import partial
from typing import Union, Optional, Tuple
from types import FunctionType, MethodType

import torch
from torch import Tensor
from torch import nn

from permutation_actions import all_transpositions
from projective_layers import ProjectiveSorting, alternation_separation


class AsWeightedFrame(nn.Module):
    def __init__(self, in_dim: int, n_frames: int):
        super(AsWeightedFrame, self).__init__()
        self.projection_sorting = ProjectiveSorting(in_dim, n_frames)

    def forward(self, ws_function: Union[nn.Module, partial, FunctionType, MethodType], x: Tensor) -> Tensor:
        weights, signs, sorted_x = self.projection_sorting(x)
        weights = alternation_separation(weights)

        framed_func = ((weights * signs) * ws_function(sorted_x)).sum(dim=1)
        framed_func = torch.where(framed_func != 0, framed_func / weights.sum(dim=1), framed_func)

        return framed_func

    @property
    def weights(self):
        return self.projection_sorting.weights

    @property
    def get_device(self):
        return self.weights.device

    @property
    def get_dtype(self):
        return self.weights.dtype


class WeakStabilizeWeightedFrame(nn.Module):
    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64):
        super(WeakStabilizeWeightedFrame, self).__init__()
        self.weighted_frame = AsWeightedFrame(in_dim, n_frames).to(device=device, dtype=dtype)
        self.transpositions = all_transpositions(in_channels, device=device)
        self.unstable_function = unstable_function.to(device, dtype)

    def stable_forward(self, sorted_x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.weighted_frame(
            self.stable_forward,
            x.to(device=self.weighted_frame.get_device, dtype=self.weighted_frame.get_dtype)
        )


class NonLinearWeightedFrame(WeakStabilizeWeightedFrame):
    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64):
        super(NonLinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, device, dtype)

    def stable_forward(self, sorted_x: Tensor) -> Tensor:
        sorted_x = sorted_x.unsqueeze(-3)
        f_x: Tensor = self.unstable_function(sorted_x)
        f_tx: Tensor = self.unstable_function(
            torch.take_along_dim(sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2), -1)
        ).clone()

        stable_func = f_x.sign() * torch.sqrt(0.5 * f_x.abs() * (f_x - f_tx).abs().min(dim=-1, keepdim=True)[0])
        return stable_func.sum(dim=-3)


if __name__ == '__main__':
    transp = all_transpositions(6)
    b, d, n = 10, 3, 6
    temp = torch.rand(b, d, n, device='cpu') * 1000

    emb = ProjectiveSorting(d, 120)

    w, ws, sx = emb(temp)
    sx = sx.unsqueeze(-3)
    transp = transp.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
    nsx = torch.take_along_dim(sx, transp, -1)
    print(sx.shape, transp.shape, alternation_separation(w).shape)

    print(nsx[0, 0, 1, 0])
    print(sx[0, 0, 0, 0])
