from functools import partial
from types import FunctionType, MethodType
from typing import Union, Optional

import torch
from torch import Tensor
from torch import nn

from ansatz_utils import all_transpositions, ProjectiveSorting, alternation_separation, linear_wsop_sub_weights, \
    random_negative_permutation


class AsWeightedFrame(nn.Module):
    def __init__(self, in_dim: int, n_frames: int):
        super(AsWeightedFrame, self).__init__()
        self.projection_sorting = ProjectiveSorting(in_dim, n_frames)

    def forward(self,
                ws_function: Union[nn.Module, partial, FunctionType, MethodType],
                x: Tensor,
                an_invariant: bool = False) -> Tensor:

        weights, signs, sorted_x = self.projection_sorting(x)
        weights = alternation_separation(weights)
        if not an_invariant:
            framed_func = ((weights * signs) * ws_function(sorted_x)).sum(dim=1)
        else:
            neg_pem = random_negative_permutation(x.shape[-1], x.device)
            framed_func = torch.where(signs == 1,
                                      weights * ws_function(sorted_x, x),
                                      0) - torch.where(signs == -1,
                                                       weights * ws_function(sorted_x, x[..., neg_pem]),
                                                       0)
            framed_func = framed_func.sum(dim=1)
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
                 an_invariant: bool = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64):
        super(WeakStabilizeWeightedFrame, self).__init__()
        self.weighted_frame = AsWeightedFrame(in_dim, n_frames).to(device=device, dtype=dtype)
        self.transpositions = all_transpositions(in_channels, device=device)
        self.unstable_function = unstable_function.to(device, dtype)
        self.an_invariant = an_invariant

    def stable_forward(self, sorted_x: Tensor, x: Tensor = None) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.weighted_frame(
            self.stable_forward,
            x.to(device=self.weighted_frame.get_device, dtype=self.weighted_frame.get_dtype),
            self.an_invariant
        )


class NonLinearWeightedFrame(WeakStabilizeWeightedFrame):
    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: bool = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64):
        super(NonLinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, an_invariant,
                                                     device, dtype)

    def stable_forward(self, sorted_x: Tensor, x: Tensor = None) -> Tensor:
        if not self.an_invariant:
            sorted_x = sorted_x.unsqueeze(-3)
            f_x: Tensor = self.unstable_function(sorted_x)
            f_tx: Tensor = self.unstable_function(
                torch.take_along_dim(sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2), -1)
            ).clone()
            stable_func = f_x.sign() * torch.sqrt(0.5 * f_x.abs() * (f_x - f_tx).abs().min(dim=-1, keepdim=True)[0])
            stable_func = stable_func.sum(dim=-3)
        else:
            f_x: Tensor = self.unstable_function(x)
            f_tx: Tensor = self.unstable_function(x[..., self.transpositions[0]]).clone()
            stable_func = f_x.sign() * torch.sqrt(0.5 * f_x.abs() * (f_x - f_tx).abs())
            stable_func = stable_func.unsqueeze(1)
        return stable_func


class LinearWeightedFrame(WeakStabilizeWeightedFrame):
    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: bool = False,
                 p_norm: Union[str, int] = 'fro',
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64):
        super(LinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, an_invariant,
                                                  device, dtype)
        self.p_norm = p_norm

    def stable_forward(self, sorted_x: Tensor, x: Tensor = None) -> Tensor:
        sorted_x = sorted_x.unsqueeze(-3)
        weight_hat = torch.norm(
            sorted_x - torch.take_along_dim(sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2), -1),
            dim=[-2, -1], p=self.p_norm
        )

        if not self.an_invariant:
            weight_hat = linear_wsop_sub_weights(weight_hat).unsqueeze(-1)
            f_x: Tensor = self.unstable_function(sorted_x)
            f_tx: Tensor = self.unstable_function(
                torch.take_along_dim(sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2), -1)
            ).clone()
            stable_func = f_x.squeeze(-2) - (weight_hat * (f_x + f_tx)).sum(dim=-2)

        else:
            f_x: Tensor = self.unstable_function(x).unsqueeze(1)
            f_tx: Tensor = self.unstable_function(x[..., self.transpositions[0]]).clone().unsqueeze(1)
            stable_func = f_x - (weight_hat * (f_x + f_tx)).sum(dim=-1, keepdim=True)

        return stable_func


if __name__ == '__main__':
    from neural_networks import MLP

    b, d, n = 10, 3, 13
    F = nn.Sequential(nn.Flatten(-2, -1), MLP(n * d, 1, [20, 20, 20]))
    X = torch.rand(b, d, n, dtype=torch.float64)

    X[0, :, 0] = X[0, :, 3]

    SF = NonLinearWeightedFrame(F, d, n, 130, an_invariant=True)

    print(SF(X))
