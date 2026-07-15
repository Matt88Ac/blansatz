from functools import partial
from types import FunctionType, MethodType
from typing import Union, Optional

import torch
from torch import Tensor
from torch import nn

from ansatz_utils import all_transpositions, ProjectiveSorting, random_transposition, permute_ij


class AsWeightedFrame(nn.Module):
    """
    This class implements a weighted frame averaging of a given function over a set of projective frames, as described in our work.

    Args:
        in_dim (int):
            The input dimension of the function to be framed.
        in_channels (int):
             The number of input channels.
        n_frames (int):
            The number of frames to use.

    Attributes:
        projection_sorting (ProjectiveSorting): 
            An instance of the ProjectiveSorting class to compute weights and signs for the frames.
    """

    def __init__(self, in_dim: int, in_channels: int, n_frames: int):
        """
        
        """
        super(AsWeightedFrame, self).__init__()
        self.in_channels = in_channels
        self.projection_sorting = ProjectiveSorting(in_dim, n_frames)

    def forward(self,
                ws_function: Union[nn.Module, partial, FunctionType, MethodType],
                x: Tensor,
                an_invariant: bool = False) -> Tensor:

        """
        Args: 
            ws_function (Union[nn.Module, partial, FunctionType, MethodType]):
                The weakly-stable function to be averaged over the frames.
            x (Tensor):
                The input tensor.
            an_invariant (bool):
                Whether the function is An-invariant or not. Default is False.
        Returns:
            Tensor: The weighted frame averaged output.
        """

        diff_proj_x, permutations = self.projection_sorting(x, False)
        signs = diff_proj_x.sign().prod(dim=-1, keepdim=True)
        weights = diff_proj_x.abs().min(dim=-1, keepdim=True)[0]

        if not an_invariant:
            framed_func = ((weights * signs) * ws_function(x, permutations)).sum(dim=1)
        else:
            framed_func = torch.where(signs == 1,
                                      weights * ws_function(x, None),
                                      0) - torch.where(signs == -1,
                                                       weights * ws_function(permute_ij(x, 0, 1), None),
                                                       0)
            framed_func = framed_func.sum(dim=1)

        weights = weights.sum(dim=1)
        framed_func = torch.where(
            framed_func == 0.0, #torch.isclose(framed_func, torch.zeros_like(framed_func, requires_grad=False), atol=1e-20),
            framed_func * 0.0,
            framed_func / weights,
        )

        return framed_func

    @property
    def weights(self):
        return self.projection_sorting.weights

    @property
    def in_dim(self):
        return self.projection_sorting.in_dim

    @property
    def get_device(self):
        return self.weights.device

    @property
    def get_dtype(self):
        return self.weights.dtype


class WeakStabilizeWeightedFrame(nn.Module):
    """
    This class implements a general framework for weakly-stabilizing weighted frame averaging of a given unstable function.
    Args:
         unstable_function (nn.Module):
             The unstable function to be stabilized.
         in_dim (int):
             The input dimension of the function.
         in_channels (int):
             The number of input channels.
         n_frames (int):
             The number of frames to use.
         an_invariant (bool, optional):
             Whether the function is An-invariant or not. Default is False.
         device (Optional):
             The device to use. Default is CPU.
         dtype (Optional):
             The data type to use. Default is torch.float64.

    Attributes:
        weighted_frame (AsWeightedFrame): 
            An instance of the AsWeightedFrame class to perform weighted frame averaging.
        transpositions (Tensor): 
            A tensor containing all (consecutive) transpositions for the given number of input channels.
        unstable_function (nn.Module): 
            The unstable function to be stabilized.
        an_invariant (bool): 
            Whether the function is An-invariant or not.

    Example:
        >>> b, d, n = 1, 3, 13
        >>> F = nn.Sequential(nn.Flatten(-2, -1), MLP(n * d, 1, [20, 20, 20]))
        >>> X = torch.rand(b, d, n, dtype=torch.float64)
        >>> X[0, :, 0] = X[0, :, 3]
        >>> SF = NonLinearWeightedFrame(F, d, n, 130, an_invariant=True)
        >>> SF(X)
        tensor([[0.0]], dtype=torch.float64)
    """

    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: bool = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, *args, **kwargs):
        super(WeakStabilizeWeightedFrame, self).__init__()
        self.weighted_frame = AsWeightedFrame(in_dim, in_channels, n_frames).to(device=device, dtype=dtype)
        self.transpositions = nn.Parameter(all_transpositions(in_channels, device=device), requires_grad=False)
        self.identity = nn.Parameter(torch.arange(in_channels, device=device, dtype=torch.int64).unsqueeze(0), requires_grad=False)
        self.unstable_function = unstable_function.to(device, dtype)
        self.an_invariant = an_invariant

    def stable_forward(self, x: Tensor, permutations: Tensor = None) -> Tensor:
        """
        This method should be implemented in subclasses to define the weakly-stabilizing transformation.
        Args:
            x (Tensor):
                The input tensor.
            permutations (Tensor):
                The frame permutations given by projective sorting (not used if the function is An-invariant).
        Returns:
            Tensor: The weakly-stabilized output.
        """
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                The input tensor.
        Returns:
            Tensor: The weakly-stabilized weighted frame averaged output.
        """
        return self.weighted_frame(
            self.stable_forward,
            x,
            self.an_invariant
        )

    @property
    def in_channels(self):
        return self.weighted_frame.in_channels

    @property
    def in_dim(self):
        return self.weighted_frame.in_dim

    @property
    def get_device(self):
        return self.weighted_frame.get_device
    @property
    def get_dtype(self):
        return self.weighted_frame.get_dtype



class NonLinearWeightedFrame(WeakStabilizeWeightedFrame):
    """
    This class implements a non-linear weakly-stabilizing weighted frame averaging, as described in our work.
    Args:
        unstable_function (nn.Module):
            The unstable function to be stabilized.
        in_dim (int):
            The input dimension of the function.
        in_channels (int):
            The number of input channels.
        n_frames (int):
            The number of frames to use.
        an_invariant (bool, optional):
            Whether the function is An-invariant or not. Default is False.
        device (Optional):
            The device to use. Default is CPU.
        dtype (Optional):
            The data type to use. Default is torch.float64.
    """

    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: bool = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, *args, **kwargs):
        super(NonLinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, an_invariant,
                                                     device, dtype, *args, **kwargs)

    def stable_forward(self, x: Tensor, permutations: Tensor = None) -> Tensor:
        """
        This method implements the non-linear weakly-stabilizing transformation.
        Args:
            x (Tensor):
                The input tensor.
            permutations (Tensor):
                The frame permutations given by projective sorting (not used if the function is An-invariant).
        Returns:
            Tensor: The weakly-stabilized output.
        """
        if not self.an_invariant:
            sorted_x = torch.take_along_dim(x.unsqueeze(1), permutations.unsqueeze(-2), -1)

            sorted_x = sorted_x.unsqueeze(-3)
            f_x: Tensor = self.unstable_function(sorted_x)
            stable_func = f_x.sign() * torch.sqrt(0.5 * f_x.abs() * (f_x - self.unstable_function(
                torch.take_along_dim(sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2),
                                     -1))).abs().min(dim=-2, keepdim=True)[0])
            stable_func = stable_func.sum(dim=-2)
        else:
            f_x: Tensor = self.unstable_function(x)
            stable_func = f_x.sign() * torch.sqrt(
                0.5 * f_x.abs() * (f_x - self.unstable_function(x[..., self.transpositions[0]])).abs())
            stable_func = stable_func.unsqueeze(1)
        return stable_func


class LinearWeightedFrame(WeakStabilizeWeightedFrame):
    """
    This class implements a linear weakly-stabilizing weighted frame averaging, as described in our work.
    Args:
        unstable_function (nn.Module):
            The unstable function to be stabilized.
        in_dim (int):
            The input dimension of the function.
        in_channels (int):
            The number of input channels.
        n_frames (int):
            The number of frames to use.
        an_invariant (bool, optional):
            Whether the function is An-invariant or not. Default is False.
        device (Optional):
            The device to use. Default is CPU.
        dtype (Optional):
            The data type to use. Default is torch.float64.
    """

    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: bool = False,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, *args, **kwargs):
        super(LinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, an_invariant,
                                                  device, dtype, *args, **kwargs)

        self.delta = nn.Parameter(torch.rand(1, device=device, dtype=dtype), requires_grad=False)

    def stable_forward(self, x: Tensor, permutations: Tensor = None) -> Tensor:
        """
        This method implements the linear weakly-stabilizing transformation.
        Args:
            x (Tensor):
                The input tensor.
            permutations (Tensor):
                The frame permutations given by projective sorting (not used if the function is An-invariant).
        Returns:
            Tensor: The weakly-stabilized output.
        """
        if self.an_invariant:
            f_x: Tensor = self.unstable_function(x).unsqueeze(1)
            f_tx: Tensor = self.unstable_function(permute_ij(x.clone(), 0, 1)).clone().unsqueeze(1)
            # stable_func = f_x - (pdist.sum(dim=-1, keepdim=True) * (f_x + f_tx))
            stable_func = f_x - (0.5 * (f_x + f_tx))
            return stable_func


        EPS = 1e-50

        pairwise_dist = torch.cdist(x.transpose(-1, -2), x.transpose(-1, -2), p=2)
        eye = torch.eye(self.in_channels, requires_grad=False, dtype=self.get_dtype, device=self.get_device)
        eye.fill_diagonal_(float('inf'))
        pairwise_dist += eye.unsqueeze(0).to(self.get_device, self.get_dtype)

        eta_x = pairwise_dist.min(dim=-1, keepdim=False)[0].min(dim=-1, keepdim=True)[0].unsqueeze(-1)

        delta = nn.functional.relu(self.delta) + EPS

        pairwise_dist = torch.minimum(pairwise_dist, eta_x + delta) / (eta_x + delta)
        pairwise_dist = torch.tan(pairwise_dist * torch.pi / 2)
        pairwise_dist = torch.true_divide(1, torch.where(pairwise_dist <= EPS, EPS, pairwise_dist))
        pairwise_dist = torch.triu(pairwise_dist, diagonal=1)

        pairwise_dist = 0.5 * pairwise_dist / torch.sum(pairwise_dist, dim=(-2,-1), keepdim=True)
        pairwise_dist = nn.functional.relu(pairwise_dist - 10*EPS)

        B, I, J = torch.nonzero(pairwise_dist, as_tuple=True)
        sorted_x = torch.take_along_dim(
                x[B],
                torch.cat([permute_ij(self.identity.clone(), _i, _j) for (_i, _j) in zip(I, J)], dim=0).unsqueeze(1),
                -1
            )
        sorted_x = torch.take_along_dim(sorted_x.unsqueeze(1), permutations[B].unsqueeze(-2), -1)
        f_tx = self.unstable_function(sorted_x)


        pairwise_dist = pairwise_dist[B, I, J].unsqueeze(-1).unsqueeze(-1)
        B, I = torch.unique(B, return_counts=True)
        J = I.cumsum(dim=0) - 1
        f_tx = (f_tx * pairwise_dist).cumsum(dim=0)[J]
        f_tx[1:] = f_tx[1:] - f_tx[:-1]

        sorted_x = torch.take_along_dim(x.unsqueeze(1), permutations.unsqueeze(-2), -1)
        stable_func = 0.5 * self.unstable_function(sorted_x) - f_tx
        
        return stable_func


class SoftNonLinearWeightedFrame(WeakStabilizeWeightedFrame):
    """
    This class implements a non-linear weakly-stabilizing weighted frame averaging, using soft-sign function, as described in our work.
    Args:
        unstable_function (nn.Module):
            The unstable function to be stabilized.
        in_dim (int):
            The input dimension of the function.
        in_channels (int):
            The number of input channels.
        n_frames (int):
            The number of frames to use.
        an_invariant (bool, optional):
            Whether the function is An-invariant or not. Default is False.
        soft (bool, optional):
            Whether to use soft-sign (True) or hard-tanh (False) as the non-linear activation. Default is True.
        device (Optional):
            The device to use. Default is CPU.
        dtype (Optional):
            The data type to use. Default is torch.float64.
    """

    def __init__(self, unstable_function: nn.Module, in_dim: int, in_channels: int, n_frames: int,
                 an_invariant: Optional[bool] = False,
                 soft: Optional[bool] = True,
                 device: Optional = torch.device('cpu'), dtype: Optional = torch.float64, *args, **kwargs):
        super(SoftNonLinearWeightedFrame, self).__init__(unstable_function, in_dim, in_channels, n_frames, an_invariant,
                                                         device, dtype, *args, **kwargs)

        if soft:
            self.activation = nn.Softsign()
        else:
            self.activation = nn.Hardtanh()

    def stable_forward(self, x: Tensor, permutations: Tensor = None) -> Tensor:
        """
        This method implements the non-linear weakly-stabilizing transformation.
        Args:
            x (Tensor):
                The input tensor.
            permutations (Tensor):
                The frame permutations given by projective sorting (not used if the function is An-invariant).
        Returns:
            Tensor: The weakly-stabilized output.
        """
        if not self.an_invariant:
            sorted_x = torch.take_along_dim(x.unsqueeze(1), permutations.unsqueeze(-2), -1)
            sorted_x = sorted_x.unsqueeze(-3)
            f_x: Tensor = self.unstable_function(sorted_x)
            diff: Tensor = 0.5 * (f_x - self.unstable_function(torch.take_along_dim(
                sorted_x, self.transpositions.unsqueeze(0).unsqueeze(0).unsqueeze(-2), -1))
                                  ).abs().min(dim=-2, keepdim=True)[0]

            stable_func = torch.where(
                torch.isclose(torch.abs(f_x), diff),
                f_x,
                f_x * self.activation(diff / (torch.abs(f_x) - diff).abs())
            )
            stable_func = stable_func.sum(dim=-2)
        else:
            f_x: Tensor = self.unstable_function(x)
            diff: Tensor = 0.5 * (f_x - self.unstable_function(x[..., self.transpositions[0]])).abs()
            stable_func = torch.where(
                torch.isclose(torch.abs(f_x), diff),
                f_x,
                f_x * self.activation(diff / (torch.abs(f_x) - diff).abs())
            )
            stable_func = stable_func.unsqueeze(1)

        return stable_func


if __name__ == '__main__':
    from neural_networks import MLP
    from projective_layers import AnInvariantEmbedding

    b, d, n = 10, 3, 5
    print(f"b = {b}, d = {d}, n = {n}")
    F = nn.Sequential(nn.Flatten(-2, -1), MLP(n * d, 1, [20, 20, 20]))
    # F = nn.Sequential(AnInvariantEmbedding(d, n, 50), MLP(50, 1, [20, 20, 20]))
    SF = LinearWeightedFrame(F, d, n, 130, an_invariant=False)

    X = torch.rand(b, d, n, dtype=torch.float64)
    X[0, :, 0] = X[0, :, 3]
    print(SF(X))
    print(SF(X).shape)

    from time import time
    from matplotlib import pyplot as plt
    import numpy as np
    from tqdm import tqdm


    T = 10
    bs = np.array([1, 5, 10, 20, 30, 40, 50, 75, 100, 128, 200, 500])
    times = np.zeros(len(bs))
    counter = tqdm(bs, colour='green', desc='Time')

    F = nn.Sequential(nn.Flatten(-2, -1), MLP(n * d, 1, [20, 20, 20]))
    # F = nn.Sequential(AnInvariantEmbedding(d, n, 50), MLP(50, 1, [20, 20, 20]))
    SF = LinearWeightedFrame(F, d, n, 130, an_invariant=False)

    for i, b in enumerate(counter):
        all_t = 0.0
        for _ in range(T):
            X = torch.rand(b, d, n, dtype=torch.float64)
            start = time()
            SF(X)
            all_t += time() - start
        all_t /= T
        times[i] = all_t

    # plt.plot(bs, times)
    plt.plot(bs, times)
    plt.grid()

    # times = np.zeros(len(bs))
    # counter = tqdm(enumerate(bs), colour='green', desc='Time')
    # F = nn.Sequential(nn.Flatten(-2, -1), MLP(n * d, 1, [20, 20, 20]))
    # F = nn.Sequential(AnInvariantEmbedding(d, n, 50), MLP(50, 1, [20, 20, 20]))
    # SF = LinearWeightedFrame(F, d, n, 130, an_invariant=True)
#
    # for i, b in counter:
    #     all_t = 0.0
    #     for _ in range(T):
    #         x = torch.rand(b, d, n, dtype=torch.float64)
    #         start = time()
    #         SF(x)
    #         all_t += time() - start
    #     all_t /= T
    #     times[i] = all_t
#
    # plt.plot(bs, times)

    plt.show()


