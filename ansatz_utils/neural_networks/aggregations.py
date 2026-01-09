from typing import Union, Iterable, Optional, Callable

import torch
from torch import Tensor, sum, mean, min, max, prod


class Aggregation(torch.nn.Module):
    """
        Get aggregation function by name.
        Args:
            aggregation (Callable): Any callable function of the form aggregation(x, dim, keepdims), where x is a tensor, dim is the dimension(s) along which to aggregate, and keepdims is a boolean indicating whether to keep reduced dimensions.
            dim (int or Iterable[int]): Dimension(s) along which to aggregate.
            keepdims (bool, optional): Whether to keep the reduced dimensions. Default is True.
        """

    def __init__(self, aggregation: Callable, dim: Union[int, Iterable[int]], keepdims: Optional[bool] = True):
        super(Aggregation, self).__init__()
        self.dim = dim
        self.keepdims = keepdims
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregation(x, dim=self.dim, keepdims=self.keepdims)


def only_min(x: Tensor, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Tensor:
    return min(x, dim=dim, keepdim=keepdims)[0]


def only_max(x: Tensor, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Tensor:
    return max(x, dim=dim, keepdim=keepdims)[0]


def get_aggregation(agg_name: str, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Aggregation:
    """
    Get aggregation function by name.
    Args:
        agg_name (str): Name of the aggregation function. One of 'sum', 'avg'/'average'/'mean', 'prod', 'max', 'min'.
        dim (int or Iterable[int]): Dimension(s) along which to aggregate.
        keepdims (bool, optional): Whether to keep the reduced dimensions. Default is True.

    Returns:
        Aggregation: An instance of the Aggregation class with the specified aggregation function.

    Raises:
        NotImplementedError: If the aggregation function is not implemented, or the name is invalid (should be one of 'sum', 'avg', 'average', 'mean', 'prod', 'max', 'min').

    Examples:
        >>> agg = get_aggregation('sum', dim=1, keepdims=False)
    """
    if agg_name.lower() == 'sum':
        return Aggregation(sum, dim, keepdims)

    elif agg_name.lower() in ['avg', 'average', 'mean']:
        return Aggregation(mean, dim, keepdims)

    elif agg_name.lower() == 'prod':
        return Aggregation(prod, dim, keepdims)

    elif agg_name.lower() == 'max':
        return Aggregation(only_max, dim, keepdims)

    elif agg_name.lower() == 'min':
        return Aggregation(only_min, dim, keepdims)

    raise NotImplementedError(r"Select one of 'sum', 'avg'/'average'/'mean', 'prod', 'max' or 'min'")
