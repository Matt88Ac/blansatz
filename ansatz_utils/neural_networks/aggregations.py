from functools import partial
from typing import Union, Iterable

from torch import Tensor, sum, mean, min, max, prod


def only_min(x: Tensor, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Tensor:
    return min(x, dim=dim, keepdim=keepdims)[0]


def only_max(x: Tensor, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Tensor:
    return max(x, dim=dim, keepdim=keepdims)[0]


def get_aggregation(agg_name: str, dim: Union[int, Iterable[int]], keepdims: bool = True) -> Union[partial]:
    if agg_name.lower() == 'sum':
        return partial(sum, dim=dim, keepdims=keepdims)

    elif agg_name.lower() in ['avg', 'average', 'mean']:
        return partial(mean, dim=dim, keepdims=keepdims)

    elif agg_name.lower() == 'prod':
        return partial(prod, dim=dim, keepdims=keepdims)

    elif agg_name.lower() == 'max':
        return partial(only_max, dim=dim, keepdims=keepdims)

    elif agg_name.lower() == 'min':
        return partial(only_min, dim=dim, keepdims=keepdims)

    raise NotImplementedError(r"Select one of 'sum', 'avg'/'average'/'mean', 'prod', 'max' or 'min'")
