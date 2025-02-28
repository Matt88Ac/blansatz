from typing import Literal, Optional
from random import sample

import torch

PERMUTATION_STRATEGIES = Literal['augment_swap', 'augment_random', 'constant_swap', 'constant_random']


def permutation_sign(p: torch.Tensor) -> int:
    """ The function computes the sign of a given permutation p.

    Args:
        p (torch.Tensor):
            A 1-D tensor of arbitrary length n, that contains distinct Long/Int elements from {0, ..., n-1}.

    Returns:
            -1 if the number of inversions in p is odd, or 1 otherwise.

    Example:
        n = 10;
        permutation = torch.randperm(n);
        sign_p = permutation_sign(permutation);
    """
    dim = len(p)
    count = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if p[i] > p[j]:
                count += 1
    sign = -2 * int(count % 2) + 1
    return sign


def random_negative_permutation(dim: int, device: Optional[str, torch.device] = 'cpu') -> torch.Tensor:
    """ The function generates a random negative-signed permutation of dimension 'dim'.
    Args:
        dim (int): The dimension of the permutation.
        device (str, torch.device, Optional): The device. Default: 'cpu'.

    Returns:
        Random negative-signed permutation element.
    """
    permutation = torch.randperm(dim, device=device)
    while permutation_sign(permutation) == 1:
        permutation = torch.randperm(dim, device=device)
    return permutation


def random_transposition(dim: int, device: Optional[str, torch.device] = 'cpu') -> torch.Tensor:
    """ The function generates a random transposition of dimension 'dim'. Note that a transposition is
        a negative-signed permutation.
    Args:
        dim (int): The dimension of the permutation.
        device (str, torch.device, Optional): The device. Default: 'cpu'.

    Returns:
        Random transposition element.
    """
    indices = range(dim)
    x1, x2 = sample(indices, 2)
    transposition = torch.tensor(list(indices), device=device, requires_grad=False)
    transposition[[x1, x2]] = transposition[[x2, x1]]
    return transposition
