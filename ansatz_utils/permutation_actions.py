from typing import Literal, Optional, Union
from random import sample

import torch

PERMUTATION_STRATEGIES = Literal['augment_swap', 'augment_random', 'constant_swap', 'constant_random']


def permutation_sign(p: torch.Tensor) -> torch.Tensor:
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
    sign = torch.ones(1, device=p.device, dtype=torch.int)
    identity = p.clone()
    for i in range(dim):
        if identity[i] != i:
            sign *= -1
            identity[identity[i]] = identity[i]
            identity[i] = i
    return sign


def random_negative_permutation(dim: int, device: Optional[Union[str, torch.device]] = 'cpu') -> torch.Tensor:
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


def random_positive_permutation(dim: int, device: Optional[Union[str, torch.device]] = 'cpu') -> torch.Tensor:
    """ The function generates a random positive-signed permutation of dimension 'dim'.
    Args:
        dim (int): The dimension of the permutation.
        device (str, torch.device, Optional): The device. Default: 'cpu'.

    Returns:
        Random negative-signed permutation element.
    """
    permutation = torch.randperm(dim, device=device)
    while permutation_sign(permutation) == -1:
        permutation = torch.randperm(dim, device=device)
    return permutation


def random_transposition(dim: int, device: Optional[Union[str, torch.device]] = 'cpu') -> torch.Tensor:
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


def vectorized_permutation_sign(p: torch.Tensor) -> torch.Tensor:
    """ The function computes the signs of a given tensor of permutations.

    Args:
        p (torch.Tensor):
            A k-D tensor of arbitrary last-dimension n, that contains distinct Long/Int elements from {0, ..., n-1}.

    Returns:
            -1 if the number of inversions in p is odd, or 1 otherwise.

    Example:
        n = 10;
        permutation = torch.randperm(n);
        sign_p = permutation_sign(permutation);
    """
    *size, dim = p.shape
    sign = torch.ones(size, device=p.device, dtype=torch.int32)
    identity = p.clone()

    for i in range(dim):
        eq_perm = identity[..., i] == i

        sign = torch.where(eq_perm, sign, -sign)
        identity[..., identity[..., i].clone()] = torch.where(eq_perm, identity[..., identity[..., i].clone()], identity[..., i]).clone()
        identity[..., i] = torch.where(eq_perm, identity[..., i], i).clone()

    return sign.unsqueeze(-1)


if __name__ == '__main__':
    per = torch.stack([torch.randperm(10, device='cuda'), torch.randperm(10, device='cuda')], dim=-1).T

    per = torch.stack([random_transposition(10, device='cuda'), random_transposition(10, device='cuda')], dim=-1).T
    print(vectorized_permutation_sign(per))
