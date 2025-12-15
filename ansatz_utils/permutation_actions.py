from random import sample
from typing import Optional, Union

import torch


@torch.jit.script
def permutation_sign(p: torch.Tensor) -> torch.Tensor:
    """ The function computes the sign of a given permutation p in O(n) time complexity.

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
    count = torch.zeros(1, device=p.device, dtype=torch.int)
    visited = torch.zeros(dim, device=p.device, dtype=torch.bool)
    true_tensor = torch.ones(1, device=p.device, dtype=torch.int)
    for i in range(dim):
        if not visited[i]:
            count += 1
            k = i
            while not visited[k]:
                visited = visited.select_scatter(true_tensor.clone()[0], 0, k)
                k = p[k].clone()
    sign = - 2 * ((dim - count) % 2) + 1
    return sign


def all_transpositions(n: int, device: Optional[Union[str, torch.device]] = 'cpu') -> torch.Tensor:
    indices = [list(range(n))] * (n - 1)
    indices = torch.tensor(indices, dtype=torch.int64, device=device)
    for j in range(n - 1):
        indices[j, [j + 1, j]] = indices[j, [j, j + 1]]
    return indices


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


@torch.jit.script
def vmap_permutation_sign(p: torch.Tensor) -> torch.Tensor:
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
    signs = torch.stack([
        permutation_sign(p_i) for p_i in torch.unbind(p, dim=0)
    ], dim=0)
    return signs


def vectorized_permutation_sign(p: torch.Tensor) -> torch.Tensor:
    shape = p.shape
    if len(shape) == 1:
        return permutation_sign(p)
    signs = vmap_permutation_sign(p.reshape(-1, shape[-1])).reshape(shape[:-1])
    return signs


if __name__ == '__main__':
    N_tests = 1000
    D = 100
    perms = torch.vstack([torch.randperm(D) for j in range(N_tests)])
    S = vectorized_permutation_sign(perms)

    ID_MAT = torch.eye(D)
    for j in range(N_tests):
        s1 = ID_MAT[perms[j]].det()
        s2 = S[j]
        if s2 != s1:
            print(s1, s2)
