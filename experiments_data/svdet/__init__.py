import numpy as np
from numpy import linalg as la

def uniform_sphere_sampling(n_samples: int, dim: int) -> np.ndarray:
    samples = np.random.normal(size=(n_samples, dim))
    return samples / la.norm(samples, ord=2, axis=-1, keepdims=True)


def all_diff(x: np.ndarray) -> np.ndarray:
    *b, n = x.shape
    diff = []
    for i in range(n-1):
        diff.append(x[..., i + 1:] - x[..., i:i + 1])

    diff = np.concatenate(diff, axis=-1)
    return diff


def sin_vdet(x: np.ndarray) -> np.ndarray:
    n_vad = 10
    b, d, n = x.shape

    diff = all_diff(x)
    f = 0
    c = np.random.uniform(1.0, n, n_vad)

    for i in range(n_vad):
        y = uniform_sphere_sampling(2 * n * d + 1, d)
        y = np.einsum('...dn,md->...mn', diff, y)
        y = y.prod(axis=-1).sum(axis=-1)
        y = np.clip(y, -np.pi / 2, np.pi / 2)
        y = np.sin(y)

        f += c[i] * y

    return f[..., None]
