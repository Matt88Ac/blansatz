import numpy as np


def multi_cross(x: np.ndarray) -> np.ndarray:
    N = x.shape[-1]
    res = x[..., 0]
    for i in range(1, N):
        res = np.cross(res, x[..., i], )

    return res


def cross_scale(x, y, scale):
    one, n, m = x.shape
    x = x * np.pow(scale, 1 / m)
    y = y * scale

    return x, y
