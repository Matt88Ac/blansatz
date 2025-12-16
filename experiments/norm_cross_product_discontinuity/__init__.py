import numpy as np


def discontinuous_multi_cross(x: np.ndarray, p: int = 2) -> np.ndarray:
    N = x.shape[-1]
    res = x[..., 0]
    for i in range(1, N):
        res = np.cross(res, x[..., i])

    res = np.sign(res.sum(axis=-1)) * np.linalg.norm(res, axis=-1, ord=p)

    return res
