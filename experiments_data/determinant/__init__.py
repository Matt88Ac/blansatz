import numpy as np
from numpy import linalg as la


def det(x):
    return la.det(x)[..., None]


def det_scale(x, y, scale):
    one, n, m = x.shape
    x, y = x * np.pow(scale, 1/n), y * scale
    return x, y
