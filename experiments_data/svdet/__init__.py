import numpy as np
from numpy import linalg as la


def all_diff(x: np.ndarray) -> np.ndarray:
    *b, n = x.shape
    diff = []
    for i in range(n-1):
        diff.append(x[..., i + 1:] - x[..., i:i + 1])

    diff = np.concatenate(diff, axis=-1)
    return diff

class sin_vdet:
    def __init__(self):
        self.n_samples = None
        self.dim = None
        self.n_vad = 10
        self.coefficients = None
        self.projections = None

    def _init_coefficients(self, d: int, n: int) -> None:
        if self.dim is None:
            print("The dimension of the input data has not been initialized. Initializing the coefficients.")
            self.dim = d
            self.n_samples = 2 * n * d + 1
            self.coefficients = np.random.uniform(1.0, n, self.n_vad)
            self.projections = np.random.normal(size=(self.n_vad, self.n_samples, self.dim))

        elif (self.dim != d) or (self.n_samples != 2 * n * d + 1):
            print("The dimension of the input data has changed. Reinitializing the coefficients.")
            self.dim = d
            self.n_samples = 2 * n * d + 1
            self.coefficients = np.random.uniform(-n, n, self.n_vad)
            self.projections = np.random.normal(size=(self.n_vad, self.n_samples, self.dim))

        return

    def __call__(self, x: np.ndarray) -> np.ndarray:
        b, d, n = x.shape
        self._init_coefficients(d, n)

        x_proj = np.einsum('bdn,vmd->bvmn', x, self.projections)
        vdet = all_diff(x_proj).prod(axis=-1).sum(axis=-1)
        vdet = vdet / np.sqrt(1 + (vdet**2))

        vdet = vdet @ self.coefficients
        return vdet[..., None]


if __name__ == '__main__':
    y = np.random.uniform(-2, 2, size=(1000, 3, 10))
    z = y.copy()
    z[..., 0] = y[..., 1].copy()
    z[..., 1] = y[..., 0].copy()

    model = sin_vdet()

    f1 = model(y)
    f2 = model(z)

    D = np.abs(f1 + f2).sum(axis=-1) / la.norm(f1, ord=2, axis=-1)

    print(D.mean())

    y[..., 0] = y[..., 1]
    print(model(y).sum())
