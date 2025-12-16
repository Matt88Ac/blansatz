import os

import numpy as np
from tqdm import tqdm

from .cross_product import multi_cross
from .determinant import det
from .norm_cross_product_discontinuity import discontinuous_multi_cross

EXPERIMENTS = {
    'determinant': det,
    'cross_product': multi_cross,
    'norm_cross_product_discontinuity': discontinuous_multi_cross
}


def generate_datasets(experiment: str, n_elements: int, dim: int,
                      n_train: int = 100_000, n_val: int = 10_000, n_test: int = 20_000,
                      lower: float = -1.5, upper: float = 1.5):
    if n_test is None:
        n_test = n_val

    PATH = os.path.dirname(__file__) + f'{os.sep}{experiment}{os.sep}data{os.sep}'
    exp_func = EXPERIMENTS[experiment]

    if not os.path.exists(PATH + f'{n_elements}_{dim}'):
        os.mkdir(PATH + f'{n_elements}_{dim}')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}train')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}validation')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}test')

    for count, name in zip([n_train, n_val, n_test], ['train', 'validation', 'test']):
        for i in tqdm(range(count), desc=name):
            matrix = np.random.uniform(lower, upper, size=(1, dim, n_elements))
            fx = exp_func(matrix)
            np.save(PATH + f'{n_elements}_{dim}{os.sep}{name}{os.sep}matrix_{i}', matrix)
            np.save(PATH + f'{n_elements}_{dim}{os.sep}{name}{os.sep}res_{i}', fx)
