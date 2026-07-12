import os

import numpy as np
from tqdm import tqdm

from .cross_product import multi_cross, cross_scale
from .determinant import det, det_scale
from .norm_cross_product_discontinuity import discontinuous_multi_cross
from .sin_vdet import sin_vdet

EXPERIMENTS = {
    'determinant': (det, det_scale),
    'cross_product': (multi_cross, cross_scale),
    'norm_cross_product_discontinuity': (discontinuous_multi_cross, cross_scale),
    'svdet': (sin_vdet, None),
}


def generate_datasets(experiment: str, n_elements: int, dim: int,
                      n_train: int = 100_000, n_val: int = 10_000, n_test: int = 20_000,
                      lower: float = -1.5, upper: float = 1.5,
                      batch_size: int = 1024
                      ):
    """
    Generates datasets for a given experiment and saves them to disk.
    Args:
        experiment (str): The name of the experiment to generate data for. Options are 'determinant', 'cross_product', and 'norm_cross_product_discontinuity'.
        n_elements (int): The number of elements in each input sample.
        dim (int): The dimensionality of each input sample.
        n_train (int, optional): Number of training samples to generate. Defaults to 100,000.
        n_val (int, optional): Number of validation samples to generate. Defaults to 10,000.
        n_test (int, optional): Number of test samples to generate. Defaults to 20,000.
        lower (float, optional): Lower bound for input values. Defaults to -1.5.
        upper (float, optional): Upper bound for input values. Defaults to 1.5.
        batch_size (int, optional): The size of each batch during generation. Defaults to 1024.
    """
    if n_test is None:
        n_test = n_val

    PATH = os.path.dirname(__file__) + f'{os.sep}{experiment}{os.sep}data{os.sep}'
    exp_func = EXPERIMENTS[experiment][0]

    if not os.path.exists(PATH + f'{n_elements}_{dim}'):
        os.mkdir(PATH + f'{n_elements}_{dim}')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}train')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}validation')
        os.mkdir(PATH + f'{n_elements}_{dim}{os.sep}test')

    for count, name in zip([n_train, n_val, n_test], ['train', 'validation', 'test']):
        index = 0
        batch = 1

        batch_size = batch_size if batch_size <= count else count
        sub_count = batch_size
        size = batch_size

        iter_number = count // batch_size
        if iter_number * batch_size > count:
            iter_number -= 1
        elif iter_number * batch_size < count:
            iter_number += 1
        pbar = tqdm(range(iter_number), colour='red', desc="generating {} samples...".format(name), leave=True)

        for b in pbar:
            if (sub_count - batch_size) >= count:
                break
            if index == count:
                break

            matrix = np.random.uniform(lower, upper, size=(size, dim, n_elements))
            fx: np.ndarray = exp_func(matrix)

            for i in range(size):
                np.save(PATH + f'{n_elements}_{dim}{os.sep}{name}{os.sep}matrix_{index}', matrix[i:i + 1])
                np.save(PATH + f'{n_elements}_{dim}{os.sep}{name}{os.sep}res_{index}', fx[i:i + 1])
                index += 1

            batch += 1
            if sub_count + batch_size > count:
                size = count - sub_count
            sub_count += size

        pbar.reset()
        pbar.clear()
        pbar.close()
