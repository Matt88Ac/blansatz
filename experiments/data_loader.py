import os
from glob import glob
from types import NoneType
from typing import Literal, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EXPERIMENTS = {'determinant', 'cross_product', 'norm_cross_product_discontinuity'}


def dataset_collector(batch: list[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(batch)
    matrices, targets = batch[0]
    for i in range(1, N):
        nx, ny = batch[i]
        matrices = torch.cat((matrices, nx))
        targets = torch.cat((targets, ny))
    return matrices, targets


class ExperimentDataset(Dataset):
    def __init__(self, experiment: EXPERIMENTS, n_elements: int, dim: int,
                 dataset: Literal['train', 'validation', 'test'] = 'train',
                 device=torch.device('cuda'), dtype=torch.float64):

        assert experiment in EXPERIMENTS

        self.dir = f'{os.sep}{experiment}{os.sep}data{os.sep}{n_elements}_{dim}{os.sep}{dataset}{os.sep}'
        self.dir = os.path.dirname(__file__) + self.dir

        self.n_files = len(glob(self.dir + '*.npy')) // 2
        self.device = device
        self.dtype = dtype
        self.pin = False
        self.dim = dim
        self.experiment = experiment
        self.n_elements = n_elements

    def pin_memory(self):
        self.pin = True

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = np.load(self.dir + f'matrix_{idx}.npy')
        target = np.load(self.dir + f'res_{idx}.npy')
        matrix = torch.tensor(matrix, dtype=self.dtype, device=self.device)
        target = torch.tensor(target, dtype=self.dtype, device=self.device)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        if self.pin:
            matrix = matrix.pin_memory(self.device)
            target = target.pin_memory(self.device)

        return matrix, target


def get_experiment_dataloader(experiment: EXPERIMENTS, n_elements: int, dim: int,
                              dataset: Literal['train', 'validation', 'test'] = 'train',
                              batch_size: Optional[int] = 1, shuffle: Optional[bool] = None, num_workers: int = 0,
                              pin_memory: bool = False, drop_last: bool = False, timeout: float = 0,
                              worker_init_fn: Optional[Callable[[int], NoneType]] = None,
                              multiprocessing_context=None, generator=None, *, prefetch_factor: Optional[int] = None,
                              persistent_workers: bool = False, pin_memory_device: str = '',
                              device=torch.device('cuda'), dtype=torch.float64) -> DataLoader:

    exp_dataset = ExperimentDataset(experiment, n_elements, dim, dataset, device, dtype)
    return DataLoader(exp_dataset, batch_size, shuffle=shuffle, collate_fn=dataset_collector,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                      worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context,
                      generator=generator, prefetch_factor=prefetch_factor,
                      persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)


if __name__ == '__main__':
    data = get_experiment_dataloader('norm_cross_product_discontinuity', 10, 3, device='cpu')

    for batch_ndx, sample in enumerate(data):
        x, y = sample

        print(x.shape, y.shape)
