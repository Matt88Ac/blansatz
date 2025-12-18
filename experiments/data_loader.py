import os
from glob import glob
from types import NoneType
from typing import Literal, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

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


class ExperimentLightningDataModule(LightningDataModule):
    def __init__(self, experiment: EXPERIMENTS, n_elements: int, dim: int, batch_size: int = 64,
                 shuffle: bool = True, n_workers: int = 0, pin_memory: bool = False,
                 persistent_workers: bool = False, device: str = 'cuda', dtype=torch.float64):
        super().__init__()
        self.n_elements = n_elements
        self.experiment = experiment
        self.batch_size = batch_size
        self.device = device
        self.dim = dim
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dtype = dtype

    def setup(self, stage):
        # assign to use in dataloaders
        self.train_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'train',
                                               device=self.device, dtype=self.dtype)
        self.val_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'validation',
                                             device=self.device, dtype=self.dtype)
        self.test_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'test',
                                              device=self.device, dtype=self.dtype)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=dataset_collector,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=dataset_collector,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=dataset_collector,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


if __name__ == '__main__':
    exit(0)
    data = get_experiment_dataloader('norm_cross_product_discontinuity', 10, 3, device='cpu')

    for batch_ndx, sample in enumerate(data):
        x, y = sample

        print(x.shape, y.shape)
