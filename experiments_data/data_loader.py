import os
from glob import glob
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from ansatz_utils import random_negative_permutation, random_positive_permutation

EXPERIMENTS = {'determinant', 'cross_product', 'norm_cross_product_discontinuity'}


def dataset_collector(batch: list[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to combine individual samples into a batch. Utility function for DataLoader."""
    N = len(batch)
    matrices, targets = batch[0]
    for i in range(1, N):
        nx, ny = batch[i]
        matrices = torch.cat((matrices, nx))
        targets = torch.cat((targets, ny))
    return matrices, targets


class CutOffTransform(torch.nn.Module):
    def __init__(self, n_elements: int, dim: int, cutoff_value: float = 100.0,
                 device=torch.device('cuda'), dtype=torch.float64):
        super(CutOffTransform, self).__init__()

        assert cutoff_value > 0

        self.n_elements = n_elements
        self.dim = dim
        self.cutoff_value = cutoff_value

        self.normal = torch.distributions.HalfNormal(torch.tensor([cutoff_value / 2], device=device, dtype=dtype))

    def forward(self, feature_matrix: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = torch.any(target.abs() > self.cutoff_value, dim=-1, keepdim=True)

        scale = self.normal.sample()
        scale = scale / target.abs().max(dim=-1, keepdim=True)[0]

        target = torch.where(cond, target * scale, target)
        feature_matrix = torch.where(cond[..., None], feature_matrix * torch.pow(scale[..., None], 1 / self.n_elements),
                                     feature_matrix)

        return feature_matrix, target


class UniformTransform(torch.nn.Module):
    def __init__(self, n_elements: int, dim: int, cutoff_value: float = 100.0,
                 device=torch.device('cuda'), dtype=torch.float64):
        super(UniformTransform, self).__init__()

        assert cutoff_value > 0

        self.n_elements = n_elements
        self.dim = dim
        self.cutoff_value = cutoff_value

        self.normal = torch.distributions.Normal(loc=torch.tensor([0.0], device=device, dtype=dtype),
                                                 scale=torch.tensor([1.0], device=device, dtype=dtype))

        self.running_mean = None
        self.running_std = None
        self.momentum = 0.1
        self.eps = 1e-8

    def forward(self, feature_matrix: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.running_mean is None:
            self.running_mean = target.mean(dim=-1, keepdim=True)
            self.running_std = target.std(dim=0, keepdim=True)
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * target.mean(dim=-1, keepdim=True)
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * target.std(dim=-1, keepdim=True)

        print(self.running_std, self.running_mean)

        norm_fx = self.normal.cdf((target - self.running_mean) / (self.running_std + self.eps))
        norm_fx = 2 * self.cutoff_value * norm_fx - self.cutoff_value

        norm_fx = torch.where(target != 0, norm_fx, target)
        norm_fx = torch.where(target.sign() != norm_fx.sign(), -norm_fx, norm_fx)

        scale = torch.where(target != 0, norm_fx / target, torch.ones_like(target))
        feature_matrix = feature_matrix * np.pow(scale[..., None], 1 / self.n_elements)

        return feature_matrix, target


class ExperimentDataset(Dataset):
    """
    Dataset class for loading experiment data.
    """

    def __init__(self, experiment: str, n_elements: int, dim: int,
                 dataset: Literal['train', 'validation', 'test'] = 'train',
                 augment: int = 0,
                 cutoff: bool = False, cutoff_value: float = 100,
                 device=torch.device('cuda'), dtype=torch.float64):

        assert experiment in EXPERIMENTS
        assert augment >= 0 and isinstance(augment, int)

        self.dir = f'{os.sep}{experiment}{os.sep}data{os.sep}{n_elements}_{dim}{os.sep}{dataset}{os.sep}'
        self.dir = os.path.dirname(__file__) + self.dir

        self.n_files = len(glob(self.dir + '*.npy')) // 2
        self.device = device
        self.dtype = dtype
        self.pin = False
        self.dim = dim
        self.experiment = experiment
        self.n_elements = n_elements
        self.augment = augment

        self.transform = None

        if cutoff:
            self.transform = CutOffTransform(n_elements, dim, device=device, dtype=dtype, cutoff_value=cutoff_value)

    def pin_memory(self):
        self.pin = True

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = torch.from_numpy(np.load(self.dir + f'matrix_{idx}.npy')).to(dtype=self.dtype, device=self.device)
        target = torch.from_numpy(np.load(self.dir + f'res_{idx}.npy')).to(dtype=self.dtype, device=self.device)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        if self.transform is not None:
            matrix, target = self.transform(matrix, target)

        if self.augment > 0:
            positives = (torch.randn(self.augment) >= 0).int().sum()
            negatives = self.augment - positives

            for i in range(positives):
                perm = random_positive_permutation(self.n_elements, self.device)
                matrix = torch.cat((matrix, matrix.clone()[..., perm]), dim=0)
                target = torch.cat((target, target.clone()), dim=0)

            for i in range(negatives):
                perm = random_negative_permutation(self.n_elements, self.device)
                matrix = torch.cat((matrix, matrix.clone()[..., perm]), dim=0)
                target = torch.cat((target, -target.clone()), dim=0)

        if self.pin:
            matrix = matrix.pin_memory(self.device)
            target = target.pin_memory(self.device)

        return matrix, target


class ExperimentLightningDataModule(LightningDataModule):
    """
    LightningDataModule for loading experiment data.
    """

    def __init__(self, experiment: str, n_elements: int, dim: int, batch_size: int = 64,
                 shuffle: bool = True, n_workers: int = 0, pin_memory: bool = False,
                 persistent_workers: bool = False,
                 augment: int = 0, cutoff: bool = False, cutoff_value: float = 100.0,
                 device: str = 'cuda', dtype=torch.float64):
        super().__init__()
        assert batch_size % (augment + 1) == 0
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
        self.augment = augment
        self.cutoff = cutoff
        self.cutoff_value = cutoff_value

    def setup(self, stage):
        # assign to use in dataloaders
        self.train_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'train',
                                               self.augment, self.cutoff, self.cutoff_value,
                                               device=self.device, dtype=self.dtype)
        self.val_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'validation',
                                             0, False, 0,
                                             device=self.device, dtype=self.dtype)
        self.test_dataset = ExperimentDataset(self.experiment, self.n_elements, self.dim, 'test',
                                              0, False, 0,
                                              device=self.device, dtype=self.dtype)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size // (self.augment + 1),
                          collate_fn=dataset_collector,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=dataset_collector,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=dataset_collector,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


if __name__ == '__main__':
    pass
    exit(0)
    data = get_experiment_dataloader('norm_cross_product_discontinuity', 10, 3, device='cpu')

    for batch_ndx, sample in enumerate(data):
        x, y = sample

        print(x.shape, y.shape)
