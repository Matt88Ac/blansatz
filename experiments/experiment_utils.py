from functools import partial

import torch
from torch.optim import Adam, AdamW, Adamax, Adagrad, Rprop, RMSprop, SGD, NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

AVAILABLE_OPTIMIZERS = {'adam', 'adamax', 'adamw', 'adagrad', 'nadam', 'rmsprop', 'rprop', 'sgd'}
AVAILABLE_LR_SCHED = {'reduce', 'cos_res', 'cos'}
AVAILABLE_LOSSES = {'mse', 'l1', 'huber', 'smooth_l1', 'mare'}


class MeanAbsoluteRelativeError(torch.nn.Module):

    def __init__(self):
        super(MeanAbsoluteRelativeError, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        zeros = (target == 0).to(dtype=target.dtype)
        loss = (target - prediction) / (target + zeros)
        loss = loss.abs().nanmean()
        return loss


def get_loss(loss: str) -> torch.nn.Module:
    assert loss.lower() in AVAILABLE_LOSSES, (
        NotImplementedError("Choose on of mse, l1, huber and smooth_l1 as loss functions"))

    if loss.lower() == 'mse':
        return torch.nn.MSELoss()
    elif loss.lower() == 'l1':
        return torch.nn.L1Loss()
    elif loss.lower() == 'huber':
        return torch.nn.HuberLoss()
    elif loss.lower() == 'smooth_l1':
        return torch.nn.SmoothL1Loss(beta=0.5)
    elif loss.lower() == 'mare':
        return MeanAbsoluteRelativeError()


def get_optimizer(optimizer: AVAILABLE_OPTIMIZERS, *args, **kwargs) -> partial:
    if optimizer.lower() == 'adam':
        return partial(Adam, *args, **kwargs)
    elif optimizer.lower() == 'adamw':
        return partial(AdamW, *args, **kwargs)
    elif optimizer.lower() == 'adamax':
        return partial(Adamax, *args, **kwargs)
    elif optimizer.lower() == 'adagrad':
        return partial(Adagrad, *args, **kwargs)
    elif optimizer.lower() == 'nadam':
        return partial(NAdam, *args, **kwargs)
    elif optimizer.lower() == 'rmsprop':
        return partial(RMSprop, *args, **kwargs)
    elif optimizer.lower() == 'rprop':
        return partial(Rprop, *args, **kwargs)
    elif optimizer.lower() == 'sgd':
        return partial(SGD, *args, **kwargs)


def get_lr_scheduler(lr_scheduler: AVAILABLE_LR_SCHED, *args, **kwargs) -> partial:
    if lr_scheduler.lower() == 'reduce':
        return partial(ReduceLROnPlateau, *args, **kwargs)
    elif lr_scheduler.lower() == 'cos_res':
        return partial(CosineAnnealingWarmRestarts, *args, **kwargs)
    elif lr_scheduler.lower() == 'cos':
        return partial(CosineAnnealingLR, *args, **kwargs)


def get_dtype(dtype: str) -> torch.dtype:
    if '64' in dtype:
        return torch.float64

    elif '32' in dtype:
        return torch.float32

    elif '16' in dtype:
        return torch.float16


