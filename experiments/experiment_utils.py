from functools import partial

import torch
from torch.optim import Adam, AdamW, Adamax, Adagrad, Rprop, RMSprop, SGD, NAdam, Adadelta
from torch_optimizer import Adahessian, Shampoo, QHAdam, Yogi
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR, ExponentialLR

AVAILABLE_OPTIMIZERS = {'adam', 'adamax', 'adamw', 'adagrad', 'nadam', 'rmsprop', 'rprop', 'sgd', 'adahessian',
                        'shampoo', 'qhadam', 'yogi', 'adadelta'}
AVAILABLE_LR_SCHED = {'reduce', 'cos_res', 'cos'}
AVAILABLE_LOSSES = {'mse', 'l1', 'huber', 'smooth_l1', 'mare', 'mard', 'msl', 'smsl'}


def correlation_factor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    if target.dim() == 1:
        mat = torch.cat(
            (
                (target - prediction).abs().unsqueeze(-1),
                ((target - prediction).abs() / (target.abs() + eps)).unsqueeze(-1)
            ), dim=-1
        )
    else:
        mat = torch.cat(
            (
                (target - prediction).abs().mean(dim=-1, keepdim=True),
                ((target - prediction).abs() / (target.norm(dim=-1, keepdim=True) + eps)).mean(dim=-1, keepdim=True)
            ), dim=-1
        )
    return torch.corrcoef(mat.T)[0, 1]


class MeanAbsoluteRelativeError(torch.nn.Module):
    """Mean Absolute Relative Error Loss"""

    def __init__(self):
        super(MeanAbsoluteRelativeError, self).__init__()
        self.eps = 1e-10

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() > 1:
            return ((target - prediction).abs() / (target.norm(dim=-1, keepdim=True) + self.eps)).mean()
        else:
            return ((target - prediction).abs() / (target.abs() + self.eps)).mean()


class MeanAbsoluteRelativeDistance(torch.nn.Module):
    """Symmetric Mean Absolute Relative Error"""

    def __init__(self):
        super(MeanAbsoluteRelativeDistance, self).__init__()
        self.error = MeanAbsoluteRelativeError()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.error(prediction, target) + self.error(target, prediction)) / 2


class MeanSquaredLogLoss(torch.nn.Module):
    """Symmetric Mean Absolute Relative Error"""

    def __init__(self):
        super(MeanSquaredLogLoss, self).__init__()
        self.epsilon = 1e-16
        self.mse = torch.nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse((prediction.abs() + self.epsilon).log(), (target.abs() + self.epsilon).log())


class SignedMeanSquaredLogLoss(torch.nn.Module):
    """Symmetric Mean Absolute Relative Error"""

    def __init__(self):
        super(SignedMeanSquaredLogLoss, self).__init__()
        self.mu = 0.1
        self.msl = MeanSquaredLogLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mu * self.msl(prediction, target) + (1 - self.mu) * self.mse(prediction, target)


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
    elif loss.lower() == 'mard':
        return MeanAbsoluteRelativeDistance()
    elif loss.lower() == 'msl':
        return MeanSquaredLogLoss()
    elif loss.lower() == 'smsl':
        return SignedMeanSquaredLogLoss()


def get_optimizer(optimizer: str, *args, **kwargs) -> partial:
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
    elif optimizer.lower() == 'adahessian':
        return partial(Adahessian, *args, **kwargs)
    elif optimizer.lower() == 'shampoo':
        return partial(Shampoo, *args, **kwargs)
    elif optimizer.lower() == 'qhadam':
        return partial(QHAdam, *args, **kwargs)
    elif optimizer.lower() == 'yogi':
        return partial(Yogi, *args, **kwargs)
    elif optimizer.lower() == 'adadelta':
        return partial(Adadelta, *args, **kwargs)
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not recognized. Choose one of {AVAILABLE_OPTIMIZERS}.')


def get_lr_scheduler(lr_scheduler: AVAILABLE_LR_SCHED, *args, **kwargs) -> partial:
    if lr_scheduler.lower() == 'reduce':
        return partial(ReduceLROnPlateau, *args, **kwargs)
    elif lr_scheduler.lower() == 'cos_res':
        return partial(CosineAnnealingWarmRestarts, *args, **kwargs)
    elif lr_scheduler.lower() == 'cos':
        return partial(CosineAnnealingLR, *args, **kwargs)

    elif lr_scheduler.lower() == 'exp':
        return partial(ExponentialLR, *args, **kwargs)


def get_dtype(dtype: str) -> torch.dtype:
    if '64' in dtype:
        return torch.float64

    elif '32' in dtype:
        return torch.float32

    elif '16' in dtype:
        return torch.float16
