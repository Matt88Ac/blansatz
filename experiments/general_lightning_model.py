from time import time
from typing import Literal, List

import torch
from pytorch_lightning import LightningModule

from .experiment_utils import (get_loss, get_optimizer, get_lr_scheduler,
                               AVAILABLE_OPTIMIZERS, AVAILABLE_LOSSES, AVAILABLE_LR_SCHED)


class GeneralTrainer(LightningModule):
    def __init__(self, model: torch.nn.Module, model_name: str, optimizer_kwargs: dict = None,
                 lr_scheduler_kwargs: dict = None, loss: AVAILABLE_LOSSES = 'mse',
                 extra_metrics=None):
        super(GeneralTrainer, self).__init__()

        if extra_metrics is None:
            extra_metrics = []

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(optimizer='adam', lr=1e-3)

        self.loss = get_loss(loss)
        self.model = model
        self.model_name = model_name
        self.optim = get_optimizer(**optimizer_kwargs)
        if lr_scheduler_kwargs is None:
            self.lr_sched = None
        else:
            self.lr_sched = get_lr_scheduler(**lr_scheduler_kwargs)

        self.loss_name = loss.lower()
        self.extra_metrics_names = extra_metrics
        self.extra_metrics = torch.nn.ModuleDict({k: get_loss(k) for k in extra_metrics})

    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters())
        if self.lr_sched is None:
            return optimizer

        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": self.lr_sched(optimizer), "monitor": 'val_loss'}}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_size(self) -> int:
        s = 0
        for p in self.model.parameters():
            if p.requires_grad:
                s += p.shape.numel()
        return s

    @torch.no_grad()
    def extra_metrics_compute(self, y_hat, y, prefix: str):
        for metric in self.extra_metrics_names:
            self.log(f'{prefix}_{metric}', self.extra_metrics[metric](y_hat, y))

    def training_step(self, batch, batch_idx):
        X, y = batch
        _t = time()
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_pred_std', y_hat.std(), prog_bar=True)
        self.log('train_time', time() - _t)
        self.extra_metrics_compute(y_hat, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        _t = time()
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log(f'{prefix}_loss', loss)
        self.log(f'{prefix}_time', time() - _t)
        self.log(f'{prefix}_pred_std', y_hat.std())
        self.extra_metrics_compute(y_hat, y, prefix)
        return loss
