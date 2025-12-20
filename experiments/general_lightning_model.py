from time import time
from typing import Literal, List, Optional, Iterable

import torch
from pytorch_lightning import LightningModule

from .experiment_utils import (get_loss, get_optimizer, get_lr_scheduler, get_dtype,
                               AVAILABLE_OPTIMIZERS, AVAILABLE_LOSSES, AVAILABLE_LR_SCHED)

from ansatzes import AfaNetModel, BiLipschitzAntiSymmetricModel, OnVandermondeModel


class GeneralTrainer(LightningModule):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 **ansatz_kwargs):

        super(GeneralTrainer, self).__init__()
        self.save_hyperparameters()
        if extra_metrics is None:
            extra_metrics = []

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(optimizer='adam', lr=1e-3)

        self.loss = get_loss(loss)
        self.optim = get_optimizer(**optimizer_kwargs)
        if lr_scheduler_kwargs is None:
            self.lr_sched = None
        else:
            self.lr_sched = get_lr_scheduler(**lr_scheduler_kwargs)

        self.loss_name = loss.lower()
        self.extra_metrics_names = extra_metrics
        self.extra_metrics = torch.nn.ModuleDict({k: get_loss(k) for k in extra_metrics})

        self.model_name = f'{model_name}_{in_dim}_{in_dim}'
        self.ansatz_kwargs = ansatz_kwargs.copy()
        self.ansatz_kwargs['in_dim'] = in_dim
        self.ansatz_kwargs['in_channels'] = in_channels
        self.ansatz_kwargs['out_dim'] = out_dim
        self.ansatz_kwargs['embedding_dim'] = embedding_dim
        self.model = None

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


class LightningAfaNetModel(GeneralTrainer):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 frame_name: Optional[str] = 'nonlinear',
                 an_invariant: Optional[bool] = False,
                 flatten: Optional[bool] = False,
                 **ansatz_kwargs):
        ansatz_kwargs['frame_name'] = frame_name
        ansatz_kwargs['an_invariant'] = an_invariant
        ansatz_kwargs['flatten'] = flatten

        super(LightningAfaNetModel, self).__init__(in_dim, in_channels, out_dim, embedding_dim, model_name,
                                                   optimizer_kwargs, lr_scheduler_kwargs, loss, extra_metrics,
                                                   **ansatz_kwargs)

        self.model = AfaNetModel(**self.ansatz_kwargs)


class LightningBiLipschitzAntiSymmetricModel(GeneralTrainer):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 **ansatz_kwargs):
        super(LightningBiLipschitzAntiSymmetricModel, self).__init__(in_dim, in_channels, out_dim, embedding_dim,
                                                                     model_name,
                                                                     optimizer_kwargs, lr_scheduler_kwargs, loss,
                                                                     extra_metrics,
                                                                     **ansatz_kwargs)

        self.model = BiLipschitzAntiSymmetricModel(**self.ansatz_kwargs)


class LightningOnVandermondeModel(GeneralTrainer):
    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'ds',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 trainable_weights: Optional[bool] = False,
                 single_model: Optional[bool] = False,
                 **ansatz_kwargs):
        ansatz_kwargs['trainable_weights'] = trainable_weights
        ansatz_kwargs['single_model'] = single_model
        super(LightningOnVandermondeModel, self).__init__(in_dim, in_channels, out_dim, embedding_dim,
                                                          model_name,
                                                          optimizer_kwargs, lr_scheduler_kwargs, loss,
                                                          extra_metrics,
                                                          **ansatz_kwargs)

        self.model = OnVandermondeModel(**self.ansatz_kwargs)
