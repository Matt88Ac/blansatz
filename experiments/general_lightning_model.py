from typing import Optional, Iterable

import torch
from pytorch_lightning import LightningModule

from ansatz_utils import get_model
from ansatzes import AfaNetModel, BiLipschitzAntiSymmetricModel, OnVandermondeModel
from experiments import (get_loss, get_optimizer, get_lr_scheduler)


class GeneralTrainer(LightningModule):
    """
    A general PyTorch Lightning trainer for different models. It supports customizable optimizers, learning rate schedulers,
    loss functions, and additional metrics. The model architecture is defined by the provided ansatz parameters.

    Args:
        in_dim (int): Input dimension.
        in_channels (int): Number of input channels.
        out_dim (int): Output dimension.
        embedding_dim (int, optional): Dimension of the embedding layer. Defaults to None, and dealt with in the ansatz.
        model_name (str, optional): Name of the model architecture to use. Defaults to 'mlp'.
        optimizer_kwargs (dict, optional): Dictionary of optimizer parameters. 
        lr_scheduler_kwargs (dict, optional): Dictionary of learning rate scheduler parameters.
        loss (str, optional): Loss function to use. Defaults to 'mse'.
        extra_metrics (Iterable[str], optional): Additional metrics to compute during training and evaluation. Defaults to None.
        ansatz_kwargs: Additional keyword arguments for the model ansatz.

    Attributes:
        model (torch.nn.Module): The antisymmetric neural network model to be trained.
        model_name (str): Name of the model architecture.
        ansatz_kwargs (dict): Keyword arguments for the model ansatz.
        loss_name (str): Name of the loss function used.
        extra_metrics_names (Iterable[str]): Names of additional metrics to compute during training and evaluation.
        optim (callable): Optimizer function.
        lr_sched (callable): Learning rate scheduler function.
    """

    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 accumulate_grad_batches: Optional[int] = 1,
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
        self.ansatz_kwargs['model_name'] = model_name
        self.ansatz_kwargs['in_dim'] = in_dim
        self.ansatz_kwargs['in_channels'] = in_channels
        self.ansatz_kwargs['out_dim'] = out_dim
        self.ansatz_kwargs['embedding_dim'] = embedding_dim
        self.model = None
        self.example_input_array_dims = (1, in_dim, in_channels)

        self.automatic_optimization = optimizer_kwargs['optimizer'] not in {'adahessian', 'shampoo'}
        self.accumulate_grad_batches = accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters())
        if self.lr_sched is None:
            return optimizer

        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": self.lr_sched(optimizer), "monitor": 'val_loss'}}

    def configure_input_array(self) -> None:
        self.example_input_array = torch.rand(*self.example_input_array_dims, dtype=self.dtype, device=self.device)

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
            self.log(f'{prefix}_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)

    def training_step(self, batch, batch_idx):
        X, y = batch

        # _t = time()
        y_hat = self.forward(X)
        loss: torch.Tensor = self.loss(y_hat, y)

        if not self.automatic_optimization:
            loss.backward(create_graph=True, retain_graph=True)

            if self.trainer.is_last_batch or ((batch_idx + 1) % self.accumulate_grad_batches == 0):
                opt = self.optimizers()
                opt.step()
                opt.zero_grad(set_to_none=True)

                if (self.lr_sched is not None) and self.trainer.is_last_batch:
                    lr_scheduler = self.lr_schedulers()
                    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(epoch=self.current_epoch)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_pred_std', y_hat.std(), prog_bar=True)
        # self.log('train_time', time() - _t)
        with torch.no_grad():
            for metric in self.extra_metrics_names:
                self.log(f'train_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        # _t = time()
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log(f'val_loss', loss)
        # self.log('val_time', time() - _t)
        self.log(f'val_pred_std', y_hat.std())
        with torch.no_grad():
            for metric in self.extra_metrics_names:
                self.log(f'val_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)

    def on_validation_end(self) -> None:
        if (self.lr_sched is not None) and (not self.automatic_optimization):
            lr_scheduler = self.lr_schedulers()
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch=self.current_epoch, metrics=self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        X, y = batch
        # _t = time()
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log(f'test_loss', loss)
        # self.log('test_time', time() - _t)
        self.log(f'test_pred_std', y_hat.std())
        with torch.no_grad():
            for metric in self.extra_metrics_names:
                self.log(f'test_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)
        return loss


class LightningAfaNetModel(GeneralTrainer):
    """
    Lightning wrapper for AfaNetModel.
    """

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

        self.model_name = f'afanet_{frame_name}_{an_invariant}_{self.model_size}_' + self.model_name


class LightningBiLipschitzAntiSymmetricModel(GeneralTrainer):
    """
    Lightning wrapper for BiLipschitzAntiSymmetricModel.
    """

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

        self.model_name = f'bl_{self.model_size}_' + self.model_name


class LightningOnVandermondeModel(GeneralTrainer):
    """
    Lightning wrapper for OnVandermondeModel.
    """

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
        self.model_name = f'on_{trainable_weights}_{single_model}_{self.model_size}_' + self.model_name


class LightningNoneAsModel(GeneralTrainer):
    """
    Lightning wrapper for none antisymmetric models (e.g, MLP).
    """

    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 **ansatz_kwargs):
        super(LightningNoneAsModel, self).__init__(in_dim, in_channels, out_dim, embedding_dim,
                                                   model_name,
                                                   optimizer_kwargs, lr_scheduler_kwargs, loss,
                                                   extra_metrics,
                                                   **ansatz_kwargs)

        self.ansatz_kwargs.pop('in_channels')
        self.ansatz_kwargs.pop('embedding_dim')
        if model_name.lower() == 'mlp':
            self.ansatz_kwargs['in_dim'] *= in_channels
            self.model = torch.nn.Sequential(torch.nn.Flatten(-2, -1), get_model(**self.ansatz_kwargs))

        else:
            if model_name.lower() in ['attention', 'transformer']:
                self.ansatz_kwargs['in_channels'] = in_channels
            self.model = get_model(**self.ansatz_kwargs)

        self.model_name = f'none_{self.model_size}_' + self.model_name
