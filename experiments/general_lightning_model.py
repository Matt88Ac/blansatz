from typing import Optional, Iterable

import torch
from pytorch_lightning import LightningModule

from ansatz_utils import get_model
from ansatzes import AfaNetModel, BiLipschitzAntiSymmetricModel, OnVandermondeModel
from experiments import (get_loss, get_optimizer, get_lr_scheduler, correlation_factor)


class UniformTransform(torch.nn.Module):
    def __init__(self, n_elements: int, dim: int, cutoff_value: float = 100.0,
                 device=torch.device('cuda'), dtype=torch.float64):
        super(UniformTransform, self).__init__()

        assert cutoff_value > 0

        self.n_elements = n_elements
        self.dim = dim
        self.cutoff_value = cutoff_value

        self.uni = torch.distributions.Uniform(low=torch.tensor([0.0], device=device, dtype=dtype),
                                               high=torch.tensor([float(cutoff_value)], device=device, dtype=dtype))

    def forward(self, feature_matrix: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_fx = self.uni.sample(target.shape) * target.sign()

        norm_fx = torch.where(torch.isclose(target, torch.zeros_like(target)), target, norm_fx)

        scale = torch.where(torch.isclose(target, torch.zeros_like(target)), torch.ones_like(target), norm_fx / target)

        feature_matrix = feature_matrix * torch.pow(scale.unsqueeze(-1), 1 / self.n_elements)

        return feature_matrix, norm_fx


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
        gradient_clip (bool, optional): Whether to apply gradient clipping. Defaults to False.
        gradient_clip_val (float, optional): Maximum norm for gradient clipping. Defaults to 0.0.
        gradient_clip_algorithm (str, optional): Algorithm for gradient clipping ('norm' or 'value'). Defaults to 'norm'.
        accumulate_grad_batches (int, optional): Number of batches to accumulate gradients over. Defaults to 1.
        transform (bool, optional): Whether to apply target-uniforming transform to the input. Defaults to False.
        cutoff_value (float, optional): The cutoff value for scaling outputs. Defaults to 100.0.
        corr_factor (bool, optional): Whether to use correlation factor in the loss computation. Defaults to False.
        corr_match (bool, optional): Whether to use correlation matching in the optimization. Defaults to False.
        corr_optimizer_kwargs (dict, optional): Dictionary of optimizer parameters for correlation matching. Defaults to None.
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
        dtype (torch.dtype, optional): Data type for model parameters. Defaults to torch.float64.
        **ansatz_kwargs: Additional keyword arguments for the model ansatz.

    Attributes:
        model (torch.nn.Module): The antisymmetric neural network model to be trained.
        model_name (str): Name of the model architecture.
        ansatz_kwargs (dict): Keyword arguments for the model ansatz.
        loss_name (str): Name of the loss function used.
        extra_metrics_names (Iterable[str]): Names of additional metrics to compute during training and evaluation.
        optim (callable): Optimizer function.
        lr_sched (callable): Learning rate scheduler function.
        example_input_array_dims (tuple): Dimensions of the example input array for model tracing.
        automatic_optimization (bool): Whether to use automatic optimization.
        corr_match (bool): Whether to use correlation matching in the optimization.
    """

    def __init__(self, in_dim: int, in_channels: int, out_dim: int, embedding_dim: Optional[int] = None,
                 model_name: Optional[str] = 'mlp',
                 optimizer_kwargs: Optional[dict] = None,
                 lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                 extra_metrics: Optional[Iterable[str]] = None,
                 gradient_clip: Optional[bool] = False,
                 gradient_clip_val: Optional[float] = 0.0,
                 gradient_clip_algorithm: Optional[str] = 'norm',
                 accumulate_grad_batches: Optional[int] = 1,
                 transform: Optional[bool] = False,
                 cutoff_value: Optional[float] = 100.0,
                 corr_factor: Optional[bool] = False,
                 corr_match: Optional[bool] = False,
                 corr_optimizer_kwargs: Optional[dict] = None,
                 device: str = 'cuda', dtype=torch.float64,
                 **ansatz_kwargs):
        super(GeneralTrainer, self).__init__()

        assert not (corr_factor and corr_match), "Cannot use both correlation factor and correlation matching."

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

        self.grad_needed = optimizer_kwargs['optimizer'] in {'adahessian', 'shampoo', 'qhadam', 'yogi'}

        self.automatic_optimization = (optimizer_kwargs['optimizer'] not in {'adahessian', 'shampoo', 'qhadam',
                                                                             'yogi'}) and (not corr_match)

        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip = gradient_clip
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.corr_factor = corr_factor
        self.corr_match = corr_match

        if corr_match:
            if corr_optimizer_kwargs is None:
                corr_optimizer_kwargs = optimizer_kwargs.copy()
            self.corr_optim = get_optimizer(**corr_optimizer_kwargs)

        self.model_name = self.model_name + f'_{self.loss_name}_{optimizer_kwargs["optimizer"]}'
        if self.lr_sched is not None:
            self.model_name = self.model_name + f'_{lr_scheduler_kwargs["lr_scheduler"]}'
        if self.corr_factor:
            self.model_name = self.model_name + '_corrfactor'
        if self.corr_match:
            self.model_name = self.model_name + f'_corrmatch_{corr_optimizer_kwargs["optimizer"]}'

        self.transformation = None
        if transform:
            self.transformation = UniformTransform(in_channels, in_dim, cutoff_value, device=device, dtype=dtype)

    def configure_optimizers(self):
        if not self.corr_match:
            optimizer = self.optim(self.model.parameters())
            if self.lr_sched is None:
                return optimizer

            return {'optimizer': optimizer,
                    'lr_scheduler': {"scheduler": self.lr_sched(optimizer), "monitor": 'val_loss'}}
        else:
            optimizer = self.optim(self.model.parameters())
            corr_optimizer = self.corr_optim(self.model.parameters())
            if self.lr_sched is None:
                return [optimizer, corr_optimizer]

            return [optimizer, corr_optimizer], [self.lr_sched(optimizer), self.lr_sched(corr_optimizer)]

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

    @torch.no_grad()
    def transform_target(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformation is not None:
            X, y = self.transformation(X, y)
        return X, y

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = self.transform_target(X, y)
        # _t = time()
        y_hat = self.forward(X).squeeze()
        y = y.squeeze()
        if self.corr_factor:
            loss: torch.Tensor = self.loss(y_hat, y) / (0.5 * (1 + correlation_factor(y_hat, y)) + 1e-10)
        else:
            loss: torch.Tensor = self.loss(y_hat, y)

        if not self.automatic_optimization:
            if self.grad_needed:
                loss.backward(create_graph=True, retain_graph=True)
            elif not self.grad_needed and self.corr_match:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            if self.trainer.is_last_batch or ((batch_idx + 1) % self.accumulate_grad_batches == 0):
                if not self.corr_match:
                    opt = self.optimizers()
                    if self.gradient_clip_val > 0 and self.gradient_clip:
                        if self.gradient_clip_algorithm == 'norm':
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
                        else:
                            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                else:
                    opt, corr_opt = self.optimizers()
                    if self.gradient_clip_val > 0 and self.gradient_clip:
                        if self.gradient_clip_algorithm == 'norm':
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
                        else:
                            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

                    opt.step()
                    opt.zero_grad(set_to_none=False)

                    corr_loss = 1 - (correlation_factor(y_hat.clone(), y.clone()) ** 2)
                    if self.grad_needed:
                        corr_loss.backward(create_graph=True, retain_graph=True)
                    else:
                        corr_loss.backward()
                    if self.gradient_clip_val > 0 and self.gradient_clip:
                        if self.gradient_clip_algorithm == 'norm':
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
                        else:
                            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

                    corr_opt.step()
                    corr_opt.zero_grad(set_to_none=True)

                    self.log('train_corr_loss', corr_loss, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        # self.log('train_time', time() - _t)
        with torch.no_grad():
            self.log('train_pred_std', y_hat.std(), prog_bar=True)
            self.log('train_target_std', y.std(), prog_bar=True)
            for metric in self.extra_metrics_names:
                self.log(f'train_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            X, y = batch
            # _t = time()
            y_hat = self.forward(X).squeeze()
            y = y.squeeze()
            self.log(f'val_loss', self.loss(y_hat, y), prog_bar=True)
            # self.log('val_time', time() - _t)
            self.log(f'val_pred_std', y_hat.std())

            for metric in self.extra_metrics_names:
                self.log(f'val_{metric}', self.extra_metrics[metric](y_hat, y), prog_bar=True)

    def on_validation_end(self) -> None:
        if (self.lr_sched is not None) and (not self.automatic_optimization):
            if not self.corr_match:
                lr_scheduler = self.lr_schedulers()
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(self.trainer.callback_metrics["val_loss"])
                else:
                    lr_scheduler.step()
            else:
                lr_scheduler, corr_lr_scheduler = self.lr_schedulers()
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(self.trainer.callback_metrics["val_loss"])
                    corr_lr_scheduler.step(self.trainer.callback_metrics["val_loss"])
                else:
                    lr_scheduler.step()
                    corr_lr_scheduler.step()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            X, y = batch
            y = y.squeeze()
            # _t = time()
            y_hat = self.forward(X).squeeze()
            loss = self.loss(y_hat, y)
            self.log(f'test_loss', loss)
            # self.log('test_time', time() - _t)
            self.log(f'test_pred_std', y_hat.std())
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
