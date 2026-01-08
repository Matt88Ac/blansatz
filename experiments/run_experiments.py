import os
from typing import Optional, Iterable

import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from experiments import (LightningOnVandermondeModel, LightningBiLipschitzAntiSymmetricModel,
                         LightningAfaNetModel, LightningNoneAsModel, GeneralTrainer)
from experiments_data import ExperimentLightningDataModule, EXPERIMENTS

ANSATZES = {'bl': LightningBiLipschitzAntiSymmetricModel,
            'on': LightningOnVandermondeModel,
            'afanet': LightningAfaNetModel,
            'none': LightningNoneAsModel}


def run_experiment(experiment: str, n_elements: int, dim: int, ansatz_name: str,
                   embedding_dim: Optional[int] = None,
                   model_name: Optional[str] = 'mlp',
                   max_epochs: Optional[int] = 100,
                   optimizer_kwargs: Optional[dict] = None,
                   lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                   extra_metrics: Optional[Iterable[str]] = None,
                   early_stopping: Optional[bool] = True,
                   early_stopping_patience: Optional[int] = 3,
                   early_stopping_min_delta: Optional[float] = 1e-4,
                   gradient_clip: Optional[bool] = False,
                   gradient_clip_val: Optional[float] = 1.0,
                   gradient_clip_algorithm: Optional[str] = 'norm',
                   accumulate_grad_batches: Optional[int] = 1,
                   batch_size: Optional[int] = 64, shuffle: Optional[bool] = True,
                   n_workers: int = 0, pin_memory: bool = False,
                   persistent_workers: bool = False,
                   device: str = 'cuda', dtype=torch.float64,
                   **ansatz_kwargs):
    """
    Run a single experiment with specified parameters.

    Args:
        experiment (str): Name of the experiment to run. Must be in EXPERIMENTS ('determinant', 'norm_cross_product_discontinuity', 'cross_product').
        n_elements (int): Number of elements in the input set.
        dim (int): Dimensionality of each element in the input set.
        ansatz_name (str): Name of the ansatz to use. Must be one of the keys in ANSATZES ('bl', 'on', 'afanet', 'none').
        embedding_dim (int, optional): Dimensionality of the embedding space. Default is None.
        model_name (str, optional): Name of the model architecture to use. Default is 'mlp'.
        max_epochs (int, optional): Maximum number of training epochs. Default is 100.
        optimizer_kwargs (dict, optional): Dictionary of optimizer parameters. Default is None.
        lr_scheduler_kwargs (dict, optional): Dictionary of learning rate scheduler parameters. Default is None.
        loss (str, optional): Loss function to use. Default is 'mse'.
        extra_metrics (Iterable[str], optional): Additional metrics to compute during training. Default is None.
        early_stopping (bool, optional): Whether to use early stopping, based on validation loss. Default is True.
        early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 3.
        early_stopping_min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Default is 1e-4.
        gradient_clip (bool, optional): Whether to apply gradient clipping. Default is False.
        gradient_clip_val (float, optional): Maximum norm for gradient clipping. Default is 1.0.
        gradient_clip_algorithm (str, optional): Algorithm for gradient clipping ('norm' or 'value'). Default is 'norm'.
        accumulate_grad_batches (int, optional): Number of batches to accumulate gradients over. Default is 1.
        batch_size (int, optional): Batch size for training. Default is 64.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        n_workers (int): Number of worker processes for data loading. Default is 0.
        pin_memory (bool): Whether to pin memory during data loading. Default is False.
        persistent_workers (bool): Whether to use persistent workers for data loading. Default is False.
        device (str): Device to run the experiment on ('cuda' or 'cpu'). Default is 'cuda'.
        dtype: Data type for tensors. Default is torch.float64.
        ansatz_kwargs: Additional keyword arguments to pass to the ansatz constructor.

    Examples:
        >>> run_experiment('determinant', n_elements=15, dim=15, ansatz_name='bl', embedding_dim=16, model_name='mlp', optimizer_kwargs={'lr': 0.001}, batch_size=128, device='cuda')
        # Runs the determinant experiment with a bi-Lipschitz antisymmetric ansatz on a 15x15 input set.

        >>> run_experiment('cross_product', n_elements=10, dim=3, ansatz_name='on', embedding_dim=32,  model_name='ds', optimizer_kwargs={'lr': 0.0005}, loss='l1', extra_metrics=['mse', 'mare'], device='cpu')
    """
    assert ansatz_name in ANSATZES.keys()
    assert experiment in EXPERIMENTS
    out_dim = 1 if experiment in ['determinant', 'norm_cross_product_discontinuity'] else 3

    # print(torch._inductor.list_options())

    ansatz: GeneralTrainer = ANSATZES[ansatz_name](dim, n_elements, out_dim, embedding_dim, model_name,
                                                   optimizer_kwargs, lr_scheduler_kwargs, loss, extra_metrics,
                                                   **ansatz_kwargs).to(device=device, dtype=dtype)

    ansatz.configure_input_array()
    if device == 'cuda':
        ansatz.compile(fullgraph=True, dynamic=True,
                       options={"triton.cudagraphs": True,
                                "cuda.use_fast_math": True,
                                "triton.autotune_at_compile_time": True,
                                "enable_auto_functionalized_v2": True,
                                "memory_planning": True,
                                "pattern_matcher": True
                                # "epilogue_fusion": True,
                                # "max_autotune": True
                                }
                       )
    else:
        ansatz.compile(fullgraph=True, dynamic=True)

    data = ExperimentLightningDataModule(experiment, n_elements, dim, batch_size, shuffle,
                                         n_workers, pin_memory, persistent_workers,
                                         device=device, dtype=dtype)

    PATH = os.path.dirname(__file__) + os.sep + f'{experiment}_logs' + os.sep + f'{ansatz.model_name}_{data.batch_size}'

    tb_logger = TensorBoardLogger(PATH + os.sep + "tb_logs", default_hp_metric=False, )  # log_graph=True)
    csv_logger = CSVLogger(PATH + os.sep + "csv_logs")
    checkpoint = callbacks.ModelCheckpoint(
        filename="best",
        monitor='val_loss',
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
    )

    call_backs = [checkpoint, callbacks.LearningRateMonitor(), callbacks.TQDMProgressBar(leave=True)]
    if early_stopping:
        call_backs.append(
            callbacks.EarlyStopping('val_loss', early_stopping_min_delta, early_stopping_patience,
                                    stopping_threshold=0)
        )

    if not gradient_clip:
        trainer = Trainer(
            logger=[csv_logger, tb_logger],
            accelerator="gpu" if device == 'cuda' else "cpu",
            max_epochs=max_epochs,
            callbacks=call_backs,
            enable_checkpointing=True,
            log_every_n_steps=1, accumulate_grad_batches=accumulate_grad_batches
        )
    else:
        trainer = Trainer(
            logger=[csv_logger, tb_logger],
            accelerator="gpu" if device == 'cuda' else "cpu",
            max_epochs=max_epochs,
            callbacks=call_backs,
            enable_checkpointing=True,
            log_every_n_steps=1,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            accumulate_grad_batches=accumulate_grad_batches
        )
    trainer.fit(ansatz, data)
    trainer.test(ansatz, data, 'best')
