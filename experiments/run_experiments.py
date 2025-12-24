import os
from typing import Optional, Iterable

import torch
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from experiments import LightningOnVandermondeModel, LightningBiLipschitzAntiSymmetricModel, LightningAfaNetModel
from experiments_data import ExperimentLightningDataModule, EXPERIMENTS

ANSATZES = {'bl': LightningBiLipschitzAntiSymmetricModel,
            'on': LightningOnVandermondeModel,
            'afanet': LightningAfaNetModel}


def run_experiment(experiment: EXPERIMENTS, n_elements: int, dim: int, ansatz_name: str,
                   embedding_dim: Optional[int] = None,
                   model_name: Optional[str] = 'mlp',
                   optimizer_kwargs: Optional[dict] = None,
                   lr_scheduler_kwargs: Optional[dict] = None, loss: Optional[str] = 'mse',
                   extra_metrics: Optional[Iterable[str]] = None,
                   batch_size: int = 64,
                   shuffle: bool = True, n_workers: int = 0, pin_memory: bool = False,
                   persistent_workers: bool = False, device: str = 'cuda', dtype=torch.float64,
                   **ansatz_kwargs):
    assert ansatz_name in ANSATZES.keys()
    out_dim = 1 if experiment in ['determinant', 'norm_cross_product_discontinuity'] else 3

    ansatz = ANSATZES[ansatz_name](dim, n_elements, out_dim, embedding_dim, model_name,
                                   optimizer_kwargs, lr_scheduler_kwargs, loss, extra_metrics,
                                   **ansatz_kwargs).to(device=device, dtype=dtype)

    data = ExperimentLightningDataModule(experiment, n_elements, dim, batch_size, shuffle,
                                         n_workers, pin_memory, persistent_workers,
                                         device=device, dtype=dtype)

    PATH = os.path.dirname(__file__) + os.sep + f'{experiment}_logs' + os.sep + f'{ansatz.model_name}_{data.batch_size}'

    tb_logger = TensorBoardLogger(PATH + os.sep + "tb_logs", log_graph=True)
    csv_logger = CSVLogger(PATH + os.sep + "csv_logs")
    checkpoint = callbacks.ModelCheckpoint(
        filename="best",
        monitor='val_loss',
        auto_insert_metric_name=True,
        save_top_k=1,
        save_last=True,
    )
    # LR_LEARN = FineTuneLearningRateFinder(k=10, strategy='mode', num_training_steps=10)

    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        accelerator="gpu" if device == 'cuda' else "cpu",
        max_epochs=100,
        callbacks=[checkpoint,
                   callbacks.LearningRateMonitor(),
                   callbacks.RichProgressBar(leave=False),
                   ],  # LR_LEARN],
        enable_checkpointing=True,
        log_every_n_steps=1,
        # gradient_clip_val=1.0, gradient_clip_algorithm='norm'
    )
    trainer.fit(ansatz, data)
