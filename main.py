import torch
from experiments import run_experiments

from experiments_data import generate_datasets

if __name__ == '__main__':
    pass
    # generate_datasets('determinant', 2, 2, 100_000, 10_000, 10_000)
    # generate_datasets('determinant', 10, 10, 100_000, 10_000, 20_000,
    #                   lower=-1.4, upper=1.4)
    # generate_datasets('determinant', 15, 15, 100_000, 10_000, 20_000,
    #                   lower=-1.2, upper=1.2)

    # generate_datasets('determinant', 30, 30, 100_000, 10_000, 20_000,
    #                   lower=-0.3, upper=1.05)

    # generate_datasets('norm_cross_product_discontinuity', 10, 3, 100_000, 10_000, 20_000, -1.5, 1.5)
    # generate_datasets('norm_cross_product_discontinuity', 50, 3, 100_000, 10_000, 20_000, -1.3, 1.3)
    # generate_datasets('norm_cross_product_discontinuity', 300, 3, 100_000, 10_000, 20_000, -1.3, 1.5)

    # generate_datasets('cross_product', 10, 3, 100_000, 10_000, 20_000, -1.5, 1.5)
    # generate_datasets('cross_product', 50, 3, 100_000, 10_000, 20_000, -1.3, 1.3)
    # generate_datasets('cross_product', 300, 3, 100_000, 10_000, 20_000, -1.3, 1.5)

    run_experiments.run_experiment(experiment='determinant', ansatz_name='bl',
                                   n_elements=10, dim=10, batch_size=256,
                                   # embedding_dim=16,
                                   optimizer_kwargs=dict(optimizer='adam', lr=1e-3, amsgrad=True),
                                   lr_scheduler_kwargs=dict(lr_scheduler='reduce', factor=0.8, patience=2,
                                                            min_lr=1e-5, ),
                                   loss='l1', extra_metrics=['mse', 'l1', 'mare'],
                                   # model_name='ds', aggregation='max',
                                   model_name='mlp',
                                   hidden_layers=[256, 256, 256], activation='tanh', biases='all_but_last',
                                   device='cuda', dtype=torch.float32,
                                   persistent_workers=True, n_workers=12
                                   )

