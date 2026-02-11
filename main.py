from typing import Optional
import argparse

import torch
from experiments import run_experiments
from experiments_data import generate_datasets, run_analysis

import json


def parse_dict(d: str) -> Optional[dict]:
    if d in ['', 'none']:
        return None
    else:
        return json.loads(d.replace("'", '"'))


def run_jason_experiment(resume_version: int, **kwargs):
    if 'device' in kwargs.keys():
        if kwargs['device'] == 'cuda':
            from multiprocessing import set_start_method
            try:
                set_start_method('spawn')
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
    else:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    if 'dtype' in kwargs.keys():
        kwargs['dtype'] = torch.float32 if '32' in kwargs['dtype'] else torch.float64

    kwargs['resume_version'] = resume_version
    run_experiments.run_experiment(**kwargs)


def parse_to_generate(parsed_args):
    generate_datasets(parsed_args.experiment, parsed_args.n_elements, parsed_args.dim,
                      parsed_args.generate_n_train, parsed_args.generate_n_val, parsed_args.generate_n_test,
                      parsed_args.generate_lower, parsed_args.generate_upper, parsed_args.batch_size)


def parse_to_experiment(parsed_args):
    if parsed_args.device == 'cuda':
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    run_experiments.run_experiment(parsed_args.experiment, parsed_args.n_elements, parsed_args.dim,
                                   parsed_args.ansatz_name,
                                   parsed_args.embedding_dim, parsed_args.model_name, parsed_args.max_epochs,
                                   parsed_args.optimizer_kwargs, parsed_args.lr_scheduler_kwargs, parsed_args.loss,
                                   parsed_args.extra_metrics, parsed_args.early_stopping,
                                   parsed_args.early_stopping_patience,
                                   parsed_args.early_stopping_min_delta, parsed_args.gradient_clip,
                                   parsed_args.gradient_clip_val, parsed_args.gradient_clip_algorithm,
                                   parsed_args.accumulate_grad,
                                   parsed_args.batch_size, parsed_args.shuffle,
                                   parsed_args.augment,
                                   parsed_args.cutoff, parsed_args.cutoff_value,
                                   parsed_args.transform,
                                   parsed_args.corr_factor, parsed_args.corr_match,
                                   parsed_args.corr_optimizer_kwargs,
                                   parsed_args.n_workers, parsed_args.pin_memory, parsed_args.persistent_workers,
                                   parsed_args.resume_version,
                                   parsed_args.device,
                                   torch.float32 if parsed_args.dtype == 'float32' else torch.float64,
                                   **parsed_args.ansatz_kwargs)


def parser_def():
    parser = argparse.ArgumentParser(prog='experiments',
                                     description='Run experiments for different ansatzes on various tasks. Make sure to generate data prioraly.',
                                     formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument('--use_json', action='store_true',
                        help='If set, load experiment configuration from path instead of command-line arguments.')
    parser.add_argument('--json_config_path', type=str, required=False,
                        default='experiments/configs/det_10_config.json',
                        help='Path to the JSON configuration file to load if --use_json is set.')

    parser.add_argument('--experiment', type=str, required=False, default='determinant',
                        help='Name of the experiment to run. Must be in EXPERIMENTS ("determinant", "norm_cross_product_discontinuity", "cross_product").')
    parser.add_argument('--n_elements', type=int, required=False, default=10, help='Number of elements in the input.')
    parser.add_argument('--dim', type=int, required=False, default=10,
                        help='Dimensionality of each element in the input.')
    parser.add_argument('--ansatz_name', type=str, required=False, default='afanet',
                        help='Name of the ansatz to use. One of ("bl", "on", "afanet", "none").')
    parser.add_argument('--embedding_dim', type=int, required=False, default=None,
                        help='Dimensionality of the embedding space.')
    parser.add_argument('--model_name', type=str, required=False, default='mlp',
                        help='Name of the model architecture to use. Must be one of ("mlp", "deepsets", "attention").')

    parser.add_argument('--optimizer_kwargs',
                        type=parse_dict,
                        required=False, default="{'optimizer': 'adam', 'lr': 1e-3}",
                        help='Dictionary of optimizer parameters, e.g., {"optimizer": "adam", "lr": 0.001}.')
    parser.add_argument('--lr_scheduler_kwargs',
                        type=parse_dict,
                        required=False,
                        default="none",
                        help='Dictionary of learning rate scheduler parameters, e.g., {"lr_scheduler": "reduce", "factor": 0.3, "min_lr": 1e-6}.')

    parser.add_argument('--loss', type=str, required=False, default='mse', help='Loss function to use (mse, l1, etc.).')
    parser.add_argument('--extra_metrics', nargs='+', type=str, required=False, default=['mse', 'l1', 'mare'],
                        help='Additional metrics to compute during training (list of strings).')

    parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to use early stopping based on validation loss.')
    parser.add_argument('--early_stopping_patience', type=int, required=False, default=15,
                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--early_stopping_min_delta', type=float, required=False, default=1e-4,
                        help='Minimum change in the monitored quantity to qualify as an improvement.')

    parser.add_argument('--gradient_clip', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to apply gradient clipping.')
    parser.add_argument('--gradient_clip_val', type=float, required=False, default=1.0,
                        help='Maximum norm for gradient clipping.')
    parser.add_argument('--gradient_clip_algorithm', type=str, required=False, default='norm',
                        help='Algorithm for gradient clipping ("norm", "value" or "noise").')

    parser.add_argument('--accumulate_grad', type=int, required=False, default=1,
                        help='Number of batches to accumulate gradients over.')

    parser.add_argument('--transform', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to apply input/output transformation.')

    parser.add_argument('--max_epochs', type=int, required=False, default=100,
                        help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', type=int, required=False, default=512, help='Batch size for training.')
    parser.add_argument('--shuffle', action='store_false',
                        required=False, default=True, help='Whether to shuffle the data.')

    parser.add_argument('--augment', type=int, required=False, default=0,
                        help='Number of augmentations to apply to the input data.')

    parser.add_argument('--n_workers', type=int, required=False, default=16,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to pin memory during data loading.')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to use persistent workers for data loading.')

    parser.add_argument('--device', type=str, required=False, default='cpu',
                        help='Device to run the experiment on (cuda or cpu).')
    parser.add_argument('--dtype', type=str, required=False, default='float64',
                        help='Data type for tensors (float32 or float64).')

    parser.add_argument('--ansatz_kwargs',
                        type=parse_dict,
                        required=False, default='none',
                        help='Additional keyword arguments to pass to the ansatz constructor.')

    parser.add_argument('--corr_factor', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to use correlation factor in the loss computation.')

    parser.add_argument('--corr_match', action=argparse.BooleanOptionalAction,
                        required=False, default=False, help='Whether to use correlation matching.')

    parser.add_argument('--corr_optimizer_kwargs', type=parse_dict,
                        required=False, default="none",
                        help='Dictionary of optimizer parameters for correlation matching, e.g., {"optimizer": "adam", "lr": 0.001}.')

    parser.add_argument('--run_generate_datasets', action='store_true',
                        help='If set, generate datasets instead of running an experiment.')
    parser.add_argument('--generate_lower', type=float, required=False, default=-1.0,
                        help='Lower bound for dataset generation.')
    parser.add_argument('--generate_upper', type=float, required=False, default=1.0,
                        help='Upper bound for dataset generation.')
    parser.add_argument('--generate_n_train', type=int, required=False, default=100_000,
                        help='Number of training samples to generate.')
    parser.add_argument('--generate_n_val', type=int, required=False, default=10_000,
                        help='Number of validation samples to generate.')
    parser.add_argument('--generate_n_test', type=int, required=False, default=20_000,
                        help='Number of test samples to generate.')
    parser.add_argument('--cutoff', action=argparse.BooleanOptionalAction,
                        required=False, default=False,
                        help='Whether to apply cutoff scaling to outputs during dataset generation.')
    parser.add_argument('--cutoff_value', type=float, required=False, default=100.0,
                        help='The cutoff value for scaling outputs during dataset generation.')
    parser.add_argument('--resume_version', type=int, required=False, default=-2,
                        help='Version of the experiment to resume training from. Use -2 to not resume. Use -1 for latest, or specific version number.')

    parser.add_argument('--run_analysis', action='store_true',
                        help='If set, run analysis on the specified experiment instead of training.')

    return parser.parse_args()


def main():
    parsed_args = parser_def()
    if parsed_args.run_generate_datasets:
        parse_to_generate(parsed_args)

    else:
        if parsed_args.run_analysis:
            run_analysis(experiment=parsed_args.experiment,
                         n_elements=parsed_args.n_elements,
                         dim=parsed_args.dim)
        else:
            if parsed_args.use_json:
                with open(parsed_args.json_config_path, 'rb') as f:
                    json_args = json.load(f)
                run_jason_experiment(parsed_args.resume_version, **json_args)
            else:
                parse_to_experiment(parsed_args)


if __name__ == '__main__':
    main()
    exit(0)
    # generate_datasets('determinant', 2, 2, 100_000, 10_000, 10_000)
    # generate_datasets('determinant', 10, 10, 100_000, 10_000, 20_000,
    #                   lower=-1.4, upper=1.4)
    # generate_datasets('determinant', 15, 15, 100_000, 10_000, 20_000,
    #                   lower=-1.2, upper=1.2)
    #
    # generate_datasets('determinant', 30, 30, 100_000, 10_000, 20_000,
    #                   lower=-0.3, upper=1.05)
    #
    # generate_datasets('norm_cross_product_discontinuity', 10, 3, 100_000, 10_000, 20_000, -1.5, 1.5)
    # generate_datasets('norm_cross_product_discontinuity', 50, 3, 100_000, 10_000, 20_000, -1.3, 1.3)
    # generate_datasets('norm_cross_product_discontinuity', 300, 3, 100_000, 10_000, 20_000, -1.3, 1.5)
    #
    # generate_datasets('cross_product', 10, 3, 100_000, 10_000, 20_000, -1.5, 1.5)
    # generate_datasets('cross_product', 50, 3, 100_000, 10_000, 20_000, -1.3, 1.3)
    # generate_datasets('cross_product', 300, 3, 100_000, 10_000, 20_000, -1.3, 1.5)
    # exit(0)

    run_experiments.run_experiment(experiment='determinant',
                                   ansatz_name='afanet',
                                   n_elements=2, dim=2, batch_size=2048,
                                   # embedding_dim=16,
                                   optimizer_kwargs=dict(optimizer='adam', lr=1e-3),
                                   lr_scheduler_kwargs=dict(lr_scheduler='reduce', factor=0.3, patience=0,
                                                            min_lr=1e-6),
                                   early_stopping_patience=15,
                                   loss='mse', extra_metrics=['mse', 'l1', 'mare'],
                                   model_name='transformer',
                                   aggregation='linear',
                                   an_invariant=False, frame_name='nonlinear',
                                   hidden_layers=[8, 8, 8], activation='tanh', biases='all_but_last',
                                   # single_model=False, trainable_weights=True,
                                   max_epochs=100,
                                   device='cpu', dtype=torch.float64,
                                   persistent_workers=True, n_workers=16, pin_memory=True
                                   )
