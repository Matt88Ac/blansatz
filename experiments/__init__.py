import os

from .data_generator import EXPERIMENTS, generate_datasets
from .data_loader import get_experiment_dataloader, ExperimentDataset, dataset_collector

PATH = os.path.dirname(__file__)

for f in EXPERIMENTS.keys():
    if not os.path.exists(PATH + os.sep + f + os.sep + 'data'):
        os.mkdir(PATH + os.sep + f + os.sep + 'data')



