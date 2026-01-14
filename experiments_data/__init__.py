import os

from .data_generator import EXPERIMENTS, generate_datasets
from .data_loader import ExperimentLightningDataModule, ExperimentDataset, dataset_collector
from .data_analysis import run_analysis

PATH = os.path.dirname(__file__)

for f in EXPERIMENTS.keys():
    if not os.path.exists(PATH + os.sep + f + os.sep + 'data'):
        os.mkdir(PATH + os.sep + f + os.sep + 'data')



