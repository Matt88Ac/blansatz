import os

from numpy import linalg as la

from .data_generator import EXPERIMENTS, generate_datasets

PATH = os.path.dirname(__file__)

for f in EXPERIMENTS.keys():
    if not os.path.exists(PATH + os.sep + f + os.sep + 'data'):
        os.mkdir(PATH + os.sep + f + os.sep + 'data')



