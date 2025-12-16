import os

import numpy as np
from numpy import linalg as la
from .norm_cross_product_discontinuity import *
from .determinant import *
from .cross_product import *

PATH = os.path.dirname(__file__)
EXPERIMENTS = {
    'determinant': la.det,
    'cross_product': multi_cross,
    'norm_cross_product_discontinuity': discontinuous_multi_cross
}

for f in EXPERIMENTS.keys():
    if not os.path.exists(PATH + os.sep + f + os.sep + 'data'):
        os.mkdir(PATH + os.sep + f + os.sep + 'data')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'train')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'validation')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'test')
