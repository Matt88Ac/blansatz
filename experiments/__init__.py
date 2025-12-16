import os
from .cross_product_discontinuity import *
from .determinant import *
from .cross_product import *

PATH = os.path.dirname(__file__)
EXPERIMENTS = ['determinant', 'cross_product', 'cross_product_discontinuity']

for f in EXPERIMENTS:
    if not os.path.exists(PATH + os.sep + f + os.sep + 'data'):
        os.mkdir(PATH + os.sep + f + os.sep + 'data')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'train')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'validation')
        os.mkdir(PATH + os.sep + f + os.sep + 'data' + os.sep + 'test')
