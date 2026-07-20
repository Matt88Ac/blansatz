import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm


def load_all_results(PATH: str, experiment: str) -> np.ndarray:

    loader = tqdm(glob(PATH + f'train{os.sep}res*.npy'), desc=f'loading training {experiment} target data', colour='blue')

    targets = np.vstack([np.load(path) for path in loader])
    return targets


def plot_stats(data: np.ndarray) -> None:
    count = len(data)
    n, d = data.shape
    data = np.abs(data)
    if d == 1:
        min = data.min()
        max = data.max()
        mean = data.mean()
        std = data.std()

        plt.grid(c='k')
        sns.kdeplot(data, ax=plt.gca(), legend=False, )
        plt.plot([0], [0], label=f'count: {count}', c='k')
        plt.plot([0], [0], label='max: {:.3f}'.format(max))
        plt.plot([0], [0], label='min: {:.3f}'.format(min))
        plt.plot([0], [0], label='mean: {:.3f}'.format(mean))
        plt.plot([0], [0], label='std: {:.3f}'.format(std))
        plt.legend(shadow=True, fancybox=True, edgecolor='k', facecolor='gold')

    else:
        min = data.min(axis=-1)
        max = data.max(axis=-1)
        mean = data.mean(axis=-1)
        std = data.std(axis=-1)
        for i in range(d):
            plt.subplot(1, d, i + 1)
            plt.grid(c='k')
            sns.kdeplot(data[:, i:i + 1], ax=plt.gca(), legend=False, )
            plt.plot([0], [0], label=f'count: {count}', c='k')
            plt.plot([0], [0], label='max: {:.3f}'.format(max[i]))
            plt.plot([0], [0], label='min: {:.3f}'.format(min[i]))
            plt.plot([0], [0], label='mean: {:.3f}'.format(mean[i]))
            plt.plot([0], [0], label='std: {:.3f}'.format(std[i]))
            plt.legend(shadow=True, fancybox=True, edgecolor='k', facecolor='gold')
            plt.xlabel(fr'dim {i + 1}')


def run_analysis(experiment: str, n_elements: int, dim: int, figsize: tuple = (12, 4), dpi: int = 130) -> None:
    plt.figure(figsize=figsize, dpi=dpi)
    PATH = os.path.dirname(__file__) + f'{os.sep}{experiment}{os.sep}data{os.sep}{n_elements}_{dim}{os.sep}'
    data = load_all_results(PATH, experiment)
    plot_stats(data)
    plt.suptitle(fr'${n_elements} \times {dim}$ {experiment} analysis')
    plt.tight_layout()
    plt.show()
