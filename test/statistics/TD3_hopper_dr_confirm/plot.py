from typing import Dict, List, Tuple
from async_timeout import sys
from git import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_single_file(path: str) -> Dict:
    assert path.split('.')[-1] == 'csv' and os.path.exists(path), "Invalid csv file path."
    with open(path, 'r', encoding='utf-8') as f:
        dataframe = pd.read_csv(f)
    data = dataframe.to_numpy()
    steps = data[:, 1]
    scores = data[:, 2]
    return {'step': steps, 'score': scores}


def plot_best_performance() -> None:
    path = '/home/xukang/GitRepo/RobustRLBenchmarks/test/statistics/TD3_hopper_dr_confirm/'
    best_score_over_all = np.zeros([3, 3])

    for i, mode in enumerate(['oral', 'dr', 'baseline']):
        for j, seed in enumerate(['10', '20', '30']):
            file_path = path + f'{mode}_seed_{seed}.csv'
            step_score = read_single_file(file_path)
            best_score_over_all[i, j] = step_score['score'].max()

    print(best_score_over_all)

    bplot = plt.boxplot(
        best_score_over_all.T,
        vert=True,
        patch_artist=True,
        labels=['Oral', 'Domain Randomization', 'Baseline']
    )
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.grid()
    plt.xlabel('Different Train Setting')
    plt.ylabel('Evaluation Score')
    plt.title('Evaluation Score in Different Training Setting')

    plt.show()


if __name__ is '__main__':
    plot_best_performance()