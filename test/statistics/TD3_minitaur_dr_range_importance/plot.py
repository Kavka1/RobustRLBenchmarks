from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str) -> Tuple[np.array, np.array]:
    with open(path, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f, sep=',')
    total_steps = np.array(df['Total_Steps'])
    evaluation_scores = np.array(df['Evaluation_Score'])
    return total_steps, evaluation_scores


def boxplot_best_performance(result_path: str) -> None:
    ranges = [0, 1, 2, 3, 4]
    seeds = [10, 20, 30]
    best_performance = np.zeros((5, 9), dtype=np.float32)
    for i, ran in enumerate(ranges):
        for j, seed in enumerate(seeds):
            path = result_path + f"index_{ran}_seed_{seed}.csv"
            total_steps, scores = load_data(path)
            scores.sort()
            best_performance[i, j*3] = scores[-1]
            best_performance[i, j*3 + 1] = scores[-2]
            best_performance[i, j*3 + 2] = scores[-3]
    print(best_performance)

    bplot = plt.boxplot(
        best_performance.T,
        vert=True,
        patch_artist=True,
        labels=['[-0.2, 0.1]', '[-0.1, 0.]', '[0., 0.1]', '[0.1, 0.2]', '[0.2, 0.3]']
    )
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'brown']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.grid()
    plt.xlabel('Different Base_Mass Range for Training')
    plt.ylabel(f'Evaluation Score for Mass Range [-0.2, 0.5]')
    plt.title('Evaluation Score in Different Training Setting')

    plt.show()


if __name__ == '__main__':
    result_path = '/home/xukang/GitRepo/RobustRLBenchmarks/test/statistics/TD3_minitaur_dr_range_importance/'
    boxplot_best_performance(result_path)