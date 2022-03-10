import random

import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure

__all__ = ['get_device_dtype', 'torch_random_choice', 'empty_metric_value', 'is_numeric_metric',
           'plot_intra_inter_hist', 'plot_cmc_line_plot']


def get_device_dtype(tensor: torch.Tensor):
    return tensor.device, tensor.dtype


def torch_random_choice(values, k):
    indices = random.sample(range(values.shape[0]), k)
    indices = torch.tensor(indices, dtype=torch.long, device=values.device)
    return values[indices]


def empty_metric_value(dtype, device):
    return {'empty': torch.tensor(0.0, dtype=dtype, device=device)}


def plot_intra_inter_hist(inter_class_distances, intra_class_distances,
                          sample_count=10000):
    inter_class_distances = inter_class_distances.cpu().numpy()
    intra_class_distances = intra_class_distances.cpu().numpy()

    sample_count = min(sample_count, len(inter_class_distances))
    inter_class_distances = np.random.choice(inter_class_distances, sample_count, replace=False)
    intra_class_distances = np.random.choice(intra_class_distances, sample_count, replace=False)
    indices = np.full((sample_count * 2,), fill_value='inter', dtype=np.object)
    indices[sample_count:] = 'intra'

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sns.kdeplot(x=np.concatenate([inter_class_distances, intra_class_distances]),
                hue=indices, ax=ax)
    return ax


def plot_cmc_line_plot(metric_accumulators):
    cmc = np.array([acc.accuracy for rank, acc in sorted(metric_accumulators.items(), key=lambda x: x[0])],
                   dtype=np.float32)
    data = {'accuracy': cmc, 'rank': range(1, len(cmc) + 1)}

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    sns.lineplot(data=data, x='rank', y='accuracy', markers=True, ax=ax)
    return ax


def is_numeric_metric(metrics):
    return type(metrics) is torch.Tensor
