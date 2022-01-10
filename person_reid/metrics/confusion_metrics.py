from dataclasses import dataclass
from logging import warning

import numpy as np
import torch
from torchmetrics import Metric
from tqdm import tqdm

from person_reid.metrics.utils import (get_device_dtype, empty_metric_value,
                                       plot_cmc_line_plot)

__all__ = ['ConfusionMetrics']


@dataclass
class MetricAccumulator:
    tp: int = 0
    fp: int = 0
    tp_overall: int = 0
    fp_overall: int = 0

    def accumulate(self):
        self.tp_overall += self.tp
        self.fp_overall += self.fp
        self.tp = 0
        self.fp = 0

    @property
    def accuracy(self):
        assert self.tp == 0 and self.fp == 0
        if self.tp_overall + self.fp_overall == 0:
            warning('Not enough data to calculate accuracy')
            return 0.0
        return self.tp_overall / (self.tp_overall + self.fp_overall)


class ConfusionMetrics(Metric):
    def __init__(self, ranks=(1, 5, 10), max_cmc_rank=100):
        super().__init__(compute_on_step=False)
        self._ranks = ranks
        self._max_cmc_rank = max_cmc_rank
        self.add_state(f'gallery_embeddings', [], dist_reduce_fx="cat")
        self.add_state(f'gallery_labels', [], dist_reduce_fx="cat")
        self.add_state(f'query_embeddings', [], dist_reduce_fx="cat")
        self.add_state(f'query_labels', [], dist_reduce_fx="cat")
        self.eps = 1e-6

    def update(self, embeddings, gt_labels, is_query):
        self.query_embeddings.append(embeddings[is_query].half())
        self.query_labels.append(gt_labels[is_query])
        self.gallery_embeddings.append(embeddings[~is_query].half())
        self.gallery_labels.append(gt_labels[~is_query])

    def _calc_metrics_by_rank(self, class_idx, distances, metrics):
        def _calc_map():
            cmc = np.array([acc.tp for rank, acc in sorted(metrics.items(), key=lambda x: x[0])],
                           dtype=np.float32)
            if cmc.sum() == 0:
                return 0.0

            cum_cmc = np.asarray([x / (i + 1.) for i, x in enumerate(cmc.cumsum())])
            return cum_cmc.sum() / cmc.sum()

        top_indexes = torch.argsort(distances)[:self._max_cmc_rank]
        top_classes = self.gallery_labels[top_indexes]
        for rank in range(self._max_cmc_rank):
            top_classes_ranked = top_classes[:rank]
            if (top_classes_ranked == class_idx).any():
                metrics[rank].tp += 1
            else:
                metrics[rank].fp += 1

        ap = _calc_map()

        for rank in range(self._max_cmc_rank):
            metrics[rank].accumulate()

        return ap

    def compute(self):
        device, dtype = get_device_dtype(self.gallery_embeddings)
        categories = set(list(self.query_labels))
        if len(categories) == 0:
            warning(f'ConfusionMetrics has received 0 categories in query data, check your test data')
            return empty_metric_value(dtype, device)

        min_metrics_by_rank = {rank: MetricAccumulator() for rank in range(self._max_cmc_rank)}
        max_metrics_by_rank = {rank: MetricAccumulator() for rank in range(self._max_cmc_rank)}
        mean_metrics_by_rank = {rank: MetricAccumulator() for rank in range(self._max_cmc_rank)}
        min_aps, max_aps, mean_aps = [], [], []
        for class_idx in tqdm(categories, desc='Calculating confusion metrics', leave=False):
            query_embeddings = self.query_embeddings[self.query_labels == class_idx]
            dists = torch.cdist(query_embeddings, self.gallery_embeddings)
            min_ap = self._calc_metrics_by_rank(class_idx, dists.min(dim=0)[0], min_metrics_by_rank)
            max_ap = self._calc_metrics_by_rank(class_idx, dists.max(dim=0)[0], max_metrics_by_rank)
            mean_ap = self._calc_metrics_by_rank(class_idx, dists.mean(dim=0), mean_metrics_by_rank)
            min_aps.append(min_ap)
            max_aps.append(max_ap)
            mean_aps.append(mean_ap)

        output = {}
        for rank in self._ranks:
            output[f'accuracy_{rank}_min'] = torch.tensor(min_metrics_by_rank[rank].accuracy,
                                                          dtype=dtype, device=device)
            output[f'accuracy_{rank}_max'] = torch.tensor(max_metrics_by_rank[rank].accuracy,
                                                          dtype=dtype, device=device)
            output[f'accuracy_{rank}_mean'] = torch.tensor(mean_metrics_by_rank[rank].accuracy,
                                                           dtype=dtype, device=device)
        output[f'mAP_min'] = torch.tensor(np.mean(min_aps), dtype=dtype, device=device)
        output[f'mAP_max'] = torch.tensor(np.mean(max_aps), dtype=dtype, device=device)
        output[f'mAP_mean'] = torch.tensor(np.mean(mean_aps), dtype=dtype, device=device)
        output[f'cmc_min'] = plot_cmc_line_plot(min_metrics_by_rank)
        output[f'cmc_max'] = plot_cmc_line_plot(max_metrics_by_rank)
        output[f'cmc_mean'] = plot_cmc_line_plot(mean_metrics_by_rank)

        return output
