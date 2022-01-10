from logging import warning

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from tqdm import tqdm

from person_reid.metrics.utils import (get_device_dtype, empty_metric_value,
                                       torch_random_choice, plot_intra_inter_hist)

__all__ = ['EmbeddingsDistances']


class EmbeddingsDistances(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state(f'embeddings', [], dist_reduce_fx="cat")
        self.add_state(f'labels', [], dist_reduce_fx="cat")
        self.eps = 1e-6

    def update(self, embeddings, gt_labels, is_query):
        self.embeddings.append(embeddings[is_query])
        self.labels.append(gt_labels[is_query])

    def compute(self):
        device, dtype = get_device_dtype(self.embeddings)
        categories = set(list(self.labels))
        if len(categories) == 0:
            warning(f'EmbeddingsDistances has received 0 categories in query data, check your test data')
            return empty_metric_value(dtype, device)

        inter_class_distances, intra_class_distances = [], []
        for cls_idx in tqdm(categories, desc='Calculating embeddings distances', leave=False):
            embeddings = F.normalize(self.embeddings[self.labels == cls_idx], eps=1e-8)
            inter_distances = torch.cdist(embeddings.double(), embeddings.double(),
                                          compute_mode='donot_use_mm_for_euclid_dist').half()
            index = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
            inter_class_distances.append(inter_distances[index].view(-1))

            other_embeddings = F.normalize(self.embeddings[self.labels != cls_idx], eps=1e-8)
            intra_distances = torch.cdist(embeddings, other_embeddings,
                                          compute_mode='donot_use_mm_for_euclid_dist').half()
            intra_class_distances.append(torch_random_choice(
                intra_distances.view(-1), k=inter_class_distances[-1].shape[0]))

        inter_class_distances = torch.cat(inter_class_distances)
        intra_class_distances = torch.cat(intra_class_distances)

        mean_inter_class_distance = inter_class_distances.mean()
        mean_intra_class_distance = intra_class_distances.mean()
        return {
            'mean_inter_class_distance': mean_inter_class_distance,
            'mean_intra_class_distance': mean_intra_class_distance,
            'class_distance_difference': mean_intra_class_distance - mean_inter_class_distance,
            'inter_class_hist': plot_intra_inter_hist(inter_class_distances, intra_class_distances)
        }
