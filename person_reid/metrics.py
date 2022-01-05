from collections import defaultdict
from typing import Dict, List

import torch
from torchmetrics import Metric

__all__ = ['RankN']


class RankN(Metric):
    def __init__(self, top_k: int):
        super().__init__(compute_on_step=False)
        self._top_k = top_k
        self.add_state(f'embeddings', [])
        self.add_state(f'gt_labels', [])
        self.add_state(f'is_db', [])
        self.eps = 1e-6

    @staticmethod
    def _filter_and_mean_vectors(valid_classes_lst: List[int], vectors: Dict[int, torch.Tensor]):
        return torch.cat([vectors[c].mean(dim=0, keepdim=True)
                          for c in valid_classes_lst if c in vectors], dim=0).to(torch.float32)

    def _parse_eval_metadata(self, db) -> Dict[int, torch.Tensor]:
        embedding_by_class = defaultdict(list)
        for gt_labels, embeddings, is_dbs in zip(self.gt_labels, self.embeddings, self.is_db):
            for gt_label, is_db, embedding in zip(gt_labels, is_dbs, embeddings):
                if db != is_db:
                    continue
                embedding_by_class[int(gt_label)].append(embedding)
        for lbl, embeddings in embedding_by_class.items():
            embedding_by_class[lbl] = torch.stack(embeddings)
        return dict(embedding_by_class)

    @staticmethod
    def _get_device_dtype(labels: Dict[int, torch.Tensor]):
        assert len(labels) > 0
        elem = labels[0]
        return elem.device, elem.dtype

    def update(self, embeddings, gt_labels, is_db):
        self.embeddings.append(embeddings)
        self.gt_labels.append(gt_labels)
        self.is_db.append(is_db.to(torch.uint8))

    def compute(self):
        device, dtype = self._get_device_dtype(self.gt_labels)
        embeddings_by_class = self._parse_eval_metadata(db=False)
        db_embeddings_by_class = self._parse_eval_metadata(db=True)

        if len(embeddings_by_class) == 0 or len(db_embeddings_by_class) == 0:
            return torch.tensor(0.0, dtype=dtype, device=device)

        vectors_classes = set(list(embeddings_by_class.keys()))
        db_vectors_classes = set(list(db_embeddings_by_class.keys()))
        valid_classes_lst = sorted(list(vectors_classes & db_vectors_classes))

        cat_mean_vectors = self._filter_and_mean_vectors(valid_classes_lst, embeddings_by_class)
        mean_db_vectors = self._filter_and_mean_vectors(valid_classes_lst, db_embeddings_by_class)

        tp = 0
        for class_idx, db_vector in enumerate(mean_db_vectors):
            dists_by_classes = torch.cdist(db_vector.unsqueeze(0), cat_mean_vectors).mean(dim=0)
            top_classes_idxes = torch.argsort(dists_by_classes)[:self._top_k]

            if (top_classes_idxes == class_idx).sum() > 0:
                tp += 1

        return torch.tensor(tp / (len(valid_classes_lst) + self.eps), dtype=torch.float32, device=device)
