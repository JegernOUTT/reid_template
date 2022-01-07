from typing import Dict, Any

import torch
import torch.nn as nn

__all__ = ['FocalLoss', 'HardMiningCRELoss', 'LabelSmoothCrossEntropyLoss',
           'SmoothingFocalLoss', 'SoftmaxLossMixin']


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class HardMiningCRELoss(nn.Module):
    def __init__(self, save_rate=2):
        super(HardMiningCRELoss, self).__init__()
        self.save_rate = save_rate
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        batch_size = input.shape[0]
        loss = self.ce(input, target)
        ind_sorted = torch.argsort(-loss)  # from big to small
        num_saved = int(self.save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(self.ce(input[ind_update], target[ind_update]))
        return loss_final


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = self.smooth_one_hot(target, self.cls, self.smoothing)
        loss_final = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return loss_final

    @staticmethod
    @torch.no_grad()
    def smooth_one_hot(true_labels, classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))

        smooth_label = torch.empty(size=label_shape, device=true_labels.device)
        smooth_label.fill_(smoothing / (classes - 1))
        smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return smooth_label


class SmoothingFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7, alpha=0.25, smoothing=0.1):
        super(SmoothingFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, pred, target):
        classes = pred.dim(1)
        pred_ls = (1 - self.smoothing) * pred + self.smoothing / classes
        pred_ls = torch.clamp(pred_ls, min=self.eps, max=1.0 - self.eps)
        cross_entropy = -target * torch.log(pred_ls)
        weight = self.alpha * target * torch.pow((1 - pred_ls), self.gamma)
        loss = weight * cross_entropy
        return loss.mean()


class SoftmaxLossMixin(nn.Module):
    LOSSES = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'FocalLoss': FocalLoss,
        'HardMiningCRELoss': HardMiningCRELoss,
        'LabelSmoothCrossEntropyLoss': LabelSmoothCrossEntropyLoss,
        'SmoothingFocalLoss': SmoothingFocalLoss,
    }
    DEFAULT_LOSS = {
        'type': 'CrossEntropyLoss'
    }

    def __init__(self, loss_dict: Dict[str, Any]):
        super().__init__()
        assert 'type' in loss_dict and loss_dict['type'] in SoftmaxLossMixin.LOSSES
        name = loss_dict.pop('type')
        self.loss = SoftmaxLossMixin.LOSSES[name](**loss_dict)

    def _calc_loss(self, pred, target):
        return self.loss(pred, target.view(-1, ).long())
