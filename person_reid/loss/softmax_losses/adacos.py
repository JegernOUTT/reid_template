import math

import torch
import torch.nn.functional as F

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['AdaCos']


class AdaCos(SoftmaxLossMixin, torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 m=0.50,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        torch.nn.Module.__init__(self)
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.n_classes = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.m = m
        self.W = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, input, label, *args, **kwargs):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        return self._calc_loss(output, label)
