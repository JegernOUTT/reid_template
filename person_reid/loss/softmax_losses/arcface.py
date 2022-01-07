import math

import torch

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin
from person_reid.modelling.common import l2_norm

__all__ = ['ArcFace']


class ArcFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=64.0,
                 m=0.50,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        super(ArcFace, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.normal_(self.kernel, std=0.01)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embbedings, label, *args, **kwargs):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)  # for numerical stability
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return self._calc_loss(output, label)
