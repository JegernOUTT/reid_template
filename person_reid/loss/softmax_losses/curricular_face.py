import math

import torch

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin
from person_reid.modelling.common import l2_norm

__all__ = ['CurricularFace']


class CurricularFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of CurricularFace (https://arxiv.org/pdf/2004.00288.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """

    def __init__(self,
                 in_features,
                 out_features,
                 m=0.5,
                 s=64.,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        torch.nn.Module.__init__(self)
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        torch.nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label, *args, **kwargs):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos
        cos_theta_m = cos_theta_m.to(target_logit.dtype)  # (target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            self.t = self.t.to(target_logit.dtype)
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return self._calc_loss(output, label)
