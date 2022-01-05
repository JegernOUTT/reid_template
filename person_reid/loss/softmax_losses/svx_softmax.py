import math

import torch
import torch.nn.functional as F

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['SVXSoftmax']


class SVXSoftmax(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of Mis-classified Vector Guided Softmax Loss for Face Recognition
        (https://arxiv.org/pdf/1912.00833.pdf):
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
                 xtype='MV-AM',
                 s=32.0,
                 m=0.35,
                 t=0.2,
                 easy_margin=False,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        super(SVXSoftmax, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)

        self.xtype = xtype
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.t = t

        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, label, *args, **kwargs):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score
        if self.xtype == 'MV-AM':
            mask = cos_theta > gt - self.m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.m, gt)
            else:
                final_gt = gt - self.m
        elif self.xtype == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)
        else:
            raise Exception('unknown xtype!')
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.s
        return self._calc_loss(cos_theta, label)
