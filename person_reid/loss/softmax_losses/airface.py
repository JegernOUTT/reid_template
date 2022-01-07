import torch
import numpy as np

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin
from person_reid.modelling.common import l2_norm

__all__ = ['AirFace']


class AirFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of AirFace:Lightweight and Efficient Model for Face Recognition
        (https://arxiv.org/pdf/1907.12256.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=64.,
                 m=0.45,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        super(AirFace, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.classnum = out_features
        self.kernel = torch.nn.Parameter(torch.Tensor(in_features, self.classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.s = s
        self.eps = 1e-7
        self.pi = np.pi

    def forward(self, embbedings, label, *args, **kwargs):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)  # for numerical stability
        theta = torch.acos(cos_theta)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        target = (self.pi - 2 * (theta + self.m)) / self.pi
        others = (self.pi - 2 * theta) / self.pi

        output = (one_hot * target) + ((1.0 - one_hot) * others)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return self._calc_loss(output, label)
