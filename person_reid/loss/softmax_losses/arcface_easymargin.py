import math

import torch
import torch.nn.functional as F

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['ArcFaceEasyMargin']


class ArcFaceEasyMargin(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m2: margin2
            m3: margin3
            cos(theta+m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=64.0,
                 m2=0.30,
                 m3=0.2,
                 easy_margin=False,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        super(ArcFaceEasyMargin, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m2 = m2
        self.m3 = m3
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m2)
        self.sin_m = math.sin(m2)
        self.th = math.cos(math.pi - m2)
        self.mm = math.sin(math.pi - m2) * m2

    def forward(self, input, label, *args, **kwargs):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = (cosine * self.cos_m - sine * self.sin_m) - self.m3
        phi = phi.to(input.dtype)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return self._calc_loss(output, label)
