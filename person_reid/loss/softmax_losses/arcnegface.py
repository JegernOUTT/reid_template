import math

import torch

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['ArcNegFace']


class ArcNegFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of Towards Flops-constrained Face Recognition (https://arxiv.org/pdf/1909.00632.pdf):
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
                 scale=64,
                 margin=0.5,
                 easy_margin=False,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        super(ArcNegFace, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.alpha = 1.2
        self.sigma = 2
        self.thresh = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label, *args, **kwargs):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())

        a = torch.zeros_like(cos)
        if self.easy_margin:
            for i in range(a.size(0)):
                lb = int(label[i])
                if cos[i, lb].data[0] > 0:
                    a[i, lb] = a[i, lb] + self.margin
            output = self.scale * torch.cos(torch.acos(cos) + a)
        else:
            b = torch.zeros_like(cos)
            a_scale = torch.zeros_like(cos)
            c_scale = torch.ones_like(cos)
            t_scale = torch.ones_like(cos)
            for i in range(a.size(0)):
                lb = int(label[i])
                a_scale[i, lb] = 1
                c_scale[i, lb] = 0
                if cos[i, lb].item() > self.thresh:
                    a[i, lb] = torch.cos(torch.acos(cos[i, lb]) + self.margin)
                else:
                    a[i, lb] = cos[i, lb] - self.mm
                reweight = self.alpha * torch.exp(-torch.pow(cos[i,] - a[i, lb].item(), 2) / self.sigma)
                t_scale[i] *= reweight.detach()
            output = self.scale * (a_scale * a + c_scale * (t_scale * cos + t_scale - 1))

        return self._calc_loss(output, label)
