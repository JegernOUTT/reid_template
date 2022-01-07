import torch

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin
from person_reid.modelling.common import l2_norm

__all__ = ['AmSoftmax']


class AmSoftmax(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        s: scale of outputs
    """

    def __init__(self,
                 in_features,
                 out_features,
                 m=0.35,
                 s=30.0,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        super(AmSoftmax, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = torch.nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel
        self.m = m
        self.s = s

    def forward(self, embbedings, label, *args, **kwargs):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return self._calc_loss(output, label)
