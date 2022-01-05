import torch
from torch import nn

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['MarginSoftmax']


class MarginSoftmax(SoftmaxLossMixin, nn.Module):
    def __init__(self,
                 s=64.0,
                 m=0.40,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        super(MarginSoftmax, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.s = s
        self.m = m

    def forward(self, cosine, label, *args, **kwargs):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0],
                            cosine.size()[1],
                            device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        output = cosine * self.s
        return self._calc_loss(output, label)
