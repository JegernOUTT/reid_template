import torch
import torch.nn.functional as F

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['CosFace']


class CosFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self,
                 in_features,
                 out_features,
                 s=64.0,
                 m=0.35,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        super(CosFace, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, *args, **kwargs):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return self._calc_loss(output, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
