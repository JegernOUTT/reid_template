import torch

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['Circle']


class Circle(SoftmaxLossMixin, torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 m=0.25,
                 gamma=256,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS):
        super(Circle, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.margin = m
        self.gamma = gamma
        self.class_num = out_features
        self.emdsize = in_features

        self.weight = torch.nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, *args, **kwargs):
        similarity_matrix = torch.nn.functional.linear(
            torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12),
            torch.nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12))

        one_hot = torch.zeros_like(similarity_matrix)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        # sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]
        mask = one_hot.logical_not()
        sn = similarity_matrix[mask]

        sp = sp.view(input.size()[0], -1)
        sn = sn.view(input.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        output = torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)

        return self._calc_loss(output, label)
