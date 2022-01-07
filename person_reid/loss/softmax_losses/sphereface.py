import torch
import torch.nn.functional as F

from person_reid.loss.softmax_losses.base import SoftmaxLossMixin

__all__ = ['SphereFace', 'SphereProduct2']


class SphereFace(SoftmaxLossMixin, torch.nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 m=4.0,
                 loss_dict=SoftmaxLossMixin.DEFAULT_LOSS,
                 *args, **kwargs):
        super(SphereFace, self).__init__()
        SoftmaxLossMixin.__init__(self, loss_dict)
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label, *args, **kwargs):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return self._calc_loss(output, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


class SphereProduct2(torch.nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, lamb=0.7, r=30, m=0.4, t=3, b=0.25,
                 *args, **kwargs):
        super(SphereProduct2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamb = lamb
        self.r = r
        self.m = m
        self.t = t
        self.b = b
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: 2 * ((x + 1) / 2) ** self.t - 1,
        ]

    def forward(self, input, label, *args, **kwargs):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.r * (self.mlambda[0](cos_theta) - self.m) + self.b
        cos_m_theta1 = self.r * (self.mlambda[0](cos_theta) + self.m) + self.b
        cos_p_theta = (self.lamb / self.r) * torch.log(1 + torch.exp(-cos_m_theta))

        cos_n_theta = ((1 - self.lamb) / self.r) * torch.log(1 + torch.exp(cos_m_theta1))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        loss = (one_hot * cos_p_theta) + (1 - one_hot) * cos_n_theta
        loss = loss.sum(dim=1)

        return loss.mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
