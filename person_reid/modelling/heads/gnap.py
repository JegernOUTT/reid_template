import torch

from person_reid.modelling.common import batch_norm1d_trt_friendly

__all__ = ['GNAP']


class GNAP(torch.nn.Module):
    def __init__(self, input_channels):
        super(GNAP, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(input_channels, affine=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = torch.nn.BatchNorm1d(input_channels, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        if x.dtype == torch.float16:
            x_norm = torch.norm(x, 2, 1, True, dtype=x.dtype)
            x_norm_mean = torch.mean(x_norm, dtype=x.dtype)
        else:
            x_norm = torch.norm(x, 2, 1, True)
            x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        if self.training:
            feature = self.bn2(x)
        else:
            feature = batch_norm1d_trt_friendly(x, self.bn2)
        return feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
