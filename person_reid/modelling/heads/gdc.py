import torch
from typing import Tuple

from person_reid.modelling.common import batch_norm1d_trt_friendly, LinearBlock

__all__ = ['GDC']


class GDC(torch.nn.Module):
    def __init__(self, input_channels: int, reid_features_number: int,
                 input_conv_kernel_size: Tuple[int, int]):
        super(GDC, self).__init__()
        self.input_conv_kernel_size = input_conv_kernel_size
        self.conv_6_dw = LinearBlock(input_channels, input_channels, groups=input_channels,
                                     kernel=self.input_conv_kernel_size,
                                     stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(input_channels, reid_features_number, bias=False)
        self.bn = torch.nn.BatchNorm1d(reid_features_number)

        self._initialize_weights()

    def forward(self, x):
        assert x.size(2) == self.input_conv_kernel_size[0] and x.size(3) == self.input_conv_kernel_size[1], \
            f'Invalid conv_kernel_size: required = {(x.size(2), x.size(3))}, set = {self.input_conv_kernel_size}'
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        if self.training:
            x = self.bn(x)
        else:
            x = batch_norm1d_trt_friendly(x, self.bn)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
