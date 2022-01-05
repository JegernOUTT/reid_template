import math

import torch

__all__ = ['ECA']


class ECA(torch.nn.Module):
    def __init__(self, input_channels: int, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(input_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
