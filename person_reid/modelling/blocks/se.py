import torch

from person_reid.modelling.common import round_channels

__all__ = ['SEBlock']


class SEBlock(torch.nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    """

    def __init__(self, channels, reduction=16, round_mid=False, use_conv=True):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1,
                                         stride=1, groups=1, bias=True)
        else:
            self.fc1 = torch.nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = torch.nn.ReLU(inplace=True)
        if use_conv:
            self.conv2 = torch.nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1,
                                         stride=1, groups=1, bias=True)
        else:
            self.fc2 = torch.nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x
