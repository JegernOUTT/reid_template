import torch

from person_reid.modelling.common import ConvBlock

__all__ = ['CBAM']


class MLP(torch.nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = torch.nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(in_features=mid_channels, out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(torch.nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(channels=channels, reduction_ratio=reduction_ratio)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        return x


class SpatialGate(torch.nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = ConvBlock(in_channels=2, out_channels=1, kernel_size=7,
                              stride=1, padding=3, bias=False, use_bn=True, activation=None)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        att1 = x.max(dim=1)[0].unsqueeze(1)
        att2 = x.mean(dim=1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.conv(att)
        att = self.sigmoid(att)
        x = x * att
        return x


class CBAM(torch.nn.Module):
    def __init__(self, input_channels: int, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ch_gate = ChannelGate(channels=input_channels, reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate()

    def forward(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x
