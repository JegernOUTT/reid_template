import math

from torch import nn


# Find the second norm of Input, divide its input by its modulus length
# Angular distillation Loss needs to be used
# Variable group convolution, S represents the number of channels per channel
def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels // S,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU()
    )


# For pointwise convolution, the kernel size here is all 1, but should we also group here? ?
def PointConv(in_channels, out_channels, stride, S, isPReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, groups=in_channels // S, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU() if isPReLU else nn.Sequential()
    )


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            # nn.ReLU6(inplace=True),
            nn.ReLU6(inplace=False),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            # nn.ReLU6(inplace=True), # Actually it should be sigmoid here
            nn.ReLU6(inplace=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


class NormalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, S=8):
        super(NormalBlock, self).__init__()
        out_channels = 2 * in_channels
        self.vargconv1 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv1 = PointConv(out_channels, in_channels, stride, S, isPReLU=True)

        self.vargconv2 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv2 = PointConv(out_channels, in_channels, stride, S, isPReLU=False)

        self.se = SqueezeAndExcite(in_channels, in_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = x
        x = self.pointconv1(self.vargconv1(x))
        x = self.pointconv2(self.vargconv2(x))
        x = self.se(x)
        out = out + x
        return self.prelu(out)


class DownSampling(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=2, S=8):
        super(DownSampling, self).__init__()
        out_channels = 2 * in_channels

        self.branch1 = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=True)
        )

        self.branch2 = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=True)
        )

        self.block3 = nn.Sequential(
            VarGConv(out_channels, 2 * out_channels, kernel_size, 1, S),  # stride =1
            PointConv(2 * out_channels, out_channels, 1, S, isPReLU=False)
        )  # The above branch

        self.shortcut = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=False)
        )

        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.shortcut(x)

        x1 = x2 = x
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = x1 + x2
        x3 = self.block3(x3)

        out = out + x3
        return self.prelu(out)


class HeadSetting(nn.Module):
    def __init__(self, in_channels, kernel_size, S=8):
        super(HeadSetting, self).__init__()
        self.block = nn.Sequential(
            VarGConv(in_channels, in_channels, kernel_size, 2, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=True),
            VarGConv(in_channels, in_channels, kernel_size, 1, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=False)
        )

        self.short = nn.Sequential(
            VarGConv(in_channels, in_channels, kernel_size, 2, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=False),
        )

    def forward(self, x):
        out = self.short(x)
        x = self.block(x)
        return out + x


class VarGFaceNet(nn.Module):
    def __init__(self):
        super(VarGFaceNet, self).__init__()
        S = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            # nn.ReLU6(inplace=True)
            nn.ReLU6(inplace=False)
        )
        self.head = HeadSetting(40, 3)
        self.stage2 = nn.Sequential(  # 1 normal 2 down
            DownSampling(40, 3, 2),
            NormalBlock(80, 3, 1),
            NormalBlock(80, 3, 1)
        )

        self.stage3 = nn.Sequential(
            DownSampling(80, 3, 2),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
        )

        self.stage4 = nn.Sequential(
            DownSampling(160, 3, 2),
            NormalBlock(320, 3, 1),
            NormalBlock(320, 3, 1),
            NormalBlock(320, 3, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    @property
    def output_channels(self):
        return 320
