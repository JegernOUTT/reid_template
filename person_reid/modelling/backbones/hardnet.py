import collections
from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm


def filter_by_out_idices(forward_func):
    def _filter_func(self, x):
        outputs = forward_func(self, x)
        if self._out_indices is None:
            return outputs[-1]
        return tuple([
            outputs[idx]
            for idx in self._out_indices
        ])

    return _filter_func


class BaseBackbone(nn.Module):
    def __init__(self, out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__()
        self._out_indices = out_indices


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, stride=1, bias=False):
        super().__init__()
        groups = in_channels
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.out_channels = out_channels
        out_ch = out_channels
        groups = 1
        pad = kernel // 2 if padding == 0 else padding
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=pad, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)


class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels


class HarDBlock_v2(nn.Module):
    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum(self.out_partition[i])
            real_out_ch = self.out_partition[i][0]
            conv_layers_.append(nn.Conv2d(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias=True))
            bnrelu_layers_.append(BRLayer(real_out_ch))
            cur_ch = real_out_ch
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += real_out_ch
        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        # Transform weight matrix from a pretrained HarDBlock v1
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k - 1][0].weight.shape[0] if k > 0 else
                       blk.layers[0][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias

            self.conv_layers[i].weight[0:part[0], :, :, :] = w_src[:, 0:in_ch, :, :]
            self.layer_bias.append(b_src)
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    # for pytorch, add bias with standalone tensor is more efficient than within conv.bias
                    # this is because the amount of non-zero bias is small,
                    # but if we use conv.bias, the number of bias will be much larger
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link)):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chos = sum(self.out_partition[ly][0:part_id])
                    choe = chos + part[0]
                    chis = sum(link_ch[0:j])
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :, :, :] = w_src[:, chis:chie, :, :]

            # update BatchNorm or remove it if there is no BatchNorm in the v1 block
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2d):
                self.bnrelu_layers[i] = nn.Sequential(
                    blk.layers[i][1],
                    blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:, 0:part[0], :, :] if len(part) > 1 else xout
            if self.layer_bias[i] is not None:
                xin += self.layer_bias[i].view(1, -1, 1, 1)

            if len(link) > 1:
                for j in range(len(link) - 1):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chs = sum(self.out_partition[ly][0:part_id])
                    che = chs + part[0]

                    xin += layers_[ly][:, chs:che, :, :]

            xin = self.bnrelu_layers[i](xin)

            if i % 2 == 0 or i == len(self.conv_layers) - 1:
                outs_.append(xin)

        out = torch.cat(outs_, 1)
        return out


class HarDNet(BaseBackbone):
    def __init__(self, depth_wise, first_ch, ch_list, gr_mul, gr,
                 n_layers, down_samp, skip_nodes, arch_name,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super(HarDNet, self).__init__(out_indices)

        self._arch_name = arch_name
        self._depth_wise = depth_wise
        self._skip_nodes = skip_nodes

        second_kernel = 3
        max_pool = True

        if self._depth_wise:
            second_kernel = 1
            max_pool = False

        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                      stride=2, bias=False))

        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], gr_mul, n_layers[i], dwconv=self._depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if down_samp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, stride=2))

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, boo, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    @filter_by_out_idices
    def forward(self, x):
        xs = []
        for idx, m in enumerate(self.base):
            x = m(x)
            if idx in self._skip_nodes:
                xs.append(x)
        xs.append(x)
        return xs


class HarDNetDet(BaseBackbone, BaseModule):
    def __init__(self, depth_wise, first_ch, ch_list, gr_mul, gr,
                 n_layers, skip_nodes, arch_name,
                 init_cfg=None, out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        BaseBackbone.__init__(self, out_indices)
        BaseModule.__init__(self, init_cfg)

        self._arch_name = arch_name
        self._depth_wise = depth_wise
        self._skip_nodes = skip_nodes

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                                   stride=2, bias=False))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], gr_mul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i != len(n_layers) - 1:
                self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if i == 0:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
            elif i != len(n_layers) - 1 and i != 1 and i != 3:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, boo, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    @filter_by_out_idices
    def forward(self, x):
        xs = []
        for idx, m in enumerate(self.base):
            x = m(x)
            if idx in self._skip_nodes:
                xs.append(x)
        xs.append(x)
        return xs


class HarDNet39Det(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet39',
            depth_wise=False,
            first_ch=[24, 48],
            ch_list=[96, 128, 256, 512],
            gr_mul=1.7,
            gr=[12, 14, 18, 40],
            n_layers=[4, 16, 8, 4],
            skip_nodes=[1, 3, 8, 11],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 196


class HarDNet39sDet(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet39',
            depth_wise=True,
            first_ch=[24, 48],
            ch_list=[96, 128, 256, 512],
            gr_mul=1.7,
            gr=[12, 14, 18, 40],
            n_layers=[4, 16, 8, 4],
            skip_nodes=[1, 3, 8, 11],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 196


class HarDNet68Det(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet68',
            depth_wise=False,
            first_ch=[32, 64],
            ch_list=[128, 256, 320, 640],
            gr_mul=1.7,
            gr=[14, 16, 20, 40],
            n_layers=[8, 16, 16, 16],
            skip_nodes=[1, 3, 8, 13],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 654


class HarDNet68sDet(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet68',
            depth_wise=True,
            first_ch=[32, 64],
            ch_list=[128, 256, 320, 640],
            gr_mul=1.7,
            gr=[14, 16, 20, 40],
            n_layers=[8, 16, 16, 16],
            skip_nodes=[1, 3, 8, 13],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 654


class HarDNet85Det(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3, 4),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet85',
            depth_wise=False,
            first_ch=[48, 96],
            ch_list=[192, 256, 320, 480, 720],
            gr_mul=1.7,
            gr=[24, 24, 28, 36, 48],
            n_layers=[8, 16, 16, 16, 16],
            skip_nodes=[1, 3, 8, 13],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 784


class HarDNet85sDet(HarDNetDet):
    def __init__(self,
                 out_indices: Optional[Sequence[int]] = (0, 1, 2, 3, 4),
                 init_cfg=None):
        super().__init__(
            arch_name='hardnet85',
            depth_wise=True,
            first_ch=[48, 96],
            ch_list=[192, 256, 320, 480, 720],
            gr_mul=1.7,
            gr=[24, 24, 28, 36, 48],
            n_layers=[8, 16, 16, 16, 16],
            skip_nodes=[1, 3, 8, 13],
            init_cfg=init_cfg,
            out_indices=out_indices)

    @property
    def output_channels(self):
        return 784
