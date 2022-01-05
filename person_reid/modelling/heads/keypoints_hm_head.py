from typing import List

import torch

from person_reid.modelling.common import kaiming_init, bias_init_with_prob, normal_init

__all__ = ['KeypointsHmHead']


def conv_block(input_channels, output_channels, kernel, act: bool = True):
    blocks = [
        torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel,
                        stride=(1, 1), padding=(1, 1), bias=False),
        torch.nn.BatchNorm2d(output_channels),
    ]
    if act:
        blocks.append(torch.nn.ReLU(inplace=False))
    return torch.nn.Sequential(*blocks)


def create_head(input_channels, cfg):
    layers = []
    for idx, layer_ch in enumerate(cfg[:-1]):
        layers.append(conv_block(input_channels, layer_ch, 3))
        input_channels = layer_ch
    layers.append(torch.nn.Conv2d(input_channels, cfg[-1], kernel_size=(1, 1),
                                  stride=(1, 1), padding=(0, 0), bias=True))
    return torch.nn.Sequential(*layers)


class KeypointsHmHead(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hm_cfg: List[int],
                 kps_cfg: List[int]):
        super().__init__()
        self.hm_head = create_head(input_channels, hm_cfg)
        self.kps_head = create_head(input_channels, kps_cfg)
        self._initialize_weights()

    def forward(self, x):
        return self.hm_head(x), self.kps_head(x)

    def _initialize_weights(self):
        def _head_init(head):
            for m in head.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.SyncBatchNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                if isinstance(m, torch.nn.Conv2d):
                    kaiming_init(m)

        _head_init(self.hm_head)
        normal_init(self.hm_head[-1], std=0.01, bias=bias_init_with_prob(0.01))
        _head_init(self.kps_head)
