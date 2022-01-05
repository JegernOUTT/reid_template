from typing import Dict, Tuple, List

import torch.nn as nn

__all__ = ['EmbeddingExtraTargetHead']


class EmbeddingExtraTargetHead(nn.Module):
    def __init__(self, input_channels, endpoints: List[Tuple[str, int]]):
        super(EmbeddingExtraTargetHead, self).__init__()
        self.input_channels = input_channels
        self.endpoints = endpoints
        for name, ch in self.endpoints:
            self.add_module(name, nn.Sequential(
                nn.Linear(input_channels, input_channels // 2),
                nn.ReLU(),
                nn.Linear(input_channels // 2, ch)
            ))

    def forward(self, x):
        return [
            self.__getattr__(name)(x)
            for name, _ in self.endpoints
        ]
