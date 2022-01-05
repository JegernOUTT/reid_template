import torch

__all__ = ['PoolHead']

POOL_TYPES = {
    'AdaptiveAvgPool2d': torch.nn.AdaptiveAvgPool2d,
    'AdaptiveMaxPool2d': torch.nn.AdaptiveMaxPool2d,
}


class PoolHead(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 reid_features_number: int,
                 dropout_rate: float = 0.,
                 pool_type: str = 'AdaptiveAvgPool2d'):
        super().__init__()
        self.pool = POOL_TYPES[pool_type](1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(input_channels, reid_features_number)
        self.dropout_rate = dropout_rate
        if self.dropout_rate != 0.:
            self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._initialize_weights()

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        if self.dropout_rate != 0.:
            x = self.dropout(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.zero_()
