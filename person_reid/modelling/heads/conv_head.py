import torch

from person_reid.modelling.common import batch_norm1d_trt_friendly, LinearBlock, l2_norm

__all__ = ['ConvHead']


class ConvHead(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 reid_features_number: int,
                 input_conv_kernel_size: int,
                 dropout_rate: float = 0.,
                 norm: bool = False):
        super().__init__()
        self.conv = LinearBlock(input_channels, input_channels, groups=input_channels,
                                kernel=(input_conv_kernel_size, input_conv_kernel_size),
                                stride=(1, 1), padding=(0, 0))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(input_channels, reid_features_number)
        self.bn = torch.nn.BatchNorm1d(reid_features_number)
        self.dropout_rate = dropout_rate
        self.norm = norm
        if self.dropout_rate != 0.:
            self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        if self.dropout_rate != 0.:
            x = self.dropout(x)
        x = self.linear(x)
        if self.training:
            x = self.bn(x)
        else:
            x = batch_norm1d_trt_friendly(x, self.bn)
        if self.norm:
            x = l2_norm(x)
        return x

    def _initialize_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.zero_()

        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
