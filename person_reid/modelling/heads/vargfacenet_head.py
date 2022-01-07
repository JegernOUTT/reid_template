import torch

__all__ = ['VarGFaceNetHead']


class VarGFaceNetHead(torch.nn.Module):
    def __init__(self, input_channels: int, reid_features_number: int, input_conv_kernel_size: int):
        super(VarGFaceNetHead, self).__init__()
        self.embedding = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU6(inplace=False),
            torch.nn.Conv2d(1024, 1024, input_conv_kernel_size,
                            1, padding=0, groups=1024 // 8, bias=False),
            torch.nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=False)
        )

        self.fc = torch.nn.Linear(in_features=512, out_features=reid_features_number)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        out = self.fc(x)
        return out
