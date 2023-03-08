import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size: int,
                 batch_norm: bool = False
                 ) -> None:
        super(ConvBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x
