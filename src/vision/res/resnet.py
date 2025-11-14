from torch import Tensor
from typing import Any
import torch.nn as nn
from basic import ResidualBlock, Block, SequentialBlock


class ResNet(Block):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, core: Block | None = None) -> None:
        super().__init__()
        
        if core is None:
            core = SequentialBlock(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.core = core
        self.residual = ResidualBlock(core)
        
        
    def forward(self, x) -> Tensor:
        identity = x
        out = self.residual(x)
        if self.skip is not None:
            identity = self.skip(identity)
        return nn.functional.relu(out + identity)