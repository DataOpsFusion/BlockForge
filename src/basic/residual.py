from .block import Block
from torch import Tensor
from typing import Any
import torch.nn as nn


class ResidualBlock(Block):
    """
    Generic residual wrapper: y = x + block(x)
    """
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.block(x, *args, **kwargs)