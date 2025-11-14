from .block import Block
from torch import Tensor
from typing import Any
import torch.nn as nn
from .sequential import SequentialBlock


class ResidualBlock(Block):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor, *extra: Any) -> Tensor:
        return x + self.block(x, *extra))