from torch import Tensor
from .block import Block
import torch.nn as nn 
from typing import Any

class SequentialBlock(Block):
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: Tensor, *extra: Any) -> Tensor:
        for layer in self.layers:
            if extra:
                x = layer(x, *extra)
            else: 
                x = layer(x)
        return x