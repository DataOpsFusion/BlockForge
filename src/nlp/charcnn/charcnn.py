from basic import SequentialBlock, Block
from torch import Tensor
from typing import Any
import torch.nn as nn 

ALPHABET_SIZE = 70
INPUT_LENGTH = 1014


class CharCNN(Block):
    def __init__(self, in_ch: int, out_ch: int, core: Block | None = None) -> None:
        super().__init__()
        if core is None: 
            core = SequentialBlock(
                nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=2, padding=1),
            )
        self.core = core

    def forward(self, x: Tensor) -> Tensor:
        return self.core(x)