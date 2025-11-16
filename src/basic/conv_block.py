from torch import Tensor
import torch.nn as nn
from basic import Block, SequentialBlock
from typing import Optional

class ConvBlockBase(Block):
    Conv = None
    Pool = None
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int , stride: Optional[int], padding: int, pool_size: Optional[int] = None) -> None:
        super().__init__()
        
        self.layers = [
            self.Conv(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(),
        ]
        
        if pool_size is not None and stride is not None:
            self.layers.append(self.Pool(pool_size, stride))