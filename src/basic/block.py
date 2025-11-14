import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor

class Block(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs) -> Tensor:
        ...