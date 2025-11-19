from basic import SequentialBlock, Block
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseCNN(Block, ABC):
    def __init__(
        self,
        nums_class: int = 10,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        core: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.nums_class = nums_class
        self.channels, self.height, self.width = input_shape

        self.core = core if core is not None else self.build_core()

        with torch.no_grad():
            dummy = torch.zeros(1, self.channels, self.height, self.width)
            out = self.core(dummy)
            feat_dim = out.flatten(1).shape[1]

        self.classifier = nn.Linear(feat_dim, self.nums_class)

    @abstractmethod
    def build_core(self) -> nn.Module:
        """
        Subclasses must implement this to return the feature extractor.
        """
        raise NotImplementedError

    def forward(self, x: Tensor, *extra: Any) -> Tensor:
        # shared forward for all CNN-like models
        if isinstance(self.core, Block):
            x = self.core(x, *extra)
        else:
            x = self.core(x)
        x = x.flatten(1)
        return self.classifier(x)

class CNNBasic(BaseCNN):
    def __init__(
        self,
        nums_class: int = 10,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.padding = padding
        super().__init__(nums_class=nums_class, input_shape=input_shape)

    def build_core(self) -> nn.Module:
        return SequentialBlock(
            nn.Conv2d(self.channels, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


class CNNAdvance(BaseCNN):
    def __init__(
        self,
        core: nn.Module,
        nums_class: int = 10,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
    ) -> None:
        super().__init__(nums_class=nums_class, input_shape=input_shape, core=core)

    def build_core(self) -> nn.Module:
        raise NotImplementedError("CNNAdvance expects an explicit `core`.")