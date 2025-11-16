import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any


class Block(nn.Module, ABC):
    """
    Abstract base class for a neural network block.

    This class provides a common interface for all custom blocks
    (e.g., a residual block, an attention block). It ensures that
    any subclass is a valid nn.Module and implements a forward pass.
    """
    
    def __init__(self):
        """
        Initializes the Block.
        """
        super().__init__() 

    @abstractmethod
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Defines the abstract forward pass for the block.
        
        Subclasses must implement this method to define the block's logic.

        Args:
            x (Tensor): The input tensor.
            *args (Any): Additional positional arguments required by the block.
            **kwargs (Any): Additional keyword arguments (e.g., attention_mask).

        Returns:
            Tensor: The output tensor from the block.
        """
        ...