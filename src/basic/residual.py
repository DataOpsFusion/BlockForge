from .block import Block
from torch import Tensor
from typing import Any
import torch.nn as nn


class ResidualBlock(Block):
    """
    A generic residual block implementing the transformation:
    
        y = skip(x) + layer(x)
    
    where `skip(x) = x` if no skip/projection module is provided.
    
    This block wraps any module (`layer`) and adds a residual connection
    between its input and output. If the input and output shapes differ
    (e.g., due to stride or channel changes), an optional `skip` module
    can be provided to project the input into the correct shape before
    adding.
    
    Parameters
    ----------
    layer : nn.Module
        The main transformation applied to the input. This is usually a
        sequence of convolution, normalization, activation, or MLP layers.
        It must output a tensor that is compatible with the output of
        `skip(x)` for the elementwise addition.
        
    skip : nn.Module or None, optional
        An optional projection module applied to `x` before the residual
        addition. This is required when the shapes of `x` and `layer(x)`
        do not match (e.g., channel changes or stride > 1). If None
        (default), the input `x` is used directly.
    
    Notes
    -----
    - `ResidualBlock` is a generic residual wrapper and can be used in
      convolutional networks, MLPs, or transformer-like architectures.
    - This block does NOT apply an activation after the residual addition.
      If needed, wrap this block inside another module.
    """
    def __init__(self, layer: nn.Module, skip: nn.Module | None = None) -> None:
        super().__init__()
        self.layer = layer
        self.skip = skip
    
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass of the residual block.

        Parameters
        ----------
        x : Tensor
            Input tensor to the block.

        Returns
        -------
        Tensor
            Output of the residual addition: `layer(x) + skip(x)` or
            `layer(x) + x` when no skip module is provided.
        """
        identity = x if self.skip is None else self.skip(x)
        out = self.layer(x, *args, **kwargs)
        return out + identity