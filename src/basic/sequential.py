from torch import Tensor
from .block import Block
import torch.nn as nn 
from typing import Any

class SequentialBlock(Block):
    """
    A sequential container similar to ``nn.Sequential`` but extended to support
    optional extra arguments for custom ``Block`` layers.

    This module evaluates each layer in order. If a layer is an instance of
    ``Block``, it is called with the full signature ``layer(x, *extra)``.
    Standard PyTorch modules (e.g. ``nn.Linear``, ``nn.ReLU``) are called with
    the vanilla signature ``layer(x)`` to avoid mismatched call signatures.

    This design allows mixing regular PyTorch layers with custom blocks that
    accept additional arguments.

    Parameters
    ----------
    *layers : nn.Module
        A variable-length sequence of modules to be applied in order.
        Layers may be either standard ``nn.Module`` objects or subclasses
        of ``Block``.
    """
    def __init__(self, *layers: nn.Module) -> None:
        """
        Initialize the sequential block by storing all layers internally.

        Parameters
        ----------
        *layers : nn.Module
            Modules to run sequentially.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: Tensor, *extra: Any) -> Tensor:
        """
        Forward pass through the sequence of layers.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        *extra : Any
            Optional extra arguments. These are forwarded **only** to
            layers that subclass ``Block``. Regular PyTorch modules
            ignore extra arguments.

        Returns
        -------
        Tensor
            The output of the final layer.
        """
        for layer in self.layers:
            if isinstance(layer, Block):
                x = layer(x, *extra)
            else:
                x = layer(x)
        return x