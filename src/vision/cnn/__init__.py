from typing import Any, Optional, Tuple, Union
import torch.nn as nn
from .model import CNNBasic, CNNAdvance

def CNN(
    core: Optional[nn.Module] = None,
    nums_class: int = 10,
    input_shape: Tuple[int, int, int] = (3, 32, 32),
    **kwargs: Any
) -> Union[CNNBasic, CNNAdvance]:
    
    """
    Factory interface for CNN models.
    
    - If `core` IS provided -> Returns CNNAdvance (wraps your custom core).
    - If `core` is NOT provided -> Returns CNNBasic (builds a default CNN).
    
    Args:
        core: A pytorch nn.Module to use as the backbone.
        nums_class: Number of output classes.
        input_shape: (C, H, W) tuple.
        **kwargs: Extra arguments passed to CNNBasic (e.g., kernel_size, padding).
    """
    
    if core is not None:
        return CNNAdvance(
            core=core, 
            nums_class=nums_class, 
            input_shape=input_shape
        )
    else:

        return CNNBasic(
            nums_class=nums_class, 
            input_shape=input_shape, 
            **kwargs
        )

__all__ = ["CNN"]