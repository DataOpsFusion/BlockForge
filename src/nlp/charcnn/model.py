from abc import abstractmethod
from vision.cnn import CNN
from torch import nn
import torch
from basic import SequentialBlock, Block

class BaseCharCNN(CNN):
    def __init__(
        self,
        embed_dim: int = 70,
        input_length: int = 1014,
        nums_class: int = 4,
    ) -> None:
        self.embed_dim = embed_dim
        self.input_length = input_length
        super().__init__(nums_class=nums_class, input_shape=(embed_dim, input_length))
        
    @abstractmethod
    def build_core(self) -> nn.Module:
        ...
        
class CharCNNBasic(BaseCharCNN):
    def __init__(
        self
    ) -> None:
        super().__init__()


    def build_core(self) -> nn.Module:
        return SequentialBlock(
            nn.Conv1d(self.embed_dim, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
class CharCNNAdvance(BaseCharCNN):
    def __init__(
        self
    ) -> None:
        super().__init__()
        
        
        if self.core is None:
            self.build_core()
