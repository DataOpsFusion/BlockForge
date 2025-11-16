from torch import Tensor
import torch.nn as nn
from basic import ResidualBlock, Block, SequentialBlock
from typing import Sequence


class ResNet(Block):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
        layers: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        """
        One configurable ResNet:

        - `channels`: feature width per stage
        - `layers`:  number of residual blocks per stage

        Examples:
          ResNet(num_classes=10)
          ResNet(num_classes=10, layers=(3,4,6,3))
          ResNet(num_classes=10, layers=(8,8,8,8))
          ResNet(num_classes=10, channels=(32,64,128), layers=(4,4,4))
        """
        super().__init__()
        assert len(channels) == len(layers), "channels and layers must have same length"

        self.inplanes = channels[0]
        self.stem = SequentialBlock(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = channels[0]
        for stage_out_ch, num_blocks in zip(channels, layers):
            stage = self._make_stage(in_ch, stage_out_ch, num_blocks)
            stages.append(stage)
            in_ch = stage_out_ch

        self.stages = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        stride = 1 if in_ch == out_ch else 2
        layers.append(_ResBlock(in_ch, out_ch, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(_ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)