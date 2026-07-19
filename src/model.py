"""Simple CNN model for spectral instrument classification."""

from __future__ import annotations

import torch
from torch import nn

from src import config


class ConvBlock(nn.Module):
    """Convolution, normalization, activation and pooling block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InstrumentCNN(nn.Module):
    """Compact CNN that returns logits for multi-label classification."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        channels: tuple[int, ...] = (16, 32, 64),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for block_index, out_channels in enumerate(channels):
            block_dropout = min(0.10 + block_index * 0.05, 0.30)
            blocks.append(ConvBlock(in_channels, out_channels, dropout=block_dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int = config.NUM_CLASSES, model_size: str = "small") -> InstrumentCNN:
    """Create a CNN model by size name."""
    if model_size == "small":
        return InstrumentCNN(num_classes=num_classes, channels=(16, 32, 64), dropout=0.25)
    if model_size == "medium":
        return InstrumentCNN(num_classes=num_classes, channels=(16, 32, 64, 128), dropout=0.30)
    raise ValueError("model_size must be 'small' or 'medium'")
