"""Multipath 3D Convolution Encoder Module (ME module).

The ME module is the core contribution of ME-RAC. It replaces the standard
single-path 3D convolution with three parallel branches of different kernel
sizes, enabling the model to capture temporal features at multiple receptive
fields simultaneously:

- Path 1: 1x1x1 conv (128 channels) — point-wise features
- Path 2: 1x1x1 -> 5x5x5 conv (256 channels) — large receptive field
- Path 3: 1x1x1 -> 3x3x3 conv (128 channels) — medium receptive field

Output channels: 128 + 256 + 128 = 512
"""

import torch
import torch.nn as nn


class multiPathConv3d(nn.Module):
    """Multipath 3D convolution module with three parallel branches."""

    def __init__(self):
        super(multiPathConv3d, self).__init__()

        # Path 1: 1x1x1 dimensionality reduction (768 -> 128)
        self.Conv3d_1x1x1_128 = nn.Conv3d(
            in_channels=768, out_channels=128,
            kernel_size=1, padding=0, dilation=1
        )

        # Path 2: 1x1x1 reduction (768 -> 256) followed by 5x5x5 conv
        self.Conv3d_1x1x1_256 = nn.Conv3d(
            in_channels=768, out_channels=256,
            kernel_size=1, padding=0, dilation=1
        )
        self.Conv3d_5x5x5_256 = nn.Conv3d(
            in_channels=256, out_channels=256,
            kernel_size=5, padding=(10, 2, 2), dilation=(5, 1, 1)
        )

        # Path 3: 1x1x1 reduction (768 -> 128) followed by 3x3x3 conv
        self.Conv3d_3x3x3_128 = nn.Conv3d(
            in_channels=128, out_channels=128,
            kernel_size=3, padding=(3, 1, 1), dilation=(3, 1, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, 768, T, H, W].

        Returns:
            Tensor of shape [batch, 512, T, H, W].
        """
        x1 = self.Conv3d_1x1x1_128(x)              # Path 1: [b, 128, T, H, W]

        x2 = self.Conv3d_1x1x1_256(x)              # Path 2: [b, 256, T, H, W]
        x2 = self.Conv3d_5x5x5_256(x2)

        x3 = self.Conv3d_1x1x1_128(x)              # Path 3: [b, 128, T, H, W]
        x3 = self.Conv3d_3x3x3_128(x3)

        return torch.cat((x1, x2, x3), dim=1)      # [b, 512, T, H, W]
