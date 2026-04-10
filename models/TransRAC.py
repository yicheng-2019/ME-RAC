"""
TransRAC: Baseline model for repetitive action counting.

This is the baseline model (TransRAC) that ME-RAC builds upon.
TransRAC uses a standard single-path 3D convolution for encoding,
while ME-RAC replaces it with the multipath 3D-Conv encoder module.

Reference:
    TransRAC: Encoding Multi-scale Temporal Correlation with Transformers
    for Repetitive Action Counting (CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from torch.cuda.amp import autocast

from models.encoder_modules import Similarity_matrix, TransEncoder
from models.base_modules import DensityMapPredictor
from models.ME_Module import multiPathConv3d


class TransferModel(nn.Module):
    """TransRAC baseline model with optional ME module support.

    Args:
        config (str): Path to the Video Swin Transformer config file.
        checkpoint (str): Path to pretrained backbone weights. None to skip loading.
        num_frames (int): Number of input frames. Default: 64.
        scales (list[int]): Multi-scale temporal sampling scales. Default: [1, 4, 8].
        OPEN (bool): If True, fine-tune the backbone. Default: False.
        is_ME (bool): If True, use multipath 3D-Conv encoder (ME-RAC).
                      If False, use standard single-path 3D-Conv (TransRAC). Default: False.
    """

    def __init__(self, config, checkpoint=None, num_frames=64, scales=None,
                 OPEN=False, is_ME=False):
        super(TransferModel, self).__init__()
        if scales is None:
            scales = [1, 4, 8]
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN
        self.is_ME = is_ME

        self.backbone = self._load_backbone()

        # Temporal padding for multi-scale crops
        self.Replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.Replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)

        # Encoding module: single-path (TransRAC) or multi-path (ME-RAC)
        if self.is_ME:
            self.multiPath_conv3D = multiPathConv3d()
        else:
            self.conv3D = nn.Conv3d(
                in_channels=768, out_channels=512,
                kernel_size=3, padding=(3, 1, 1), dilation=(3, 1, 1)
            )

        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        # Temporal correlation via similarity matrix
        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(
            in_channels=4 * len(self.scales),
            out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)

        # Transformer encoder + density predictor
        self.transEncoder = TransEncoder(
            d_model=512, n_head=4, dropout=0.2,
            dim_ff=512, num_layers=1, num_frames=self.num_frames
        )
        self.FC = DensityMapPredictor(512, 512, 256, 1)

    def _load_backbone(self):
        """Load pretrained Video Swin Transformer backbone."""
        cfg = Config.fromfile(self.config)
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        if self.checkpoint:
            load_checkpoint(model, self.checkpoint, map_location='cpu')
        return model.backbone

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, channels, num_frames, height, width].

        Returns:
            density_map: Predicted density map [batch_size, num_frames].
            similarity_matrix: Multi-scale similarity matrix [batch_size, 4*num_scales, F, F].
        """
        with autocast():
            batch_size, c, num_frames, h, w = x.shape
            multi_scales = []

            for scale in self.scales:
                # Multi-scale temporal cropping
                if scale == 4:
                    x_padded = self.Replication_padding2(x)
                    crops = [x_padded[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                elif scale == 8:
                    x_padded = self.Replication_padding4(x)
                    crops = [x_padded[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                else:
                    crops = [x[:, :, i:i + 1, :, :] for i in range(self.num_frames)]

                # Feature extraction with Video Swin Transformer
                slices = []
                if not self.OPEN:
                    with torch.no_grad():
                        for crop in crops:
                            slices.append(self.backbone(crop))
                else:
                    for crop in crops:
                        slices.append(self.backbone(crop))

                x_scale = torch.cat(slices, dim=2)  # [b, 768, f, 7, 7]

                # Encoding: single-path or multi-path
                if self.is_ME:
                    x_scale = F.relu(self.bn1(self.multiPath_conv3D(x_scale)))
                else:
                    x_scale = F.relu(self.bn1(self.conv3D(x_scale)))

                x_scale = self.SpatialPooling(x_scale)      # [b, 512, f, 1, 1]
                x_scale = x_scale.squeeze(3).squeeze(3)      # [b, 512, f]
                x_scale = x_scale.transpose(1, 2)            # [b, f, 512]

                # Similarity matrix
                x_sims = F.relu(self.sims(x_scale, x_scale, x_scale))  # [b, 4, f, f]
                multi_scales.append(x_sims)

            x = torch.cat(multi_scales, dim=1)  # [b, 4*num_scales, f, f]
            x_matrix = x

            x = F.relu(self.bn2(self.conv3x3(x)))  # [b, 32, f, f]
            x = self.dropout1(x)
            x = x.permute(0, 2, 3, 1)              # [b, f, f, 32]

            # Transformer encoder
            x = x.flatten(start_dim=2)              # [b, f, 32*f]
            x = F.relu(self.input_projection(x))    # [b, f, 512]
            x = self.ln1(x)
            x = x.transpose(0, 1)                   # [f, b, 512]
            x = self.transEncoder(x)
            x = x.transpose(0, 1)                   # [b, f, 512]

            # Density map prediction
            x = self.FC(x)                          # [b, f, 1]
            x = x.squeeze(2)                        # [b, f]

            return x, x_matrix
