import torch
import torch.nn as nn
from encoder_modules import Similarity_matrix, TransEncoder
from base_modules import PositionalEncoding, DensityMapPredictor

class MultiPath3dConv_Encoder_RAC(nn.Module):
    def __init__(self, config, checkpoint, num_frames, scales,
                 OPEN=False,
                 # is_ME=False
                 ):
        super(MultiPath3dConv_Encoder_RAC, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN
        # self.is_ME = is_ME

        self.backbone = self.load_model()  # load pretrain model

        self.Replication_padding1 = nn.ConstantPad3d((0, 0, 0, 0, 1, 1), 0)
        self.Replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.Replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)

        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))

        self.multiPath_conv3D = multiPathConv3d()

        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.FC = DensityMapPredictor(512, 512, 256, 1)  #


class multiPathConv3d(nn.Module):
    # 加入一个基于multipath的新模块，替换原来的Conv3d模块

    def __init__(self):
        super(multiPathConv3d, self).__init__()
        self.Conv3d_3x3x3_128= nn.Conv3d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=3,
                                         padding=(3, 1, 1),
                                         dilation=(3, 1, 1))

        self.Conv3d_5x5x5_256 = nn.Conv3d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=5,
                                          padding=(10, 2, 2),
                                          dilation=(5, 1, 1))

        self.Conv3d_1x1x1_256 = nn.Conv3d(in_channels=768,
                                          out_channels=256,
                                          kernel_size=(1, 1, 1),
                                          padding=(0, 0, 0),
                                          dilation=(1, 1, 1))
        self.Conv3d_1x1x1_128 = nn.Conv3d(in_channels=768,
                                          out_channels=128,
                                          kernel_size=(1, 1, 1),
                                          padding=(0, 0, 0),
                                          dilation=(1, 1, 1))
        self.bn_256 = nn.BatchNorm3d(256)
        self.bn_128 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.Conv3d_1x1x1_128(x)  # path1

        x2 = self.Conv3d_1x1x1_256(x)  # path2
        x2 = self.Conv3d_5x5x5_256(x2)

        x3 = self.Conv3d_1x1x1_128(x)  # path3
        x3 = self.Conv3d_3x3x3_128(x3)
        out = torch.cat((x1, x2, x3), 1)

        return out
