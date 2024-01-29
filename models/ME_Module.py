import torch
import torch.nn as nn

class multiPathConv3d(nn.Module):

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