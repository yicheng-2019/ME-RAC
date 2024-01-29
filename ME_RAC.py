import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from torch.cuda.amp import autocast


from models.encoder_modules import Similarity_matrix, TransEncoder
from models.base_modules import PositionalEncoding, DensityMapPredictor
from models.ME_Module import multiPathConv3d


class MultiPath3dConv_Encoder_RAC(nn.Module):
    def __init__(self, config,  checkpoint=None, num_frames=64, scales=[1, 4, 8], OPEN=False):
        super(MultiPath3dConv_Encoder_RAC, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN

        self.backbone = self.load_backbone()  # load pretrain model

        self.Replication_padding1 = nn.ConstantPad3d((0, 0, 0, 0, 1, 1), 0)
        self.Replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.Replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)

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


    def load_backbone(self):
        # load  pretrained model of video swin transformer by mmaction and mmcv
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # # # load hyperparameters by mmcv api
        if self.checkpoint:
            load_checkpoint(model, self.checkpoint, map_location='cpu')

        backbone = model.backbone

        return backbone

    def forward(self, x):
        # x: tensor([batch_size, channel, temporal_dim, height, width])
        with autocast():
            batch_size, c, num_frames, h, w = x.shape
            multi_scales = []
            for scale in self.scales:
                if scale == 4:
                    x = self.Replication_padding2(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                elif scale == 8:
                    x = self.Replication_padding4(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                else:
                    crops = [x[:, :, i:i + 1, :, :] for i in range(0, self.num_frames)]

                slice = []

                ## feature extract with video SwinTransformer
                if not self.OPEN:
                    with torch.no_grad():
                        for crop in crops:
                            crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]
                            slice.append(crop)
                else:  # train  the feature extractor (video SwinTransformer backbone)
                    for crop in crops:
                        crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]
                        slice.append(crop)

                x_scale = torch.cat(slice, dim=2)  # -> [b,768,f,size,size]
                # if self.is_ME:
                x_scale = F.relu(self.bn1(self.multiPath_conv3D(x_scale)))  # ->[b,512,f,7,7]

                x_scale = self.SpatialPooling(x_scale)  # ->[b,512,f,1,1]
                x_scale = x_scale.squeeze(3).squeeze(3)  # -> [b,512,f]
                x_scale = x_scale.transpose(1, 2)  # -> [b,f,512]

                # -------- similarity matrix ---------
                x_sims = F.relu(self.sims(x_scale, x_scale, x_scale))  # -> [b,4,f,f]
                multi_scales.append(x_sims)

            x = torch.cat(multi_scales, dim=1)  # [B,4*scale_num,f,f]
            x_matrix = x
            x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]
            x = self.dropout1(x)

            x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
            # --------- transformer encoder ------
            x = x.flatten(start_dim=2)  # ->[b,f,32*f]
            x = F.relu(self.input_projection(x))  # ->[b,f, 512]
            x = self.ln1(x)

            x = x.transpose(0, 1)  # [f,b,512]
            x = self.transEncoder(x)  #
            x = x.transpose(0, 1)  # ->[b,f, 512]

            x = self.FC(x)  # ->[b,f,1]
            x = x.squeeze(2)

            return x, x_matrix


if  __name__ == '__main__':
    model = MultiPath3dConv_Encoder_RAC(
        config ='./configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
    )
    model = nn.DataParallel(model.to("cuda:0"), device_ids=[0])
    input_tensor = torch.randn(2, 3, 64, 224, 224).to("cuda:0")  # N, C, T, H, W

    with torch.no_grad():
        model.eval()
        density_map, _ = model(input_tensor)
    print(density_map.shape)
