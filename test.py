"""test RAC model"""
import os
from dataset.RepCountA_raw_Loader import MyData
from ME_RAC import MultiPath3dConv_Encoder_RAC
from testing.test_looping import test_loop

N_GPU = 1
device_ids = [i for i in range(N_GPU)]

# dataset dir
root_path = "data/LLSP"

test_video_dir = 'test'
test_label_dir = 'test.csv'

# video swin transformer pretrained model and config
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
# checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'

# ME-RAC trained model checkpoint, we will upload soon.
lastckpt = "weights/ME-RAC_TSRC_LLSP_SOTA.pt"

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]
test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = MultiPath3dConv_Encoder_RAC(
    config=config, checkpoint=None, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 1

test_loop(NUM_EPOCHS, my_model, test_dataset, lastckpt=lastckpt)
