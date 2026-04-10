"""
Testing entry point for ME-RAC / TransRAC.

Usage:
    python test.py --data_dir ./data/RepCountA/test \
                   --label_file ./data/RepCountA/test.csv \
                   --checkpoint ./checkpoints/best.pt

See README.md for detailed setup instructions.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ME_RAC import MultiPath3dConv_Encoder_RAC
from models.TransRAC import TransferModel
from dataset.RepCountA_Loader import RepCountADataset
from testing.test_looping import test


def parse_args():
    parser = argparse.ArgumentParser(description='Test ME-RAC or TransRAC')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test NPZ data directory')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to test CSV annotation file')
    parser.add_argument('--config', type=str,
                        default='./configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py',
                        help='Path to Video Swin Transformer config')
    parser.add_argument('--backbone_ckpt', type=str, default=None,
                        help='Path to pretrained backbone checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='me_rac',
                        choices=['me_rac', 'transrac'],
                        help='Model to test: me_rac or transrac')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--scales', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build model
    if args.model == 'me_rac':
        model = MultiPath3dConv_Encoder_RAC(
            config=args.config,
            checkpoint=args.backbone_ckpt,
            num_frames=args.num_frames,
            scales=args.scales,
            OPEN=False,
        )
    else:
        model = TransferModel(
            config=args.config,
            checkpoint=args.backbone_ckpt,
            num_frames=args.num_frames,
            scales=args.scales,
            OPEN=False,
            is_ME=False,
        )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device)

    # Data
    test_dataset = RepCountADataset(args.data_dir, args.label_file, args.num_frames)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Test
    metrics = test(model, test_loader, device)

    print("=" * 50)
    print("Test Results:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  OBO:  {metrics['obo']:.4f}")
    print(f"  OB10: {metrics['ob10']:.4f}")
    print(f"  OB20: {metrics['ob20']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
