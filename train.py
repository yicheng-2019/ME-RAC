"""
Training entry point for ME-RAC / TransRAC.

Usage:
    python train.py --data_dir ./data/RepCountA/train \
                    --label_file ./data/RepCountA/train.csv \
                    --val_dir ./data/RepCountA/valid \
                    --val_label ./data/RepCountA/valid.csv

See README.md for detailed setup instructions.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from ME_RAC import MultiPath3dConv_Encoder_RAC
from models.TransRAC import TransferModel
from dataset.RepCountA_Loader import RepCountADataset
from training.train_looping import train_one_epoch, validate
from TSRC import temporal_sequence_random_combination


def parse_args():
    parser = argparse.ArgumentParser(description='Train ME-RAC or TransRAC')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training NPZ data directory')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to training CSV annotation file')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to validation NPZ data directory')
    parser.add_argument('--val_label', type=str, default=None,
                        help='Path to validation CSV annotation file')
    parser.add_argument('--config', type=str,
                        default='./configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py',
                        help='Path to Video Swin Transformer config')
    parser.add_argument('--backbone_ckpt', type=str,
                        default='./pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth',
                        help='Path to pretrained backbone checkpoint')
    parser.add_argument('--model', type=str, default='me_rac',
                        choices=['me_rac', 'transrac'],
                        help='Model to train: me_rac or transrac')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=8e-7)
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--scales', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--use_tsrc', action='store_true',
                        help='Enable TSRC data augmentation')
    parser.add_argument('--tsrc_prob', type=float, default=0.3,
                        help='Probability of applying TSRC augmentation')
    return parser.parse_args()


def build_model(args):
    """Build the model based on args."""
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
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build model
    model = build_model(args)
    model = model.to(device)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Data
    train_dataset = RepCountADataset(
        args.data_dir, args.label_file, args.num_frames,
        use_tsrc=args.use_tsrc, tsrc_prob=args.tsrc_prob
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if args.val_dir and args.val_label:
        val_dataset = RepCountADataset(args.val_dir, args.val_label, args.num_frames)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer and AMP
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)

    best_mae = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_metrics['loss']:.4f} "
              f"MAE: {train_metrics['mae']:.4f} "
              f"OBO: {train_metrics['obo']:.4f}")

        # Validate
        if val_loader and epoch >= 50:
            val_metrics = validate(model, val_loader, device)
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"MAE: {val_metrics['mae']:.4f} "
                  f"OBO: {val_metrics['obo']:.4f}")

            # Save best model
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                save_path = os.path.join(args.save_dir, f'{epoch+1}_{best_mae:.4f}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mae': best_mae,
                }, save_path)
                print(f"  Saved best model: {save_path}")

    print(f"Training complete. Best MAE: {best_mae:.4f}")


if __name__ == '__main__':
    main()
