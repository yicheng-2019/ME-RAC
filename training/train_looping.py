"""
Training loop for ME-RAC and TransRAC models.

Implements the training and validation loops with:
- MSE loss for density map prediction
- Smooth L1 loss for count prediction
- Mixed precision training (AMP)
- MAE and OBO metric tracking
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """Train the model for one epoch.

    Args:
        model (nn.Module): ME-RAC or TransRAC model.
        dataloader (DataLoader): Training data loader.
        optimizer: Optimizer instance.
        device: CUDA device.
        scaler (GradScaler, optional): AMP gradient scaler.

    Returns:
        dict: Training metrics {'loss', 'mae', 'obo'}.
    """
    model.train()
    mse_loss_fn = nn.MSELoss()
    smooth_l1_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_mae = 0.0
    total_obo = 0.0
    num_batches = 0

    for frames, density_gt, count_gt in dataloader:
        frames = frames.to(device)
        density_gt = density_gt.to(device)
        count_gt = count_gt.float().to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                density_pred, _ = model(frames)
                count_pred = torch.sum(density_pred, dim=1)

                loss_density = mse_loss_fn(density_pred, density_gt)
                loss_count = smooth_l1_fn(count_pred, count_gt)
                loss = loss_density + loss_count

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            density_pred, _ = model(frames)
            count_pred = torch.sum(density_pred, dim=1)

            loss_density = mse_loss_fn(density_pred, density_gt)
            loss_count = smooth_l1_fn(count_pred, count_gt)
            loss = loss_density + loss_count

            loss.backward()
            optimizer.step()

        # Metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(count_pred - count_gt) / (count_gt + 1e-1))
            obo = torch.mean((torch.abs(count_pred - count_gt) <= 1).float())

        total_loss += loss.item()
        total_mae += mae.item()
        total_obo += obo.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'mae': total_mae / max(num_batches, 1),
        'obo': total_obo / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model on the validation set.

    Args:
        model (nn.Module): ME-RAC or TransRAC model.
        dataloader (DataLoader): Validation data loader.
        device: CUDA device.

    Returns:
        dict: Validation metrics {'loss', 'mae', 'obo'}.
    """
    model.eval()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_mae = 0.0
    total_obo = 0.0
    num_batches = 0

    for frames, density_gt, count_gt in dataloader:
        frames = frames.to(device)
        density_gt = density_gt.to(device)
        count_gt = count_gt.float().to(device)

        density_pred, _ = model(frames)
        count_pred = torch.sum(density_pred, dim=1)

        loss = mse_loss_fn(density_pred, density_gt)
        mae = torch.mean(torch.abs(count_pred - count_gt) / (count_gt + 1e-1))
        obo = torch.mean((torch.abs(count_pred - count_gt) <= 1).float())

        total_loss += loss.item()
        total_mae += mae.item()
        total_obo += obo.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'mae': total_mae / max(num_batches, 1),
        'obo': total_obo / max(num_batches, 1),
    }
