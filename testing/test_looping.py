"""
Testing loop for ME-RAC and TransRAC models.

Evaluates the model on test datasets using MAE, OBO, and OBX metrics
as described in the paper.
"""

import torch


@torch.no_grad()
def test(model, dataloader, device):
    """Evaluate the model and compute all metrics.

    Args:
        model (nn.Module): ME-RAC or TransRAC model.
        dataloader (DataLoader): Test data loader.
        device: CUDA device.

    Returns:
        dict: Test metrics {'mae', 'obo', 'ob10', 'ob20'}.
    """
    model.eval()

    all_preds = []
    all_gts = []

    for frames, density_gt, count_gt in dataloader:
        frames = frames.to(device)
        density_pred, _ = model(frames)
        count_pred = torch.sum(density_pred, dim=1).round().cpu()
        all_preds.append(count_pred)
        all_gts.append(count_gt.float())

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    N = len(preds)

    # MAE: Mean Absolute Error (normalized)
    mae = torch.mean(torch.abs(preds - gts) / (gts + 1e-1)).item()

    # OBO: Off-By-One count error (|pred - gt| <= 1)
    obo = torch.mean((torch.abs(preds - gts) <= 1).float()).item()

    # OBX: Off-By-X% error (|pred - gt| / gt <= X%)
    relative_error = torch.abs(preds - gts) / (gts + 1e-1)
    ob10 = torch.mean((relative_error <= 0.10).float()).item()
    ob20 = torch.mean((relative_error <= 0.20).float()).item()

    return {'mae': mae, 'obo': obo, 'ob10': ob10, 'ob20': ob20}
