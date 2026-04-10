"""
TSD: Temporal Sequence Decision framework.

This module implements the TSD (Temporal-Sequence Decision) framework proposed
in the paper. TSD is a two-stage system for long video repetitive action counting:

Stage 1: Object detection (YOLOv5) processes each frame to determine whether
    an action-mark object is present, filtering relevant frames.
Stage 2: Repetitive action counting (ME-RAC) counts actions within the
    filtered frame segments.

The framework uses a threshold-based state machine to decide when action
sequences start and end, enabling accurate counting in arbitrarily long videos.

Reference:
    Multipath 3D-Conv encoder and temporal-sequence decision for
    repetitive-action counting (Expert Systems with Applications, 2024)
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from ME_RAC import MultiPath3dConv_Encoder_RAC


def load_action_model(config, backbone_checkpoint, model_checkpoint, device='cuda:0'):
    """Load the ME-RAC action counting model with pretrained weights.

    Args:
        config (str): Path to the Video Swin Transformer config file.
        backbone_checkpoint (str): Path to pretrained backbone weights.
        model_checkpoint (str): Path to trained ME-RAC model checkpoint.
        device (str): Device to load the model on. Default: 'cuda:0'.

    Returns:
        nn.Module: Loaded ME-RAC model ready for inference.
    """
    model = MultiPath3dConv_Encoder_RAC(
        config=config,
        checkpoint=backbone_checkpoint,
        num_frames=64,
        scales=[1, 4, 8],
        OPEN=False
    )
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    return model


def prepare_frame_tensor(frames, num_samples=64):
    """Convert a list of video frames to a normalized input tensor.

    Args:
        frames (list[np.ndarray]): List of BGR video frames.
        num_samples (int): Number of frames to sample. Default: 64.

    Returns:
        torch.Tensor: Tensor of shape [1, 3, num_samples, 224, 224].
    """
    total = len(frames)
    tensors = []

    if num_samples <= total:
        indices = [i * total // num_samples for i in range(num_samples)]
    else:
        indices = list(range(total)) + [total - 1] * (num_samples - total)

    for idx in indices:
        frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        tensors.append(transforms.ToTensor()(frame))

    return torch.stack(tensors).permute(1, 0, 2, 3).unsqueeze(0)


def predict_action_count(frames, model, device):
    """Run action counting inference on a segment of frames.

    Args:
        frames (list[np.ndarray]): List of BGR video frames.
        model (nn.Module): Loaded ME-RAC model.
        device (torch.device): Device for inference.

    Returns:
        int: Predicted repetitive action count.
    """
    with torch.no_grad():
        model.eval()
        input_tensor = prepare_frame_tensor(frames, num_samples=64).to(device)
        density_map, _ = model(input_tensor)
    return int(torch.sum(density_map, dim=1).round().cpu().item())


def temporal_sequence_decision(
    video_source,
    yolo_model,
    action_model,
    device='cuda:0',
    k=0.3,
):
    """Run the TSD framework on a video for long repetitive action counting.

    The algorithm maintains a state machine with two states (action / no-action)
    determined by consecutive frame counts exceeding the threshold K.
    K is computed as k * total_frames, where k is the continuous cumulative
    threshold ratio (default 30% as described in the paper).

    When an action segment accumulates enough frames, it is sent to the
    ME-RAC model for counting.

    Args:
        video_source (str): Path to the input video file.
        yolo_model: Loaded YOLOv5 detection model (confidence and NMS thresholds
            should be configured on the model itself).
        action_model (nn.Module): Loaded ME-RAC action counting model.
        device (str): CUDA device. Default: 'cuda:0'.
        k (float): Continuous cumulative threshold ratio. The frame threshold
            K is computed as int(k * total_frames). Default: 0.3 (30%).

    Returns:
        int: Total predicted repetitive action count for the entire video.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_threshold = max(1, int(total_frames * k))

    is_action_started = False
    no_action_count = 0
    total_count = 0
    action_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Object detection to filter action-relevant frames
        results = yolo_model(frame)
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
        has_person = len(detections) > 0

        if has_person:
            action_frames.append(frame)
            no_action_count = 0

            # Accumulated enough frames -> run action counting
            if len(action_frames) >= frame_threshold:
                if not is_action_started:
                    is_action_started = True

                # Stage 2: Repetitive action counting on the segment
                count = predict_action_count(action_frames, action_model, device)
                total_count += count
                action_frames.clear()
        else:
            no_action_count += 1

            # Continue collecting frames if action is ongoing
            if action_frames and is_action_started:
                action_frames.append(frame)

                if len(action_frames) >= frame_threshold:
                    count = predict_action_count(action_frames, action_model, device)
                    total_count += count
                    action_frames.clear()
                    is_action_started = False

            # No action for long enough -> end action segment
            if no_action_count >= frame_threshold:
                is_action_started = False
                no_action_count = 0

                if action_frames:
                    count = predict_action_count(action_frames, action_model, device)
                    total_count += count
                    action_frames.clear()

    # Handle remaining frames
    if action_frames:
        count = predict_action_count(action_frames, action_model, device)
        total_count += count

    cap.release()
    return total_count


if __name__ == '__main__':
    print("TSD Framework - Temporal Sequence Decision")
    print("Usage: Integrate with YOLOv5 and ME-RAC for long video action counting.")
    print("See README.md for detailed instructions.")
