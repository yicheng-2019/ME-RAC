"""
Video to NPZ conversion tool.

Converts raw video files (.mp4) to compressed NPZ format for faster
data loading during training. Each NPZ file contains uniformly sampled
frames resized to 224x224.

Usage:
    python tools/video2npz.py --input_dir ./data/raw_videos \
                               --output_dir ./data/npz \
                               --num_frames 64
"""

import argparse
import os

import cv2
import numpy as np


def video_to_npz(video_path, output_path, num_frames=64, size=(224, 224)):
    """Convert a single video to NPZ format.

    Args:
        video_path (str): Path to input video file.
        output_path (str): Path to output NPZ file.
        num_frames (int): Number of frames to sample. Default: 64.
        size (tuple): Output frame size (H, W). Default: (224, 224).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open {video_path}, skipping.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: No frames in {video_path}, skipping.")
        cap.release()
        return

    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        return

    # Sample frames uniformly
    sampled = []
    if num_frames <= total_frames:
        indices = [i * total_frames // num_frames for i in range(num_frames)]
        sampled = [all_frames[i] for i in indices]
    else:
        sampled = all_frames + [all_frames[-1]] * (num_frames - total_frames)

    # [F, H, W, C] -> [F, C, H, W]
    frames_array = np.array(sampled).transpose(0, 3, 1, 2).astype(np.uint8)

    np.savez_compressed(output_path, imgs=frames_array, num_frames=total_frames)
    print(f"Saved: {output_path} ({total_frames} frames -> {num_frames} sampled)")


def main():
    parser = argparse.ArgumentParser(description='Convert videos to NPZ format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save NPZ files')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to sample per video')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(args.input_dir)
              if os.path.splitext(f)[1].lower() in video_exts]

    print(f"Found {len(videos)} videos in {args.input_dir}")

    for video_name in sorted(videos):
        video_path = os.path.join(args.input_dir, video_name)
        npz_name = os.path.splitext(video_name)[0] + '.npz'
        output_path = os.path.join(args.output_dir, npz_name)
        video_to_npz(video_path, output_path, args.num_frames)

    print("Conversion complete.")


if __name__ == '__main__':
    main()
