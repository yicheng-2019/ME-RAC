"""
RepCount-A dataset loader for repetitive action counting.

Supports loading pre-processed NPZ files with optional TSRC data augmentation.
Each sample contains uniformly sampled frames and the corresponding
Gaussian density map label.

Reference:
    Hu et al., TransRAC: Encoding Multi-scale Temporal Correlation with
    Transformers for Repetitive Action Counting (CVPR 2022)
"""

import os
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.label_norm import gen_density_label
from TSRC import temporal_sequence_random_combination


class RepCountADataset(Dataset):
    """RepCount-A dataset for repetitive action counting.

    Args:
        data_dir (str): Path to the directory containing NPZ video files.
        label_file (str): Path to the CSV annotation file.
        num_frames (int): Number of frames to sample per video. Default: 64.
        use_tsrc (bool): Enable TSRC data augmentation. Default: False.
        tsrc_prob (float): Probability of applying TSRC per sample. Default: 0.3.
    """

    def __init__(self, data_dir, label_file, num_frames=64,
                 use_tsrc=False, tsrc_prob=0.3):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.use_tsrc = use_tsrc
        self.tsrc_prob = tsrc_prob
        self.samples = []

        self._parse_annotations(label_file)

    def _parse_annotations(self, label_file):
        """Parse CSV annotation file into sample list."""
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                video_name = row[0]
                npz_path = os.path.join(self.data_dir, video_name.replace('.mp4', '.npz'))
                if not os.path.exists(npz_path):
                    continue

                count = int(float(row[1])) if row[1] else 0
                time_points = []
                for val in row[2:]:
                    val = val.strip()
                    if val and val != 'L':
                        try:
                            time_points.append(float(val))
                        except ValueError:
                            continue

                self.samples.append({
                    'path': npz_path,
                    'count': count,
                    'time_points': time_points,
                    'name': video_name,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load pre-processed frames from NPZ
        data = np.load(sample['path'])
        frames = data['imgs']  # [num_frames, C, H, W]
        total_frames = int(data.get('num_frames', len(frames)))

        time_points = sample['time_points']

        # Apply TSRC data augmentation
        if self.use_tsrc and random.random() < self.tsrc_prob:
            result = temporal_sequence_random_combination(
                total_frames, time_points, num_samples=self.num_frames
            )
            if result is not None:
                sample_ids, time_points, _ = result
                # Re-sample frames according to TSRC indices
                sample_ids = [max(0, min(i, len(frames) - 1)) for i in sample_ids]
                frames = frames[sample_ids]

        # Normalize to [-1, 1]
        frames = (frames.astype(np.float32) - 127.5) / 127.5
        frames = torch.from_numpy(frames).permute(1, 0, 2, 3)  # [C, F, H, W]

        # Generate density map label
        density = gen_density_label(time_points, num_frames=self.num_frames)
        density = torch.from_numpy(density)

        return frames, density, sample['count']
