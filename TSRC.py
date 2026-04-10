"""
TSRC: Temporal Sequence Random Combination data augmentation.

This module implements the TSRC data augmentation method proposed in the paper.
TSRC prevents over-fitting by randomly selecting and recombining action segments
from the original training videos to generate new training samples with different
action counts and temporal distributions.

Reference:
    Multipath 3D-Conv encoder and temporal-sequence decision for
    repetitive-action counting (Expert Systems with Applications, 2024)
"""

import random


def temporal_sequence_random_combination(vid_length, labels, num_samples=64):
    """Generate a new training sample by randomly recombining action segments.

    Given the original video's temporal action annotations (start/end pairs),
    this function randomly selects a subset of actions and recombines them into
    a new virtual video with different action count and frame distribution.

    Args:
        vid_length (int): Total number of frames in the original video.
        labels (list[int]): Flat list of temporal annotations as alternating
            [start1, end1, start2, end2, ...] frame indices.
        num_samples (int): Number of frames to sample from the new virtual video.
            Default: 64.

    Returns:
        tuple: (new_sample_ids, new_labels, new_total_frames) or None if
            the video has fewer than 2 actions.
            - new_sample_ids (list[int]): Frame indices in the original video
              corresponding to each sampled frame position.
            - new_labels (list[int]): New temporal annotations for the
              recombined video as [start1, end1, start2, end2, ...].
            - new_total_frames (int): Total frame count of the virtual video.
    """
    # Parse start/end time points from flat label list
    start_points = []
    end_points = []
    full_counts = len(labels) // 2

    if full_counts < 2:
        return None

    for i in range(len(labels)):
        if (i + 1) % 2:
            start_points.append(labels[i])
        else:
            end_points.append(labels[i])

    # Randomly select a subset of actions
    rand_counts = random.randint(1, full_counts - 1)
    rand_indices = sorted(random.sample(range(len(start_points) - 1), rand_counts))
    rand_end_times = [end_points[i] for i in rand_indices]

    # Compute new temporal annotations for the recombined segments
    new_start_points = []
    new_end_points = []

    for i, j in enumerate(rand_indices):
        if i == 0:
            if j == 0:
                new_start = start_points[j]
                new_end = end_points[j]
            else:
                new_start = start_points[j] - start_points[j - 1]
                new_end = end_points[j] - start_points[j] + new_start
        else:
            new_start = new_end_points[-1] + start_points[j] - end_points[j - 1] + 1
            new_end = end_points[j] - start_points[j] + new_start

        new_start_points.append(new_start)
        new_end_points.append(new_end)

    # Build new label list
    new_labels = []
    for s, e in zip(new_start_points, new_end_points):
        new_labels.append(s)
        new_labels.append(e)

    # Calculate new total frames and sample frame positions
    new_total_frames = new_end_points[-1] + vid_length - end_points[-1]
    new_sample_frames = []

    if num_samples <= new_total_frames:
        new_sample_frames = [i * new_total_frames // num_samples for i in range(num_samples)]
    else:
        new_sample_frames = list(range(new_total_frames))
        new_sample_frames += [new_total_frames - 1] * (num_samples - new_total_frames)

    # Map new sample positions back to original video frame indices
    new_sample_ids = []
    for new_frame in new_sample_frames:
        distances = [new_frame - e for e in new_end_points]
        abs_distances = [abs(d) for d in distances]
        closest_idx = abs_distances.index(min(abs_distances))

        if distances[closest_idx] <= 0:
            new_sample_ids.append(rand_end_times[closest_idx] + distances[closest_idx])
        else:
            if closest_idx < len(new_end_points) - 1:
                closest_idx += 1
                offset = new_frame - new_end_points[closest_idx]
                new_sample_ids.append(rand_end_times[closest_idx] + offset)
            else:
                offset = new_frame - (new_total_frames - 1)
                new_sample_ids.append(vid_length - 1 + offset)

    return new_sample_ids, new_labels, new_total_frames
