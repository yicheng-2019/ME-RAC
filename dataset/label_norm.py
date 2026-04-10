"""
Density map label generation for repetitive action counting.

Converts discrete temporal annotations (action start/end frame pairs)
into continuous Gaussian density maps used as training targets.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


def gen_density_label(labels, num_frames=64):
    """Generate a Gaussian density map from temporal action annotations.

    Each action interval contributes a Gaussian-weighted segment to the
    density map, where the integral over each action sums to 1.0.
    This provides a smoother training target than binary labels.

    Args:
        labels (list[int]): Flat list of temporal annotations as alternating
            [start1, end1, start2, end2, ...] frame indices.
        num_frames (int): Number of sampled frames (density map length).
            Default: 64.

    Returns:
        np.ndarray: Density map of shape [num_frames] where each value
            represents the density contribution at that temporal position.
    """
    density = np.zeros(num_frames, dtype=np.float32)

    if len(labels) < 2:
        return density

    # Parse start/end pairs
    starts = labels[0::2]
    ends = labels[1::2]

    for start, end in zip(starts, ends):
        if end <= start:
            continue

        interval = end - start
        sigma = interval / 6.0
        mu = (start + end) / 2.0

        # Compute density for each frame in the interval
        for i in range(max(0, int(start)), min(num_frames, int(end) + 1)):
            # Integrate the Gaussian PDF over the frame interval [i, i+1]
            val, _ = quad(norm.pdf, i, i + 1, args=(mu, sigma))
            density[i] += val

    return density
