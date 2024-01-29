import os
import os.path as osp
import numpy as np
import math
import random
import cv2
from torch.utils.data import Dataset
import torch
import csv


def temporal_sequence_random_combination(vid_length, labels, sample_lens=64):
    """
    This function generates a new temporal sequence by randomly combining segments from the original sequence.

    :param vid_length: Length of the original video.
    :param labels: A list of time points indicating the start and end of actions time points in the original video.
    :param sample_lens: Length of the new temporal sequence to be generated.
    :return: A tuple containing the new sample IDs, new labels, and the total frames in the new sequence.
    """

    # Splitting the original time points into start and end points
    start_time_points, end_time_points = labels[::2], labels[1::2]

    # Determining the number of full segments in the original sequence
    full_counts = len(start_time_points)

    # Early exit if there are no complete segments
    if full_counts == 0:
        return [], [], 0

    # Randomly selecting a number of segments to include in the new sequence
    rand_counts = random.randint(1, full_counts - 1)

    # Early exit if no segments are selected
    if rand_counts == 0:
        return [], [], 0

    # Randomly selecting start points and corresponding end points
    rand_start_ids = random.sample(range(full_counts), rand_counts)
    rand_start_ids.sort()
    rand_end_times = [end_time_points[i] for i in rand_start_ids]

    # Generating new start and end points for the selected segments
    new_start_time_points, new_end_time_points = [], []
    for i, start_id in enumerate(rand_start_ids):
        if i == 0 or start_id == 0:
            new_start = start_time_points[start_id]
        else:
            new_start = new_end_time_points[-1] + start_time_points[start_id] - end_time_points[start_id - 1] + 1
        new_end = new_start + rand_end_times[i] - start_time_points[start_id]
        new_start_time_points.append(new_start)
        new_end_time_points.append(new_end)

    # Generating the new label sequence
    new_labels = [val for pair in zip(new_start_time_points, new_end_time_points) for val in pair]
    new_total_frames = new_end_time_points[-1] + vid_length - end_time_points[-1]

    # Sampling frames from the new sequence
    new_sample_frames = [(i * new_total_frames // sample_lens) for i in range(sample_lens)]
    if sample_lens > new_total_frames:
        new_sample_frames += [new_total_frames - 1] * (sample_lens - new_total_frames)

    # Mapping the sampled frames to the original video frame indices
    new_sample_ids_in_org = []
    for new_sample_frame in new_sample_frames:
        distances = [new_sample_frame - x for x in new_end_time_points]
        min_id = distances.index(min(distances, key=abs))

        if distances[min_id] <= 0:
            new_sample_ids_in_org.append(rand_end_times[min_id] + distances[min_id])
        else:
            min_id += (min_id < len(new_end_time_points) - 1)
            distance = new_sample_frame - new_end_time_points[min_id]
            new_sample_ids_in_org.append(rand_end_times[min_id] + distance)

    return new_sample_ids_in_org, new_labels, new_total_frames
