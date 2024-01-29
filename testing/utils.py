""" some tools """
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import csv

from matplotlib.lines import Line2D
import seaborn as sns


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    median_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            median_grads.append(p.grad.abs().median())

    width = 0.3
    plt.bar(np.arange(len(max_grads)), max_grads, width, color="c")
    plt.bar(np.arange(len(max_grads)) + width, ave_grads, width, color="b")
    plt.bar(np.arange(len(max_grads)) + 2 * width, median_grads, width, color='r')

    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="r", lw=4)], ['max-gradient', 'mean-gradient', 'median-gradient'])
    plt.show()


def paint_smi_matrixs(matrixs, index=0):
    """paint similarity matrix (TSM/ KQ) """
    plt.clf()
    b, c, w, h = matrixs.shape
    for i in range(c):
        matrix = matrixs[0, i, :, :].detach().cpu().numpy()
        plt.imshow(matrix)
        plt.colorbar()
        dir = 'graph/matrixs{0}'.format(index)
        if not os.path.exists(dir):
            os.mkdir('graph/matrixs{0}'.format(index))
        plt.savefig(fname="graph/matrixs{0}/matrix{1}.png".format(index, str(i)), dpi=400)
        plt.close()


def plot_inference(precount, count):
    # plot count result
    precount = precount.cpu()
    count = count.cpu()
    plt.plot(precount, color='blue')
    plt.plot(count, color='red')
    plt.savefig(fname="plot/inference.jpg", dpi=400)


def density_map(maps, count, index, file_name):
    # paint density map
    plt.clf()
    map = maps.detach().cpu().numpy().reshape(1, 64)
    sns.set()
    fig = plt.figure(figsize=(64, 4))
    sns_plot = sns.heatmap(map, xticklabels=False, cbar=False, cmap='Greens')
    plt.savefig(fname="density_map/{0}_{1}.png".format(file_name, index), dpi=500)
    plt.close()


# qiuyc增加的
def get_frame(vid_path):
    cap = cv2.VideoCapture(vid_path)
    # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    assert cap.isOpened()
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# qiuyc增加的
def get_vid_input(vid_path, num_sample):
    frames = get_frame(vid_path)
    total_frames_num = len(frames)
    frames_tensor = []
    if num_sample <= total_frames_num:
        for i in range(num_sample):
            #  select 64 frames from total original frames, proportionally
            frame = frames[i * total_frames_num // num_sample]  # 采样方法
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
    else:  # if raw frames number lower than 64, padding it. # 总帧数小于采样数
        for i in range(total_frames_num):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
        for i in range(num_sample - total_frames_num):
            frame = frames[total_frames_num - 1]  # 用最后一帧补充
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)

    Frame_Tensor = torch.as_tensor(np.stack(frames_tensor)).transpose(0, 1).unsqueeze(0)
    frames_tensor.clear()

    return Frame_Tensor


def get_frame_input(vid_path, num_sample):
    cap = cv2.VideoCapture(vid_path)
    total_frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    sample_id_list = []
    if num_sample <= total_frames_num:
        for i in range(num_sample):
            #  select 64 frames from total original frames, proportionally
            sample_id = i * total_frames_num // num_sample  # 采样方法
            sample_id_list.append(sample_id)
    else:  # if raw frames number lower than 64, padding it. # 总帧数小于采样数
        for i in range(total_frames_num):
            sample_id = i
            sample_id_list.append(sample_id)
        for i in range(num_sample - total_frames_num):
            sample_id = total_frames_num - 1  # 用最后一帧补充
            sample_id_list.append(sample_id)

    times = 0
    frames_tensor = []
    while True:
        # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
        success, frame = cap.read()
        if success:

            if times in sample_id_list:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
            times += 1
        else:
            break
    cap.release()
    Frame_Tensor = torch.as_tensor(np.stack(frames_tensor)).transpose(0, 1).unsqueeze(0)
    frames_tensor.clear()

    return Frame_Tensor


def get_labels_dict(path):
    labels_dict = {}
    # check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            labels_dict[row['name']] = cycle

    return labels_dict