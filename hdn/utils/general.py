# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio

import os
import numpy as np
import matplotlib.pyplot as plt
def geometricDistance(correspondence, h):
    """
    Correspondence err
    :param correspondence: Coordinate
    :param h: Homography
    :return: L2 distance
    """

    p1 = np.transpose(np.matrix([correspondence[0][0], correspondence[0][1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[1][0], correspondence[1][1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def create_gif(image_list, gif_name, duration=0.35):
    """
    create the gif
    :param image_list:
    :param gif_name:
    :param duration:
    :return:
    """
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return