from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

"""
cpu version
"""
#generate grid  for NM
def generate_points(stride, size):
    ori = - (size // 2) * stride # -96
    x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                       [ori + stride * dy for dy in np.arange(0, size)])
    points = np.zeros((size * size, 2), dtype=np.float32)
    points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    return points
#generate grid for LP
def generate_points_lp(stride_w, stride_h, size):
    ori_x = - (size // 2) * stride_w  # -96
    ori_y = - (size // 2) * stride_h  # -96
    x, y = np.meshgrid([ori_x + stride_w * dx for dx in np.arange(0, size)],
                       [ori_y + stride_h * dy for dy in np.arange(0, size)])
    points = np.zeros((size * size, 2), dtype=np.float32)
    points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
    return points



"""
gpu version 
"""
def lp_pick(cls, loc, CLS_OUT_CHANNELS, STRIDE, STRIDE_LP, OUTPUT_SIZE_LP, MAP_SIZE):
    sizes = cls.size()#[batch_size, 2, 13, 13]
    batch = sizes[0]
    score = cls.view(batch, CLS_OUT_CHANNELS, -1).permute(0, 2, 1)#[batch_size,13*13,2]
    best_idx = torch.argmax(score.softmax(2)[:, :, 1], 1)#batch_size
    idx = best_idx.unsqueeze(1).unsqueeze(2)
    delta = loc.view(batch, 4, -1).permute(0, 2, 1)#torch.Size([8, 4, 13, 13])-> torch.Size([8, 13*13, 4])
    dummy = idx.expand(batch, 1, delta.size(2)) ##torch.Size([batch_size, 1, 4])
    points = generate_points(STRIDE, OUTPUT_SIZE_LP)
    point = torch.from_numpy(points).cuda()
    point = point.expand(batch, point.size(0), point.size(1))#torch.Size([batch_size, 625, 2])
    delta = torch.gather(delta, 1, dummy).squeeze(1)
    point = torch.gather(point, 1, dummy[:, :, 0:2]).squeeze(1)
    scale = point[:, 0] - delta[:, 0] * STRIDE_LP
    rot = point[:, 1] - delta[:, 2] * STRIDE_LP
    rot = rot * (2 * np.pi / MAP_SIZE)
    mag = np.log(MAP_SIZE / 2) / MAP_SIZE
    scale = torch.exp(scale * mag)
    return scale, rot


def get_center(self, score_o, loc_c_o, label_c_o, points, REFINE_CLS_POS_THRESH):
    """
    :param score_o: score [28 25 25 2]
    :param loc_c_o: loc_c [28 2 25 25]
    :param label_c_o: label_c[28 2 25 25]
    :param points:
    :return:
    """
    score = score_o.clone().detach().permute(0,3,1,2)
    loc_c = loc_c_o.clone().detach()
    label_c = label_c_o.clone().detach()
    score = self._convert_score(score)#(28, 625)
    pred_c = self._convert_c(loc_c, points)#(28, 2, 625)
    label_c = self._convert_c(label_c,points) # print('score',score)
    best_idx = score.argmax(1) #28
    tmp_idx = range(0,best_idx.shape[0])
    cls_pos = score[tmp_idx, best_idx].gt(REFINE_CLS_POS_THRESH).nonzero().squeeze()
    pred_c = pred_c[tmp_idx, :, best_idx] #(28, 2)
    label_c = label_c[tmp_idx, :, best_idx]#(28, 2)
    best_idx = best_idx[cls_pos]
    pred_c = pred_c[cls_pos]
    label_c = label_c[cls_pos]
    return best_idx, pred_c, label_c, cls_pos



class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center
        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)

        return points


