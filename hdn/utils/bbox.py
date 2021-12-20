# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np
import math
import cv2


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
SimT = namedtuple('SimT', 'x y sx sy rot')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h

def cetner2poly(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2, x3, y3 ,x4, y4)
    Args:
        center: Center or np.array (4 * N)
    Return:
        polygon or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]

    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return np.array([x1, y1, x2, y1, x2, y2, x1,y2])

def getRotMatrix(cx,cy,rot):
    """
    crop image with similarity
    we first do the rotation before the original _crop_hwc
    [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
    [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
    [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    """
    #rbbox = [float(x) for x in crop_bbox]
    aa = cx #(rbbox[2] + rbbox[0]) / 2
    bb = cy #(rbbox[3] + rbbox[1]) / 2
    cc = np.cos(rot)
    ss = np.sin(rot)
    mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss],
                            [ss, cc, bb - bb * cc - aa * ss],
                            [0, 0, 1]]).astype(np.float)

    return mapping_rot

def transformPoly(polygon, m):
    """
    projective transform
    out = polygon @ m

    :param polygon:  polygon
    :param m:  homo matrix
    :return:
    """
    polygon = polygon.reshape(-1,2)
    out = np.ones([polygon.shape[0],3])
    out[:,0:2] = polygon
    out = out @ m.transpose(1,0)
    return out[:, 0:2]



def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


def cxy_wh_2_rect(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 0-index
    """
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])


def rect_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 0-index
    """
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), \
        np.array([rect[2], rect[3]])


def cxy_wh_2_rect1(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 1-index
    """
    return np.array([pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]])


def rect1_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 1-index
    """
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), \
        np.array([rect[2], rect[3]])


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1

    # if nv == 8:
    #     x1 = min(region[0::2])
    #     x2 = max(region[0::2])
    #     y1 = min(region[1::2])
    #     y2 = max(region[1::2])
    #     cx = (x1 + x2 ) / 2
    #     cy = (y1 + y2) / 2
    #     w = x2 - x1
    #     h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h



def get_min_max_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by mim-max box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h

def xywh2xyxy(region):
    x1, y1 =  region[0], region[1]
    x2, y2 = region[0]+region[2], region[1]+region[3]
    return x1, y1, x2, y2

def get_w_h_from_poly(region):
    """
    return the width and height according to the given polygon
    :param region:
    :return:
    """
    poly = np.array(region).reshape(-1, 2).astype(np.int32)
    rot_rect = cv2.minAreaRect(poly)
    rot_points = cv2.boxPoints(rot_rect)  # 4*2

    center_x = (rot_points[0][0] + rot_points[2][0]) / 2
    center_y = (rot_points[0][1] + rot_points[2][1]) / 2
    axis1_long = np.linalg.norm([rot_points[1][0] - rot_points[0][0], rot_points[1][1] - rot_points[0][1]])
    axis2_long = np.linalg.norm([rot_points[2][0] - rot_points[1][0], rot_points[2][1] - rot_points[1][1]])

    if (axis1_long > axis2_long):
        if abs(rot_points[1][0] - rot_points[0][0]) < 0.5:
            theta = math.pi / 2
        elif abs(rot_points[1][1] - rot_points[0][1]) < 0.5:
            theta = 0

        else:
            theta = math.atan((rot_points[1][1] - rot_points[0][1]) / (rot_points[1][0] - rot_points[0][0]))
        w = axis1_long
        h = axis2_long
    else:
        if abs(rot_points[2][0] - rot_points[1][0]) < 0.5:
            theta = math.pi / 2
        elif abs(rot_points[2][1] - rot_points[1][1]) < 0.5:
            theta = 0
        else:
            theta = math.atan((rot_points[2][1] - rot_points[1][1]) / (rot_points[2][0] - rot_points[1][0]))
        w = axis2_long
        h = axis1_long
    if theta > math.pi / 2:
        theta = theta - math.pi
    elif theta < -math.pi / 2:
        theta = math.pi - theta
    return (center_x, center_y, w, h, theta)


def get_points_from_xyxy(region):
    """ convert region to (x1,y1, x2,y2, x3,y3, x4,y4) that represent by 4 points
    """
    x1 = region[0]
    y1 = region[1]
    w = region[2]
    h = region[3]
    x2 = x1 + w
    y2 = y1
    x3 = x1
    y3 = y1 + h
    x4 = x1 + w
    y4 = y1 + h
    return (x1, y1, x2, y2, x3, y3, x4, y4)


def get_points_from_xywh(region):
    """ convert region to (x1,y1, x2,y2, x3,y3, x4,y4) that represent by 4 points
    """
    x1 = region[0]
    y1 = region[1]
    w = region[2]
    h = region[3]
    x2 = x1 + w
    y2 = y1
    x4 = x1
    y4 = y1 + h
    x3 = x1 + w
    y3 = y1 + h
    return (x1, y1, x2, y2, x3, y3, x4, y4)




def poly2mask(img_size, polygons):
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0],-1,2)
    cv2.fillPoly(mask, polygons,color=1)
    return mask