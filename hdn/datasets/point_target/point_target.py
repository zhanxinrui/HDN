from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hdn.core.config import cfg
from hdn.utils.bbox import corner2center
from hdn.utils.point import Point
import math
import cv2
import matplotlib.pyplot as plt
import logging
class PointTarget:
    def __init__(self, ):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)#8 25, 255

    def __call__(self, img, target, size, angle=0.0, neg=False, init_wh=[]):
        # -1 ignore 0 negative 1 positive
        cls = 0 * np.ones((size, size), dtype=np.float32)# cls:  neg=-2, ignore=-1 pos=0~1
        img_mask = np.ones((cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE), dtype=np.int32)
        img_mask = np.argwhere(img_mask).reshape(cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE,2).transpose(2,0,1)
        self.img = img
        delta = np.zeros((4, size, size), dtype=np.float32)
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        def get_hamming_window(w, h, rot, sx, sy, out_size):
            alpha = 1
            w = math.floor(w * alpha)
            h = math.floor(h * alpha)
            ham_window = np.outer(np.hamming(h), np.hamming(w))
            if ham_window.shape[0] == 0 or ham_window.shape[1] == 0:
                logger = logging.getLogger('global')
                logger.info('tw {} th {} angle{} tcx{} tcy{} cfg.TRAIN.SEARCH_SIZE{}'.format(w, h, rot, sx, sy,
                                                                                         out_size))
            sx -= w/2
            sy -= h/2
            aa = w / 2
            bb = h / 2# rot center
            cc = math.cos(rot)
            ss = math.sin(rot)
            mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss + sx],
                                    [ss, cc, bb - bb * cc - aa * ss + sy],
                                    [0, 0, 1]]).astype(np.float)

            project = np.array([[1, 0, 0],
                                [0, 1, 0]]).astype(np.float)
            new_ham_window = cv2.warpAffine(ham_window, project @ mapping_rot, (out_size,out_size),
                                  borderMode=cv2.BORDER_REPLICATE)
            return new_ham_window
        tcx, tcy, tw, th = corner2center(target)
        tw, th = init_wh[0], init_wh[1]
        tw = 1 if tw < 1 else tw
        th = 1 if th < 1 else th

        hamming_window = get_hamming_window(tw, th, angle , tcx, tcy, cfg.TRAIN.SEARCH_SIZE)
        points = self.points.points

        if neg:
            neg = np.where(
                np.square((points[0] - tcx) * np.cos(angle) + (points[1] - tcy) * np.sin(angle)) / np.square(tw / 2) +
                np.square((points[0] - tcx) * np.sin(angle) - (points[1] - tcy) * np.cos(angle)) / np.square(th / 2) < 1)

            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0
            return img, cls, delta, hamming_window
        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]
        # ellipse label with angle
        pos = np.where(np.square((points[0] - tcx)*np.cos(angle) + (points[1] - tcy) * np.sin(angle) ) / np.square(tw / 2) +
                       np.square((points[0] - tcx)*np.sin(angle) - (points[1] - tcy) * np.cos(angle) ) / np.square(th / 2) < 1)
        neg = np.where(np.square((points[0] - tcx)*np.cos(angle) + (points[1] - tcy) * np.sin(angle) ) / np.square(tw / 2) +
                       np.square((points[0] - tcx)*np.sin(angle) - (points[1] - tcy) * np.cos(angle) ) / np.square(th / 2) > 1)

        weights = hamming_window[(np.array(points[1][pos]).astype(np.int32), \
                                  np.array(points[0][pos]).astype(np.int32))]#points[0][pos]get the point in 256*256 map,note the order of x,y
        cls[pos] = weights #actually theese pos are neg. except we wanna give them some weights
        cls[neg] = 0
        if cfg.DATASET.OCC > 0.0:
            gtz_pos = np.where(cls > 0.1)
            if len(gtz_pos) >= 1 and gtz_pos[0].shape[0] >=1:
                occ_c_idx = np.random.randint(0, gtz_pos[0].shape[0])
                occ_x = points[0][gtz_pos[1][occ_c_idx], gtz_pos[0][occ_c_idx]]  # points[0] 25*25
                occ_y = points[1][gtz_pos[1][occ_c_idx], gtz_pos[0][occ_c_idx]]
                radius_r = cfg.DATASET.OCC * np.random.random()
                occ_angle = 2 * math.pi * np.random.random()
                occ_p_map = np.where(np.square((points[0] - occ_x)*np.cos(occ_angle) + (points[1] - occ_y) * np.sin(occ_angle) ) / np.square(tw * radius_r / 2) +
                               np.square((points[0] - occ_x)*np.sin(occ_angle) - (points[1] - occ_y) * np.cos(occ_angle) ) / np.square(th * radius_r / 2) < 1)
                cls[occ_p_map] = 0
                occ_p_img = np.where(np.square((img_mask[0] - occ_x)*np.cos(occ_angle) + (img_mask[1] - occ_y) * np.sin(occ_angle) ) / np.square(tw * radius_r / 2) +
                               np.square((img_mask[0] - occ_x)*np.sin(occ_angle) - (img_mask[1] - occ_y) * np.cos(occ_angle) ) / np.square(th * radius_r / 2) < 1)

                img[occ_p_img] = 0
        # #######################################
        # #---------plot hamming window----------
        # fig, ax = plt.subplots(1,2, dpi=50)
        # ax[0].imshow(img/255)
        # ax[0].plot(tcx, tcy, 'rx', label="point")
        # ax[1].imshow(cls)
        # plt.show()
        # plt.close('all')
        # np.array()
        # max_c = np.max(cls)
        # pos_c = np.where(cls==max_c)
        # if max_c > 0.95:
        #     pos_c = np.where(cls==max_c)
        #     cls[pos_c] = -2
        # #cls[pos_c] = -2
        # #cls[pos] = 1
        # #cls[neg] = 0
        # #plot heatmap of ellipse
        # fig, ax = plt.subplots(2,2, dpi=300)
        # ax[0][0].imshow(cls)
        # ax[0][1].imshow(hamming_window)
        # plt.show()
        # plt.close('all')
        # #---------plot hamming window----------
        # #######################################
        return img, cls, delta, hamming_window



class PointTargetRot:
    # only rot eclipse
    def __init__(self, ):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, angle=0.0, neg=False, init_wh=[]):
        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((2, size, size), dtype=np.float32)
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num
        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points
        tw, th = init_wh[0], init_wh[1]

        if neg:
            neg = np.where(
                np.square((points[0] - tcx) * np.cos(angle) + (points[1] - tcy) * np.sin(angle)) / np.square(tw / 4) +
                np.square((points[0] - tcx) * np.sin(angle) - (points[1] - tcy) * np.cos(angle)) / np.square(
                    th / 4) < 1)

            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0

        delta[0] = (points[0] - tcx) / cfg.POINT.STRIDE
        delta[1] = (points[1] - tcy) / cfg.POINT.STRIDE

        # ellipse label with angle
        pos = np.where(
            np.square((points[0] - tcx) * np.cos(angle) + (points[1] - tcy) * np.sin(angle)) / np.square(tw / 4) +
            np.square((points[0] - tcx) * np.sin(angle) - (points[1] - tcy) * np.cos(angle)) / np.square(th / 4) < 1)
        neg = np.where(
            np.square((points[0] - tcx) * np.cos(angle) + (points[1] - tcy) * np.sin(angle)) / np.square(tw / 2) +
            np.square((points[0] - tcx) * np.sin(angle) - (points[1] - tcy) * np.cos(angle)) / np.square(th / 2) > 1)

        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        cls[pos] = 1
        cls[neg] = 0

        # print('cls.shape',cls.shape)
        return cls, delta

class PointTargetLP:
    def __init__(self, ):
        # self.points = self.generate_points(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE_LP)
        self.points = self.generate_points(cfg.POINT.STRIDE_LP, cfg.POINT.STRIDE_LP, cfg.TRAIN.OUTPUT_SIZE_LP)


    def generate_points(self, stride_w, stride_h, size):
        ori_x = - (size // 2) * stride_w  # -96
        ori_y = - (size // 2) * stride_h  # -96
        x, y = np.meshgrid([ori_x + stride_w * dx for dx in np.arange(0, size)],
                           [ori_y + stride_h * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)
        return points


    def get_offset(self, sim):
        mag = np.log(cfg.TRAIN.EXEMPLAR_SIZE  / 2) / cfg.TRAIN.EXEMPLAR_SIZE
        sx = np.log(sim.sx)/mag
        sy = np.log(sim.sy)/mag
        r = sim.rot/(2*np.pi/cfg.TRAIN.EXEMPLAR_SIZE)

        return sx, sy, r

    def __call__(self, sim, target, size, neg=False, init_wh=[]):
        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones(( size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        points = self.points
        sx, sy, r = self.get_offset(sim)
        tw, th = 6, 6 #(2,6) (4,4) (2,2) (10,10)

        if neg:
            neg = np.where(np.square(sx - points[0]) / np.square(tw*2.5) +
                           np.square(r - points[1]) / np.square(th*2.5) < 1)
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0

            return cls, delta

        delta[0] = (points[0] - sx) / cfg.POINT.STRIDE_LP
        delta[1] = (sx - points[0]) / cfg.POINT.STRIDE_LP
        delta[2] = (points[1] - r) / cfg.POINT.STRIDE_LP
        delta[3] = (r - points[1]) / cfg.POINT.STRIDE_LP
        # ellipse label

        pos = np.where(np.square(sx - points[0]) / np.square(tw) +
                       np.square(r - points[1]) / np.square(th) < 1)
        neg = np.where(np.square(sx - points[0]) / np.square(tw * 2.5) +
                       np.square(r - points[1]) / np.square(th * 2.5) > 1)

        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        cls[pos] = 1
        cls[neg] = 0

        # fig, ax = plt.subplots(2,2, dpi=300)
        # print('cls.shape',cls.shape)
        # ax[0][0].imshow(cls)
        # ax[0][1].imshow(delta[0])
        # ax[0][0].imshow(cls,cmap='gray')
        # plt.show()
        # plt.close('all')
        return cls, delta



if __name__ == '__main__':
    from hdn.utils.bbox import SimT, Corner
    target = Corner(50,50,72,122)
    fun = PointTarget()
    c,l = fun( target, 25, angle=np.pi/6, neg=False)#