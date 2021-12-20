#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import math

from hdn.core.config import cfg
from hdn.tracker.base_tracker import SiameseTracker
from hdn.utils.bbox import corner2center, cetner2poly, getRotMatrix, transformPoly,center2corner
from hdn.utils.point import Point, generate_points, generate_points_lp
from hdn.utils.transform import img_rot_around_center, img_rot_scale_around_center, img_shift, img_shift_crop_w_h, get_hamming_window
import cv2
import matplotlib.pyplot as plt
class hdnTracker(SiameseTracker):
    def __init__(self, model):
        super(hdnTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = generate_points(cfg.POINT.STRIDE, self.score_size)
        self.p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.EXEMPLAR_SIZE // 2)
        self.points_lp = generate_points_lp(cfg.POINT.STRIDE_LP, cfg.POINT.STRIDE_LP, cfg.TRAIN.OUTPUT_SIZE_LP) #self.p.points.transpose((1, 2, 0)).reshape(-1, 2)
        self.model = model

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride # -96
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def generate_points_lp(self, stride_w, stride_h, size):
        # ori = - (size // 2) * stride  # -96
        ori_x = - (size // 2) * stride_w  # -96
        ori_y = - (size // 2) * stride_h  # -96
        x, y = np.meshgrid([ori_x + stride_w * dx for dx in np.arange(0, size)],
                           [ori_y + stride_h * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        return points

    def _convert_logpolar_simi(self, delta, point, peak_idx, idx=0):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        # rotation
        delta[2, :] = point[:, 1] - delta[2, :] * cfg.POINT.STRIDE_LP
        delta[3, :] = point[:, 1] + delta[3, :] * cfg.POINT.STRIDE_LP

        delta[0, :] = point[:, 0] - delta[0, :] * cfg.POINT.STRIDE_LP
        delta[1, :] = point[:, 0] + delta[1, :] * cfg.POINT.STRIDE_LP
        scale = delta[0, :]
        rotation = delta[2, :]
        rotation = rotation * (2 * np.pi / cfg.TRAIN.EXEMPLAR_SIZE)
        mag = np.log(cfg.TRAIN.EXEMPLAR_SIZE / 2) / cfg.TRAIN.EXEMPLAR_SIZE
        delta[0, :] = np.exp(scale * mag)
        delta[1, :] = delta[0, :]
        delta[2, :] = rotation
        return delta

    def _convert_logpolar_simi_in_lp(self, delta, point, peak_idx, idx=0):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        # rotation
        delta[2, :] = point[:, 1] - delta[2, :] * cfg.POINT.STRIDE_LP
        delta[3, :] = point[:, 1] + delta[3, :] * cfg.POINT.STRIDE_LP

        delta[0, :] = point[:, 0] - delta[0, :] * cfg.POINT.STRIDE_LP
        delta[1, :] = point[:, 0] + delta[1, :] * cfg.POINT.STRIDE_LP
        return delta



    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def mask_img(self, img, points ):# cx, cy, w, h, rot
        mask = np.zeros([img.shape[0], img.shape[1]])
        contours = [points.astype(np.int32)]
        cv2.drawContours(mask, contours, 0, (1), -1)
        img[np.where(mask<=0)] = 0
        return img
    def get_window_scale_coef(self,region):
        region = region.reshape(8,-1)
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        return s

    def init(self, img, bbox, poly, first_point):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
            poly: (cx, cy, w, h, theta)
            first_point: (x1, y1) first point of gt
        """
        # self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
        #                             bbox[1]+(bbox[3]-1)/2])
        self.center_pos = np.array([poly[0],poly[1]])
        self.init_rot = poly[4]        # self.init_rot = 0
        self.rot = poly[4]            # self.rot = 0
        polygon = cetner2poly(poly[:4])
        tran = getRotMatrix(poly[0], poly[1], poly[4])
        polygon = transformPoly(polygon, tran)
        self.scale_coeff = self.get_window_scale_coef(polygon)
        fir_dis = (polygon - first_point) ** 2
        fir_dis = np.argmin(fir_dis[:,0] + fir_dis[:,1])
        self.poly_shift_l = fir_dis
        self.scale = 1
        self.lp_shift = [0,0]
        self.v = 0
        self.size = np.array([poly[2], poly[3]])
        self.align_size = np.array([bbox[2], bbox[3]])
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_z = np.floor(s_z)
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average, islog=1)
        self.model.template(z_crop)
        self.init_img = img
        self.init_crop_size = np.array([w_z, h_z])
        self.init_size = self.size
        self.init_s_z = s_z
        self.init_pos = np.array([poly[0],poly[1]])
        self.window_scale_factor = 1.0
        self.lost = True
        self.lost_count = 0
        self.last_lost = False

    def update_template(self):
        img = self.init_img
        img = img_rot_around_center(img, self.init_pos[0], self.init_pos[1], img.shape[1], img.shape[0], self.lp_shift[1])
        z_crop = self.get_subwindow(img, self.init_pos,
                                     cfg.TRACK.EXEMPLAR_SIZE,
                                     self.init_s_z,self.channel_average, islog=1)
        self.model.template(z_crop)

    def update_template_window(self,sc):
        img = self.init_img
        z_crop = self.get_subwindow(img, self.init_pos,
                                     cfg.TRACK.EXEMPLAR_SIZE,
                                     self.init_s_z*sc ,self.channel_average, islog=1)
        self.model.template(z_crop)



    def track_new(self, fr_idx, img, gt_box, gt_poly):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        if w_z > img.shape[0]/2:
            self.window_scale_factor = 1
        self.window_scale_factor = 1
        s_z = np.floor(np.sqrt(w_z * h_z))
        INS_EXAM_RATIO = np.round(cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        scale_z = cfg.TRACK.EXEMPLAR_SIZE  / s_z
        s_x = np.floor(s_z * INS_EXAM_RATIO * self.window_scale_factor) #window_scale_factor may influence the accuracy due to the floor in sub window
        self.window_scale_factor = s_x / (s_z * INS_EXAM_RATIO)

        #without rot template(if update the template state)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x, self.channel_average)
        outputs = self.model.track_new(x_crop)  # add log-polar translation
        score = self._convert_score(outputs['cls'])
        pred_c = self._convert_c(outputs['loc_c'], self.points)

        #*--------------------plot search, cls_label & response map
        pscore = score
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)

        stop_update_flag = 0
        if pscore[best_idx] < 0.05 :
            center = [0,0]
            stop_update_flag = 1
        else:
            center = pred_c[:, best_idx] / scale_z * self.window_scale_factor

        new_window_scale_factor = 1
        if pscore[best_idx] < cfg.TRACK.SCALE_SCORE_THRESH:
            new_window_scale_factor = 1.5
            if self.lost_count ==0:
                self.last_lost = True
            self.lost_count+=1
            if not self.last_lost and self.lost_count < 5:
                self.lost_count = 0
                self.last_lost = False

        if fr_idx == 1:
            self.v = math.sqrt(center[0] * center[0] + center[1] * center[1])
        else:
            self.v = (self.v + math.sqrt(center[0] * center[0] + center[1] * center[1])) / 2

        cx = center[0] + self.center_pos[0]
        cy = center[1] + self.center_pos[1]

        # smooth bbox
        self.center_pos = np.array([cx, cy])

        # lp_result
        x_crop_moved = self.get_subwindow(img, self.center_pos,
                                          cfg.TRACK.INSTANCE_SIZE,
                                          s_x, self.channel_average)


        #if update template
        outputs = self.model.track_new_lp(x_crop_moved, [0,0])
        #if not update template
        score_lp = self._convert_score(outputs['cls_lp'])
        peak_map = score_lp.copy()
        peak_idx = np.argmax(peak_map)
        pred_center_lp = self._convert_logpolar_simi(outputs['loc_lp'], self.points_lp, peak_idx, fr_idx)

        #*----------------------------------plt lp map
        pscore_lp = score_lp
        best_idx_lp = np.argmax(score_lp)
        sim_lp = pred_center_lp[:, best_idx_lp]
        if stop_update_flag or pscore_lp[best_idx_lp] < 0.25:
            sim_lp = [1, 1, 0, 0]
        width = self.size[0] * sim_lp[0] * self.window_scale_factor
        height = self.size[1] * sim_lp[1] * self.window_scale_factor
        width = max(10*self.init_size[0]/self.init_size[1], min(width, img.shape[:2][1]))
        height = max(10, min(height, img.shape[:2][0]))
        # clip boundary
        self.size = np.array([width, height])

        self.lp_shift[1] += sim_lp[2]
        self.rot += sim_lp[2]
        self.scale = width / self.init_size[0]
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if self.rot >= 2 *math.pi :
            self.rot -= math.pi * 2
            self.lp_shift[1] -= math.pi * 2
        elif self.rot < -2*math.pi :
            self.rot += math.pi * 2
            self.lp_shift[1] += math.pi * 2

        best_score = score[best_idx]
        polygon = cetner2poly([cx, cy, width, height])
        tran = getRotMatrix(cx, cy, self.rot)
        polygon = transformPoly(polygon, tran)
        polygon = np.roll(polygon, 4 - self.poly_shift_l, 0)
        max_p = np.max(polygon, 0)
        min_p = np.min(polygon, 0)
        align_bbox = [min_p[0], min_p[1],
                     max_p[0] - min_p[0], max_p[1] - min_p[1]]
        self.align_size = [align_bbox[2],align_bbox[3]] #real
        bbox_align = align_bbox

        # update template state
        self.update_template()
        self.window_scale_factor = new_window_scale_factor


        return {
            'bbox': bbox,
            'bbox_aligned':bbox_align,
            'best_score': best_score,
            'rot': self.rot,
            'polygon': polygon,
        }

