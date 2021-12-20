#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import math
from hdn.tracker.hdn_tracker import hdnTracker

from hdn.core.config import cfg
from hdn.utils.bbox import corner2center, cetner2poly, getRotMatrix, transformPoly,center2corner
from hdn.utils.point import Point
from hdn.utils.transform import img_rot_around_center, img_rot_scale_around_center, img_shift, img_shift_crop_w_h, get_hamming_window, homo_add_shift, \
    rot_scale_around_center_shift_tran, img_proj_trans, shift_tran, find_homo_by_imgs_opencv_ransac, get_mask_window, decompose_affine,compose_affine_homo_RKS
from homo_estimator.Deep_homography.Oneline_DLTv1.tools.get_img_info import get_search_info, get_template_info, merge_tmp_search
import cv2
import matplotlib.pyplot as plt

class hdnTrackerHomo(hdnTracker):
    def __init__(self, model):
        super(hdnTrackerHomo, self).__init__(model)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.EXEMPLAR_SIZE // 2)
        self.points_lp = self.generate_points_lp(cfg.POINT.STRIDE_LP, cfg.POINT.STRIDE_LP, cfg.TRAIN.OUTPUT_SIZE_LP) #self.p.points.transpose((1, 2, 0)).reshape(-1, 2)
        self.model = model
        self.model.eval()

    def mask_img(self, img, points ):# cx, cy, w, h, rot
        mask = np.zeros([img.shape[0], img.shape[1]])
        contours = [points.astype(np.int32)]
        cv2.drawContours(mask, contours, 0, (1), -1)
        img[np.where(mask<=0)] = 0
        return img

    def homo_estimate(self, tmp, search, tmp_mask):
        merge_info = merge_tmp_search(tmp, search)
        org_imgs = torch.Tensor(merge_info['org_imgs']).float().unsqueeze(0).cuda()
        input_tensors = torch.Tensor(merge_info['input_tensors']).float().unsqueeze(0).cuda()
        patch_indices = torch.Tensor(merge_info['patch_indices']).float().unsqueeze(0).cuda()
        four_points = torch.Tensor(merge_info['four_points']).float().unsqueeze(0).cuda()  # ([1, 2, 360, 640])
        tmp_mask  = torch.from_numpy(tmp_mask).cuda()
        data = {}
        data['org_imgs'] = org_imgs
        data['input_tensors'] = input_tensors
        data['h4p'] = four_points
        data['patch_indices'] = patch_indices
        H, homo_score, simi_score = self.model.track_proj(data, tmp_mask)
        return H, homo_score, simi_score



    def init(self, img, bbox, poly, gt_points, first_point):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
            poly: (cx, cy, w, h, theta)
            first_point: (x1, y1) first point of gt
        """

        self.center_pos = np.array([poly[0],poly[1]])
        self.init_rot = poly[4]        # self.init_rot = 0
        self.rot = poly[4]            # self.rot = 0
        polygon = cetner2poly(poly[:4])
        tran = getRotMatrix(poly[0], poly[1], poly[4])
        polygon = transformPoly(polygon, tran)
        fir_dis = (polygon - first_point) ** 2
        fir_dis = np.argmin(fir_dis[:,0] + fir_dis[:,1])
        self.poly_shift_l = fir_dis
        self.scale = 1
        self.lp_shift = [0,0]
        self.v = 0
        self.size = np.array([poly[2], poly[3]])
        self.align_size = np.array([bbox[2], bbox[3]])
        # print('cotext',cfg.TRACK.CONTEXT_AMOUNT)
        # window range
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        w_z_sm = self.size[0] + 0 * np.sum(self.size)#crop size
        h_z_sm = self.size[1] + 0 * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_z_sm = np.sqrt(w_z_sm * h_z_sm)
        s_z = np.floor(s_z)
        s_z_sm = np.floor(s_z_sm)
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        self.z_crop, self.z_crop_points = self.get_subwindow_for_homo(img, self.center_pos,
                                                                      cfg.TRACK.EXEMPLAR_SIZE,
                                                                      s_z,
                                                                      self.channel_average, islog=1) # normal template
        self.z_crop_sm, self.z_crop_points_sm = self.get_subwindow_for_homo(img, self.center_pos,
                                                                            cfg.TRACK.EXEMPLAR_SIZE,
                                                                            s_z_sm,
                                                                            self.channel_average, islog=1) #for homo-estimation
        self.model.template(self.z_crop)
        self.init_img = img
        self.init_crop_size = np.array([w_z, h_z])
        self.init_size = self.size
        self.init_s_z = s_z
        self.init_s_z_sm = s_z_sm        # self.init_s_z_r = s_z_r
        self.init_pos = np.array([poly[0],poly[1]])
        self.window_scale_factor = 1.0
        self.lost = True
        self.lost_count = 0
        self.last_lost = False
        self.init_points = np.array(gt_points).astype(np.float32)
        self.init_homo_tmp, self.print_tmp_img = get_template_info(self.z_crop_sm[:, 0:3, :, :])
        self.H_total = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float32)
        self.H_total_sim = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float32)
        self.uncertain = 0
        self.recover_H = np.identity(3).astype('float')
    def update_template(self):
        img = self.init_img
        img = img_rot_around_center(img, self.init_pos[0], self.init_pos[1], img.shape[1], img.shape[0], self.lp_shift[1])
        self.z_crop = self.get_subwindow(img, self.init_pos,
                                         cfg.TRACK.EXEMPLAR_SIZE,
                                         self.init_s_z,self.channel_average, islog=1)
        self.model.template(self.z_crop)

    def update_template_window(self,sc):
        img = self.init_img
        z_crop = self.get_subwindow(img, self.init_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    self.init_s_z*sc ,self.channel_average, islog=1)
        self.model.template(z_crop)

    def get_points_by_homo(self, uni_points, H):
        pred_points = H @ uni_points
        pred_points = np.vsplit(pred_points,[2])[0]
        pred_points = pred_points.transpose([1,0])
        return pred_points
    def track_new(self, fr_idx, img, gt_box=None, gt_poly=None, gt_points=None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        ####################################################################
        ####---------------1.translation estimation---------------------####
        if np.linalg.det(self.H_total) == 0:
            self.H_total = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]]).astype(np.float32)  # we will do inv after, so make sure H_total is non-singular
        img = cv2.warpPerspective(img, np.linalg.inv(self.H_total), (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)
        init_points = self.init_points.reshape(-1, 2).astype(np.float32)
        s_z = self.init_s_z
        cur_sz = self.init_s_z
        center_pos = self.init_pos
        INS_EXAM_RATIO = np.round(cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        scale_z = cfg.TRACK.EXEMPLAR_SIZE  / s_z
        s_x = np.floor(s_z * INS_EXAM_RATIO) #window_scale_factor may influence the accuracy due to the floor in sub window
        x_crop = self.get_subwindow(img, center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x, self.channel_average)
        outputs = self.model.track_new(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_c = self._convert_c(outputs['loc_c'], self.points)
        pscore = score
        # TODO window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        stop_update_flag = 0
        if pscore[best_idx] < 0.05 :
            center = [0,0]
            stop_update_flag = 1
        else:
            center = pred_c[:, best_idx] / scale_z
        cx = center[0] + center_pos[0]
        cy = center[1] + center_pos[1]
        delta_cx = center[0]
        delta_cy = center[1]
        self.center_pos = np.array([cx, cy])
        ####---------------1.translation estimation---------------------####
        ####################################################################



        ####################################################################
        ####--------------------2.scale-rot estimation--------------------------####
        # lp_result
        x_crop_moved = self.get_subwindow(img, self.center_pos,
                                          cfg.TRACK.INSTANCE_SIZE,
                                          s_x, self.channel_average)
        #if update template
        outputs = self.model.track_new_lp(x_crop_moved, [0,0])
        score_lp = self._convert_score(outputs['cls_lp'])
        peak_map = score_lp.copy()
        peak_idx = np.argmax(peak_map)
        peak_map = np.zeros([13*13])
        peak_map[peak_idx] = 1
        pred_center_lp = self._convert_logpolar_simi(outputs['loc_lp'], self.points_lp, peak_idx, fr_idx)
        pscore_lp = score_lp
        best_idx_lp = np.argmax(pscore_lp)
        sim_lp = pred_center_lp[:, best_idx_lp]
        if stop_update_flag or pscore_lp[best_idx_lp] < 0.25:
            sim_lp = [1, 1, 0, 0]
        best_score = score[best_idx]
        scale_delta = sim_lp[0] * cur_sz / self.init_s_z #actually if there is no big delta, we can just sample the patch the same size as template.
        rot_delta = sim_lp[2]
        #similarity H
        H_sim = rot_scale_around_center_shift_tran(cx, cy, rot_delta, scale_delta, delta_cx, delta_cy)#(cx, cy, rot, scale, sx, sy)
        self.rot += rot_delta
        self.scale *= scale_delta
        ####--------------------2.lp estimation--------------------------####
        ######################################################################


        ######################################################################
        ####---------------3.residual estimation------------------------####
        rot_img_homo = img_rot_around_center(img, cx, cy, img.shape[1], img.shape[0], -rot_delta)
        x_crop_homo, crop_points = self.get_subwindow_for_homo(rot_img_homo, self.center_pos,
                                                               cfg.TRACK.EXEMPLAR_SIZE,
                                                               self.init_s_z_sm * scale_delta,#self.init_s_z_sm * scale_delta
                                                               self.channel_average)  # TODO reduce the size to 127*127

        crop_points_w = self.z_crop_points_sm[2] - self.z_crop_points_sm[0] + 1
        crop_points_h = self.z_crop_points_sm[3] - self.z_crop_points_sm[1] + 1
        resize_crop_w = 127
        resize_crop_h = 127


        #mask window
        sc = resize_crop_w/crop_points_w
        mask_tmp = get_mask_window(self.size[0]*sc, self.size[1]*sc, self.init_rot, \
                                   cfg.TRAIN.EXEMPLAR_SIZE/2, cfg.TRAIN.EXEMPLAR_SIZE/2, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE)#w, h, rot, sx, sy, out_size_w, out_size_h
        homo_search_img, print_search_img = get_search_info(x_crop_homo[:, 0:3, :, :])
        # #homo estimation
        H_hm_comp = np.identity(3)
        for i in range(1): # iterate
            H_hm, homo_score, simi_score = self.homo_estimate(self.init_homo_tmp, homo_search_img, mask_tmp)
            homo_score =  homo_score.detach().cpu().numpy()
            H_hm = H_hm.detach().cpu().squeeze(0).numpy()
            H_hm = np.linalg.inv(H_hm)
            H_hm = (1.0 / H_hm.item(8)) * H_hm
            homo_search_img = np.expand_dims(cv2.warpPerspective(homo_search_img[0], np.linalg.inv(H_hm), (127,127),
                                                                 borderMode=cv2.BORDER_REPLICATE), 0)
            H_hm_comp = H_hm_comp @ H_hm
        scale_H_1 = np.array([[resize_crop_w/crop_points_w , 0, 0],
                              [0, resize_crop_h/crop_points_h, 0],
                              [0, 0, 1]]).astype(np.float32)#recover the square to rect
        H_hm_comp = np.linalg.inv(scale_H_1) @ H_hm_comp @ scale_H_1
        shift_H = np.array([[1, 0, -self.z_crop_points_sm[0]],
                            [0, 1, -self.z_crop_points_sm[1]],
                            [0, 0, 1]]).astype(np.float32)#recover the square to rect
        H_hm_comp = np.linalg.inv(shift_H) @ H_hm_comp @ shift_H
        H_homo = H_hm_comp
        #Total Homography
        if  homo_score > 2.5:# in got 0.5x could be bad
            H = self.H_total @ H_sim
        else:
            H = self.H_total @ H_sim @ H_homo# Idon't know why it fails
        H = (1.0 / H.item(8)) * H
        self.H_total = H

        ####---------------3.residual estimation------------------------####
        ######################################################################

        #points
        pred_points = cv2.perspectiveTransform(np.expand_dims(init_points, 0), self.H_total)[0]
        max_p = np.max(pred_points, 0)
        min_p = np.min(pred_points, 0)
        align_bbox = [min_p[0], min_p[1],
                      max_p[0] - min_p[0], max_p[1] - min_p[1]]
        self.align_size = [align_bbox[2],align_bbox[3]] #real

        return {
            'bbox_aligned':align_bbox,
            'best_score': best_score,
            'polygon': pred_points,
            'points': pred_points,
            'bbox': align_bbox
        }

