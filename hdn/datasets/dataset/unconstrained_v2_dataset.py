#Copyright 2021, XinruiZhan
"""
 this file implements the perspective transforma augmentation on template image as search, or just use sampled two images from video as template and search.
we just need to adjust the interval, if there is interval we use unsupervised, if not, then use supervised
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torchvision.transforms as transforms
from hdn.datasets.custom_transforms import Normalize, ToTensor
import time
import json
import logging

import sys
import os
import math

import cv2
import numpy as np
from torch.utils.data import Dataset
from hdn.utils.transform import img_shift_crop_w_h
from hdn.utils.bbox import center2corner, Center, corner2center, SimT
from hdn.datasets.point_target.point_target import PointTarget, PointTargetLP, PointTargetRot
from hdn.datasets.augmentation.homo_augmentation_e2e import Augmentation
from hdn.core.config import cfg
from hdn.models.logpolar import getPolarImg
import matplotlib.pyplot as plt
logger = logging.getLogger("global")
from hdn.datasets.dataset.dataset import SubDataset
# setting opencv
from memory_profiler import profile

pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)
class BANDataset(Dataset):
    # @profile
    def __init__(self,):
        super(BANDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.Grayscale(1),
            # Normalize(),
            # ToTensor()
        ])

        self.maxscale = 0.0
        self.totalscale = 0.0
        self.curScale=0

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
                       cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()
        self.point_target_lp = PointTargetLP()
        self.point_target_c = PointTargetRot()
        # create sub dataset
        self.all_dataset = []
        start = 0
        # start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                # unsup_start,
                start,
                subdata_cfg.IF_UNSUP
            )
            start += sub_dataset.num
            self.num += sub_dataset.num_use
            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_sup_aug = Augmentation(# rho, shift, scale, blur, flip, color, rotation
            cfg.DATASET.TEMPLATE.RHO,
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR,
            cfg.DATASET.TEMPLATE.ROTATION,
            cfg.DATASET.TEMPLATE.DISTORTION,
            cfg.DATASET.TEMPLATE.AFFINE_A,
            cfg.DATASET.TEMPLATE.AFFINE_C,
            cfg.DATASET.TEMPLATE.AFFINE_D
        )
        self.search_sup_aug = Augmentation(
            cfg.DATASET.SEARCH.RHO,
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR,
            cfg.DATASET.SEARCH.ROTATION,
            cfg.DATASET.SEARCH.DISTORTION,
            cfg.DATASET.SEARCH.AFFINE_A,
            cfg.DATASET.SEARCH.AFFINE_C,
            cfg.DATASET.SEARCH.AFFINE_D,
            cfg.DATASET.SEARCH.IMG_COMP_ALPHA,
            cfg.DATASET.SEARCH.IMG_COMP_BETA,
            cfg.DATASET.SEARCH.IMG_COMP_GAMMA,
        )

        self.template_unsup_aug = Augmentation(# rho, shift, scale, blur, flip, color, rotation
            cfg.DATASET.TEMPLATE.UNSUPERVISED.RHO,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.SHIFT,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.SCALE,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.BLUR,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.FLIP,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.COLOR,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.ROTATION,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.DISTORTION,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_A,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_C,
            cfg.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_D,

        )
        self.search_unsup_aug = Augmentation(
            cfg.DATASET.SEARCH.UNSUPERVISED.RHO,
            cfg.DATASET.SEARCH.UNSUPERVISED.SHIFT,
            cfg.DATASET.SEARCH.UNSUPERVISED.SCALE,
            cfg.DATASET.SEARCH.UNSUPERVISED.BLUR,
            cfg.DATASET.SEARCH.UNSUPERVISED.FLIP,
            cfg.DATASET.SEARCH.UNSUPERVISED.COLOR,
            cfg.DATASET.SEARCH.UNSUPERVISED.ROTATION,
            cfg.DATASET.SEARCH.UNSUPERVISED.DISTORTION,
            cfg.DATASET.SEARCH.UNSUPERVISED.AFFINE_A,
            cfg.DATASET.SEARCH.UNSUPERVISED.AFFINE_C,
            cfg.DATASET.SEARCH.UNSUPERVISED.AFFINE_D,
            cfg.DATASET.SEARCH.IMG_COMP_ALPHA,
            cfg.DATASET.SEARCH.IMG_COMP_BETA,
            cfg.DATASET.SEARCH.IMG_COMP_GAMMA,
        )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH * cfg.TRAIN.EPOCH  #30000
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            last_name = ""
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            #fixme random
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]
    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx
    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox, scale_z

    def _get_trans_of_poly_from_ori_to_patch(self, poly, patch_size, scale):
        poly_mat = np.array(poly).reshape(4, 2)
        # we use outer bounding box here
        max_p = np.max(poly_mat, 0)
        min_p = np.min(poly_mat, 0)
        aligned_bbox_in_ori = [int(min_p[0]), int(min_p[1]), int(max_p[0]),
                               int(max_p[1]), ]  # (xmin, ymin, xmax, ymax)
        center_in_ori = [(aligned_bbox_in_ori[0] + aligned_bbox_in_ori[2])/2 , (aligned_bbox_in_ori[1] + aligned_bbox_in_ori[3])/2]
        center_in_patch = [patch_size/2, patch_size/2]
        shift_trans_before = np.array([[1, 0, -center_in_ori[0]],
                                       [0, 1, -center_in_ori[1]],
                                       [0, 0, 1]]).astype(np.float32)
        scale_trans = np.array([[scale, 0, 0],
                                [0, scale, 0],
                                [0, 0, 1]]).astype(np.float32)
        shift_trans_after = np.array([[1, 0, center_in_patch[0]],
                                      [0, 1, center_in_patch[1]],
                                      [0, 0, 1]]).astype(np.float32)
        trans = shift_trans_after@scale_trans@shift_trans_before
        poly_new = cv2.perspectiveTransform(np.expand_dims(poly_mat, 0).astype('float32'), trans)[0]

        return trans, poly_new
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        while True:
            index = self.pick[index]
            dataset, index = self._find_dataset(index)
            if_unsup = dataset.if_unsup
            gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
            neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
            if_comp = False
            if_light = False
            if_dark = False
            img_add = None
            if cfg.DATASET.COMP > np.random.random():
                if_comp = True
                img_add = cv2.imread(np.random.choice(self.all_dataset).get_random_target()[0])
            if cfg.DATASET.LIGHT > np.random.random():
                if_light = True
            if cfg.DATASET.DARK > np.random.random():
                if_dark = True
            if neg:

                template = dataset.get_random_target(index)# #(x_min, y_min, x_max, y_max)#[w, h, theta, center_x, center_y]
                search = np.random.choice(self.all_dataset).get_random_target()
            else:
                template, search = dataset.get_positive_pair(index)
            # get image
            template_image = cv2.imread(template[0])#511*511*3
            search_image = cv2.imread(search[0])#511*511*3
            template_box, tmp_scale_z = self._get_bbox(template_image, template[1][0])
            search_box, sea_scale_z = self._get_bbox(search_image, search[1][0])#aligned bbox and scale_z

            #poly points
            tmp_ori_bbox = template[1][0] # upright
            sear_ori_bbox = search[1][0]

            template_poly = [tmp_ori_bbox[0], tmp_ori_bbox[1], tmp_ori_bbox[2], tmp_ori_bbox[1], \
                             tmp_ori_bbox[2], tmp_ori_bbox[3], tmp_ori_bbox[0], tmp_ori_bbox[3]]
            search_poly = [sear_ori_bbox[0], sear_ori_bbox[1], sear_ori_bbox[2], sear_ori_bbox[1], \
                           sear_ori_bbox[2], sear_ori_bbox[3], sear_ori_bbox[0], sear_ori_bbox[3]]
            # get tranformation the polygon representation of target  from ori to patch
            temp_points_trans, temp_poly = self._get_trans_of_poly_from_ori_to_patch(template_poly, template_image.shape[0], tmp_scale_z)
            search_points_trans, sear_poly = self._get_trans_of_poly_from_ori_to_patch(search_poly, search_image.shape[0], sea_scale_z)
            tmp1 = corner2center(template_box)
            tmp2 = corner2center(search_box)
            tmp_theta =  template[1][1][2]
            search_theta =  search[1][1][2]
            sx = tmp2.w / tmp1.w
            sy = tmp2.h / tmp1.h


            # augmentation
            if if_unsup:
                template, box_lp, temp_poly, sim_tmp, init_wh_tmp = self.template_unsup_aug(template_image,
                                                                                            template_box,
                                                                                            temp_poly,
                                                                                            cfg.TRAIN.EXEMPLAR_SIZE,
                                                                                            gray=gray,
                                                                                            theta=tmp_theta)
            else:
                template, box_lp, temp_poly, sim_tmp, init_wh_tmp = self.template_sup_aug(template_image,
                                                                                          template_box,
                                                                                          temp_poly,
                                                                                          cfg.TRAIN.EXEMPLAR_SIZE,
                                                                                          gray=gray,
                                                                                          theta=tmp_theta)
            tmp_box = corner2center(box_lp)
            template_lp = getPolarImg(template, (tmp_box.x, tmp_box.y))
            if if_unsup:
                search, bbox, sear_poly, sim, init_wh = self.search_unsup_aug(search_image,
                                                                              search_box,
                                                                              sear_poly,
                                                                              cfg.TRAIN.SEARCH_SIZE,
                                                                              # cfg.TRAIN.EXEMPLAR_SIZE,
                                                                              gray=gray,
                                                                              theta=search_theta,
                                                                              if_comp=if_comp,
                                                                              if_light=if_light,
                                                                              if_dark=if_dark,
                                                                              img_add=img_add)#del origin rot casuse we want rot =0 in template
            else:
                search, bbox, sear_poly, sim, init_wh = self.search_sup_aug(search_image,
                                                                            search_box,
                                                                            sear_poly,
                                                                            cfg.TRAIN.SEARCH_SIZE,
                                                                            # cfg.TRAIN.EXEMPLAR_SIZE,
                                                                            gray=gray,
                                                                            theta=search_theta,
                                                                            if_comp=if_comp,
                                                                            if_light=if_light,
                                                                            if_dark=if_dark,
                                                                            img_add=img_add)#del origin rot casuse we want rot =0 in template





            ##to mask the border(black edge) which is not related to the obj,
            if neg:
                mask_tmp = np.zeros([cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE])
                mask_search = np.zeros([cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE])
                if_pos = 0
            else:
                if_pos = 1
                mask_tmp = np.ones([cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE])
                mask_search = np.ones([cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE])

            search_rot = sim.rot
            delta_theta = sim.rot - sim_tmp.rot
            delta_sx = sim.sx / sim_tmp.sx
            if  delta_theta > (math.pi * 2):
                delta_theta -= math.pi * 2
            elif delta_theta <= (-math.pi * 2):
                delta_theta += math.pi * 2

            sim = SimT(sim.x, sim.y, delta_sx, delta_sx, delta_theta)


            # checking point
            shape = search_image.shape
            crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                             cfg.TRAIN.SEARCH_SIZE-1, cfg.TRAIN.SEARCH_SIZE-1))
            crop_bbox_center = corner2center(crop_bbox)
            h, w = crop_bbox_center.h, crop_bbox_center.w

            if sx > float(shape[0]) / w or sy > float(shape[1]) / h:
                continue

            self.totalscale = self.totalscale + sim.sx
            self.curScale = self.curScale + 1
            if sim.sx > self.maxscale:
                self.maxscale = sim.sx

            #fixme for homo-estimator test
            if cfg.TRAIN.MODEL_TYPE == 'SEP':
                search = img_shift_crop_w_h(search, 0, 0, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE)

            #fixme for e2e
            elif cfg.TRAIN.MODEL_TYPE == 'E2E':
                search_127 = img_shift_crop_w_h(search, 0, 0, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE)
                search, cls, delta, window_map = self.point_target(search, bbox, cfg.TRAIN.OUTPUT_SIZE, search_rot, neg, init_wh)



            #for homo-estimator training
            if cfg.TRAIN.MODEL_TYPE == 'SEP':
                _mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
                _std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
                template = (template - _mean_I) / _std_I
                search = (search - _mean_I) / _std_I
                template = template.transpose((2, 0, 1)).astype(np.float32)
                search = search.transpose((2, 0, 1)).astype(np.float32)
                return {
                    'template': template,
                    'template_lp': template_lp,
                    'search': search,
                    'template_poly': temp_poly,
                    'search_poly': sear_poly,

                }
            #for e2e
            elif cfg.TRAIN.MODEL_TYPE == 'E2E':
                cls_c, delta_c = self.point_target_c(bbox, cfg.TRAIN.OUTPUT_SIZE, search_rot, neg, init_wh)
                cls_lp, delta_lp = self.point_target_lp(sim, bbox, cfg.TRAIN.OUTPUT_SIZE_LP, neg,  init_wh)

                _mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
                _std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
                template_hm = (template - _mean_I) / _std_I
                search_hm = (search - _mean_I) / _std_I#255*255
                template = template.transpose((2, 0, 1)).astype(np.float32)
                search = search.transpose((2, 0, 1)).astype(np.float32)
                template_lp = template_lp.transpose((2, 0, 1)).astype(np.float32)
                template_hm = template_hm.transpose((2, 0, 1)).astype(np.float32)
                search_hm = search_hm.transpose((2, 0, 1)).astype(np.float32)
                return {
                    'template': template,
                    'template_lp': template_lp,
                    'search': search,
                    'template_poly': temp_poly,
                    'search_poly': sear_poly,
                    'label_cls': cls,
                    'label_loc': delta,
                    'label_cls_lp': cls_lp,
                    'label_loc_lp': delta_lp,
                    'scale_dist': sim.sx,
                    'label_cls_c': cls_c,
                    'label_loc_c': delta_c,
                    'window_map': window_map,
                    'template_hm':template_hm,
                    'search_hm': search_hm,
                    'template_window': mask_tmp,
                    'search_window': mask_search,
                    'if_pos': if_pos,
                    'temp_cx': sim_tmp.x,
                    'temp_cy': sim_tmp.y,
                    'if_unsup': if_unsup
                }