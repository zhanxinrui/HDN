# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from hdn.core.config import cfg
from hdn.models.logpolar import getPolarImg, getLinearPolarImg
from hdn.utils.transform import img_shift_left_top_2_center
from hdn.utils.point import generate_points, generate_points_lp
class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):

    def _convert_bbox(self, delta, point):# delta:1*4*25*25
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()#4*625
        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta
    def _convert_delta(self, delta):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        return delta

    def _convert_c(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(2,-1)
        delta = delta.detach().cpu().numpy()
        delta[0, :] = point[:, 0] - delta[0, :]*8
        delta[1, :] = point[:, 1] - delta[1, :]*8
        return delta

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans, islog=False):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        # c = (original_sz + 1) / 2
        c = (original_sz - 1) / 2

        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        if len(im.shape)==2:
            im = im.reshape(im.shape[0], im.shape[1], 1)
            r, c, k = im.shape

            # k = 1
        else:
            r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        if islog:
            if islog == 1:
                im_log = getPolarImg(im_patch)
                # im_log = shift_left_top_2_center(im_log)
            else:
                im_log = getLinearPolarImg(im_patch)
                # im_log = shift_left_top_2_center(im_log)

            im_patch = np.concatenate((im_patch, im_log), 2)
        if len(im_patch.shape)==2:
            im_patch = im_patch.reshape(im_patch.shape[0],im_patch.shape[1],1)
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def get_subwindow_for_homo(self, im, pos, model_sz, original_sz, avg_chans, islog=False):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        # c = (original_sz + 1) / 2
        c = (original_sz - 1) / 2

        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        if len(im.shape) == 2:
            im = im.reshape(im.shape[0], im.shape[1], 1)
            r, c, k = im.shape

            # k = 1
        else:
            r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        if islog:
            if islog == 1:
                im_log = getPolarImg(im_patch)
                # im_log = shift_left_top_2_center(im_log)
            else:
                im_log = getLinearPolarImg(im_patch)
                # im_log = shift_left_top_2_center(im_log)

            im_patch = np.concatenate((im_patch, im_log), 2)
        if len(im_patch.shape) == 2:
            im_patch = im_patch.reshape(im_patch.shape[0], im_patch.shape[1], 1)
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch, (context_xmin, context_ymin, context_xmax+1, context_ymax+1)




    def get_subwindow_for_sift(self, im, pos, obj_sz):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        #we only need to mask the non-object field.



        if isinstance(pos, float):
            pos = [pos, pos]
        mask = np.zeros([im.shape[0], im.shape[1]])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('uint8')
        # im = np.mean(im, axis=2, keepdims=True)[:,:,0]

        mask[int(int(pos[1])-obj_sz[1]//2+1):int(int(pos[1])+obj_sz[1]//2+1),int(int(pos[0])-obj_sz[0]//2+1):int(int(pos[0])+obj_sz[0]//2+1), ] = 1
        im = (im * mask).astype('uint8')
        # im = np.expand_dims(im, axis=2)
        return im


    def get_subwindow_for_SuperGlue(self, im, pos, obj_sz):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        #we only need to mask the non-object field.



        if isinstance(pos, float):
            pos = [pos, pos]
        mask = np.zeros([im.shape[0], im.shape[1]])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('uint8')
        mask[int(int(pos[1])-obj_sz[1]//2+1):int(int(pos[1])+obj_sz[1]//2+1),int(int(pos[0])-obj_sz[0]//2+1):int(int(pos[0])+obj_sz[0]//2+1), ] = 1
        im = (im * mask).astype('uint8')
        return im


    def get_subwindow_for_lisrd(self, im, pos, obj_sz):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        mask = np.zeros([im.shape[0], im.shape[1],3])
        mask[int(int(pos[1])-obj_sz[1]//2+1):int(int(pos[1])+obj_sz[1]//2+1),int(int(pos[0])-obj_sz[0]//2+1):int(int(pos[0])+obj_sz[0]//2+1), ] = 1
        im = (im * mask).astype('uint8')
        return im
