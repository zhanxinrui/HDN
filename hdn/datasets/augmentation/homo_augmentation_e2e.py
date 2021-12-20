#Copyright 2021, XinruiZhan
"""
This class is designed to augment the non-homo-dataset
it's augmentation is composed by parameters not move the 4pts.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import math

from hdn.utils.bbox import corner2center, \
    Center, center2corner, Corner, SimT
from hdn.utils.transform import rot_scale_around_center_shift_tran
import matplotlib.pyplot as plt
import random
from hdn.utils.image_aug import addImage, add_spot_light
class Augmentation:
    def __init__(self, rho, shift, scale, blur, flip, color, rotation, distortion, affine_a, affine_c, affine_d, comp_alpha=1.0, comp_beta=0.0, comp_gamma=0.0):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rot = rotation
        self.rho = rho
        self.distortion = distortion
        self.affine_a = affine_a
        self.affine_c = affine_c
        self.affine_d = affine_d
        self.comp_a =  comp_alpha
        self.comp_b = comp_beta
        self.comp_g = comp_gamma

        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0



    def composeHomo(self, cx, cy, rot, scale, sx, sy, bbox, out_sz):
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -bbox[0]
        d = -bbox[1]
        mapping_shift = np.array([[1, 0, c],
                                  [0, 1, d],
                                  [0, 0, 1]]).astype(np.float)
        mapping_scale = np.array([[a, 0, 0],
                                  [0, b, 0],
                                  [0, 0, 1]]).astype(np.float)
        mapping_scale_img = mapping_scale@mapping_shift

        #move roi center to origin
        shift_origin = np.array([[1, 0, -cx],
                                 [0, 1, -cy],
                                 [0, 0, 1]]).astype(np.float)
        mapping_shift = np.array([[1, 0, sx],
                                  [0, 1, sy],
                                  [0, 0, 1]]).astype(np.float)

        """
        projective transformation.
        T * T_0^-1 * R * K * V * T_0 = 
        [ 1, 0, sx][ 1, 0, cx][ s*cos(theta), -s*sin(theta), 0][a, c,   0][1, 0,   0][ 1, 0, -cx]
        [ 0, 1, sy][ 0, 1, cy][ s*sin(theta),  s*cos(theta), 0][0, 1/a, 0][0, 1,   0][ 0, 1, -cy]
        [ 0, 0,  1][ 0, 0,  1][      0,          0 ,         1][0, 0,   1][v1, v2, u][ 0, 0,   1]
    
        """
        cc = math.cos(rot)
        ss = math.sin(rot)
        R = np.array([[cc, -ss, 0],
                      [ss,  cc, 0],
                      [ 0,   0, 1]
                      ]).astype(np.float)

        # since K is a upper triangle matrix, and det(K)==1, we can limit the Diagonal multiplication to be 1,
        """
        K = [a, c,   0]
            [0, 1/a, 0]
            [0, 0,   1]
        """

        if self.affine_a:
            a = (np.random.random()*2 -1) * self.affine_a + 1
            b = 1 / a
        else:
            a,b = 1,1
            # a,b,c = 0.8,1.2,0.4

        if self.affine_c:
            c = (np.random.random()*2 -1) * self.affine_c
            #c = np.random.random() * self.affine_c
        else:
            c = 0
        if self.affine_d:
            d = (np.random.random()*2 -1) * self.affine_d
            # d = np.random.random() * self.affine_d
        else:
            d = 0
        K = np.array([[a, c, 0],
                      [d, b, 0],
                      [0, 0, 1]]).astype(np.float)

        """
        V = [1, 0,   0]
            [0, 1,   0]
            [v1, v2, u]
        """

        if self.distortion:
            v1 = (np.random.random() * 2 - 1) * self.distortion
            v2 = (np.random.random() * 2 - 1) * self.distortion
        else:
            v1, v2 = 0, 0
        u = 1
        # v1, v2, u = 0, 0, 1 #default
        V = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [v1, v2, u]]).astype(np.float)  ## (v2<0,以中间为轴上端朝里。 v2>0, 以中间为轴下端朝里), （v1<0,以中间为轴左端朝里。 v1>0, 以中间为轴右端朝里)， 负是向上,最后的u是缩放因子
        # return R @ K @ V
        shift_origin_back = np.array([[1, 0, cx],
                                      [0, 1, cy],
                                      [0, 0, 1]]).astype(np.float)

        return mapping_shift @ shift_origin_back @ R @ K @ V @ shift_origin @ mapping_scale_img # we first aug with rot matrix then K and V


    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image
    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, 3*np.random.randn(3, 1))
        # offset = np.dot(self.rgbVar, 0.83)
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _proj_aug(self, image, bbox, poly, crop_bbox, size, theta=0,  if_comp=False, if_light=False, if_dark=False, img_add=None):
        '''
        This method is used to generate perspective transformation of an image,
        later we add some concrete simi(shift scale rot) to enhance the robustness.
        :param image: input image
        :param bbox:  the target bbox
        :param poly:  the target poly  (if we don't have poly label, default set same as bbox,)
        :param crop_bbox: the crop bbox 127*127 or 255*255
        :param size:  the final patch_size
        :param gray: gray image aug ratio
        :param same: to control the negative samples
        :param theta: rotation angle (rad)
        :param image_add: this image added on the original image
        :return:
        '''
        #note that this scale(sx,sy) means the obj_scale

        im_h, im_w = image.shape[:2]
        crop_bbox_center = corner2center(crop_bbox)
        scale_x = 1.0
        scale_y = 1.0
        # scale augmentation
        if self.scale:
            rand_scale = Augmentation.random() * (self.scale - 1)
            if rand_scale >= 0:
                scale_x = 1 + rand_scale  # * sx
            elif rand_scale < 0:
                scale_x = 1 + rand_scale / self.scale  # 1/k - x/k
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = scale_x
            if (w * scale_x) < 1.0:
                scale_x = 1 / w + 0.1
            if (h * scale_y) < 1.0:
                scale_y = 1 / h + 0.1
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)
        crop_bbox = center2corner(crop_bbox_center)
        # rotation augmentation
        rot = 0.0
        if self.rot:
            rot = Augmentation.random() * self.rot
            # self.rot = 0.3
            """
            crop image with similarity
            we first do the rotation before the original _crop_hwc
            [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
            [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
            [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
            """
        tot_rot = theta + rot
        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)  # crop_box的左上角为原点
        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)
        box_center = corner2center(bbox)
        init_w, init_h = box_center[2], box_center[3]
        if tot_rot != 0:
            """
            crop image with similarity
            we first do the rotation before the original _crop_hwc
            [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
            [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
            [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
            """
            aa = (bbox.x1 + bbox.x2) / 2
            bb = (bbox.y1 + bbox.y2) / 2
            cc = math.cos(tot_rot)
            ss = math.sin(tot_rot)
            mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss],
                                    [ss, cc, bb - bb * cc - aa * ss],
                                    [0, 0, 1]]).astype(np.float)
            p = np.array([[bbox.x1, bbox.x2, bbox.x1, bbox.x2],
                          [bbox.y1, bbox.y1, bbox.y2, bbox.y2],
                          [1, 1, 1, 1]]).astype(np.float)

            new_p = mapping_rot @ p
            max_p = np.max(new_p, 1)
            min_p = np.min(new_p, 1)

            rot_bbox = new_p
            # new bbox
            bbox = Corner(min_p[0], min_p[1],
                          max_p[0], max_p[1])

        sx = 0
        sy = 0
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift
            x1, y1, x2, y2 = crop_bbox
            if sx < 0:
                sx = max(sx, -x1)
            else:
                sx = min(im_w - 1 - x2, sx)
            if sy < 0:
                sy = max(sy, -y1)
            else:
                sy = min(im_h - 1 - y2, sy)
        homo_H = self.composeHomo(size//2, size//2, tot_rot, scale_x, sx, sy, crop_bbox, size)#cx,cy,rot,scale,sx,sy,
        image = cv2.warpPerspective(image, homo_H, (size, size),
                                    borderMode=cv2.BORDER_REPLICATE)
        if if_comp:
            comp_a = np.random.random()*self.comp_a
            comp_b = np.random.random()*self.comp_b
            comp_g = np.random.random()*self.comp_g
            image = addImage(image, img_add, comp_a, comp_b, comp_g)

        #mask the smooth & withe_noise
        if if_light:
            image = add_spot_light(image, max_brightness=128, min_brightness=0 )
        if if_dark:
            image = add_spot_light(image, max_brightness=255, min_brightness=128 )
        img_total_homo = homo_H
        bbox = Corner(bbox.x1 + sx, bbox.y1 + sy,
                      bbox.x2 + sx, bbox.y2 + sy)  # crop_box的左上角为原点
        new_poly = cv2.perspectiveTransform(np.expand_dims(poly, 0), img_total_homo)[0]
        tot_scale_x = (size / crop_bbox_center.w)
        tot_scale_y = (size / crop_bbox_center.h)
        return image, bbox, new_poly, SimT((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2, tot_scale_x, tot_scale_y,
                                           tot_rot), init_w, init_h



    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, poly, size, gray=False, theta=0, if_comp=False, if_light=False, if_dark=False, img_add=None):
        shape = image.shape #[511,511,3]
        crop_bbox = center2corner(Center(shape[0]/2, shape[1]/2,
                                         size, size))#127 or 255

        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox, poly, sim, init_w, init_h = self._proj_aug(image, bbox, poly, crop_bbox, size, theta=theta, if_comp=if_comp, if_light=if_light, if_dark=if_dark, img_add=img_add)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
            sim.x, sim.y, sim.rot = (bbox.x1+bbox.x2)/2, (bbox.y1+bbox.y2)/2, sim.rot
        return image, bbox, poly, sim, [init_w, init_h]
