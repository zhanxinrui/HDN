import cv2
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from hdn.core.config import cfg

def getPolarImg(img, original = None):
    """
    some assumption that img W==H
    :param img: image
    :return: polar image
    """
    sz = img.shape
    # maxRadius = math.hypot(sz[0] / 2, sz[1] / 2)
    maxRadius = sz[1]/2
    m = sz[1] / math.log(maxRadius)
    o = tuple(np.round(original)) if original is not None else (sz[0] // 2, sz[1] // 2)
    result = cv2.logPolar(img, o, m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR )
    #test
    # plt.imshow(result/255)
    # plt.show()
    # plt.imshow(img/255)
    # plt.show()
    # plt.close('all')
    return result

def getLinearPolarImg(img, original = None):
    """
    some assumption that img W==H
    :param img: image
    :return: polar image
    """
    sz = img.shape
    # maxRadius = math.hypot(sz[0] / 2, sz[1] / 2)
    maxRadius = sz[1]/2
    o = tuple(np.round(original)) if original is not None else (sz[0] // 2, sz[1] // 2)
    result = cv2.linearPolar(img, o, maxRadius, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR )
    # plt.imshow(result)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    return result



class STN_Polar(nn.Module):
    """
    STN head
    """
    def __init__(self, image_sz):
        super(STN_Polar, self).__init__()
        self._orignal_sz = [image_sz//2, image_sz//2]  # sample center position

    def _prepare_grid(self, sz, delta):
        assert len(sz) == 2  # W, H
        x_ls = torch.linspace(0, sz[0]-1, sz[0])
        y_ls = torch.linspace(0, sz[1]-1, sz[1])

        # get log polar coordinates
        mag = math.log(sz[0]/2) / sz[0]
        # rho = (torch.exp(mag * x_ls) - 1.0) + delta[0]
        rho = (torch.exp(mag * x_ls) - 1.0)
        theta = y_ls * 2.0 * math.pi / sz[1] + delta[1]# add rotation
        y, x = torch.meshgrid([theta, rho])
        cosy = torch.cos(y)
        siny = torch.sin(y)

        # construct final indices
        self.indices_x = torch.mul(x, cosy)
        self.indices_y = torch.mul(x, siny)

        # # test
        # y, x = torch.meshgrid([x_ls, y_ls])
        # self.indices_x = x.cuda()
        # self.indices_y = y.cuda()
    def _prepare_batch_grid(self, sz, delta, batch):
        assert len(sz) == 2  # W, H
        x_ls = torch.linspace(0, sz[0]-1, sz[0])
        y_ls = torch.linspace(0, sz[1]-1, sz[1])

        # get log polar coordinates
        mag = math.log(sz[0]/2) / sz[0]
        rho_batch = delta[0] + (torch.exp(mag * x_ls) - 1.0)
        theta_batch = delta[1] + y_ls * 2.0 * math.pi / sz[1]
        for rho, theta in rho_batch,theta_batch:
            y, x = torch.meshgrid([theta, rho])
            cosy = torch.cos(y)
            siny = torch.sin(y)

            # construct final indices
            self.indices_x = torch.mul(x, cosy)
            self.indices_y = torch.mul(x, siny)



    def get_logpolar_grid(self, polar, sz):
        """
        This implementation is based on OpenCV source code to match the transformation.
        :param polar: N*2 N pairs of original of coordinates [-1.0, 1.0]
        :param sz: 4 the size of the output
        :return: N*W*H*2 the grid we generated
        """
        assert len(sz) == 4 # N, C, W, H
        batch = sz[0]
        # generate grid mesh
        x = self.indices_x.cuda() # for multi-gpus
        y = self.indices_y.cuda()
        indices_x = x.repeat([batch, 1, 1]) + polar[:, 0].unsqueeze(1).unsqueeze(1)
        indices_y = y.repeat([batch, 1, 1]) + polar[:, 1].unsqueeze(1).unsqueeze(1)
        # print('indices_x.shape',indices_x.shape)
        # print('indices_y.shape',indices_y.shape)
        indices = torch.cat((indices_x.unsqueeze(3)/(sz[2]//2), indices_y.unsqueeze(3)/(sz[3]//2)), 3)

        return indices

    def forward(self, x, polar, delta=[0,0]):
        self._prepare_grid(self._orignal_sz, delta)
        grid = self.get_logpolar_grid(polar, x.size())#[1 127 127 2]
        # self.test_polar_points(grid.cpu().squeeze(0).view(-1,2))
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')

        # test plt log-polar img
        # x_lp_cpu = x[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        # plt.imshow(x_lp_cpu/256)
        # fig = plt.figure()
        # fig,ax = plt.subplots(1,dpi=96)
        # # ax.plot([polar[0], ], [polar[1], ], c='r', marker='x')
        # plt.show()
        # plt.close('all')
        return x, grid



class STN_LinearPolar(nn.Module):
    """
    STN head
    """
    def __init__(self, image_sz):
        super(STN_LinearPolar, self).__init__()
        self._orignal_sz = [image_sz//2, image_sz//2]  # sample center position
        self._prepare_grid(self._orignal_sz)

    def _prepare_grid(self, sz):
        assert len(sz) == 2  # W, H
        x_ls = torch.linspace(0, sz[0]-1, sz[0])
        y_ls = torch.linspace(0, sz[1]-1, sz[1])

        # get linear polar coordinates
        maxR =sz[0]/2
        rho = maxR * x_ls / sz[0]
        theta = y_ls * 2.0 * math.pi / sz[1]
        y, x = torch.meshgrid([theta, rho])
        cosy = torch.cos(y)
        siny = torch.sin(y)

        # construct final indices
        self.indices_x = torch.mul(x, cosy)
        self.indices_y = torch.mul(x, siny)

        # # test
        # y, x = torch.meshgrid([x_ls, y_ls])
        # self.indices_x = x.cuda()
        # self.indices_y = y.cuda()


    def get_logpolar_grid(self, polar, sz):
        """
        This implementation is based on OpenCV source code to match the transformation.
        :param polar: N*2 N pairs of original of coordinates [-1.0, 1.0]
        :param sz: 4 the size of the output
        :return: N*W*H*2 the grid we generated
        """
        assert len(sz) == 4 # N, C, W, H
        batch = sz[0]
        # generate grid mesh
        x = self.indices_x.cuda() # for multi-gpus
        y = self.indices_y.cuda()
        indices_x = x.repeat([batch, 1, 1]) + polar[:, 0].unsqueeze(1).unsqueeze(1)
        indices_y = y.repeat([batch, 1, 1]) + polar[:, 1].unsqueeze(1).unsqueeze(1)
        indices = torch.cat((indices_x.unsqueeze(3)/(sz[2]//2), indices_y.unsqueeze(3)/(sz[3]//2)), 3)

        return indices

    def forward(self, x, polar):
        grid = self.get_logpolar_grid(polar, x.size())
        # x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')

        return x


class Polar_Pick(nn.Module):
    """
    SiamFC head
    """
    def __init__(self):
        super(Polar_Pick, self).__init__()
        points = self.generate_points(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE)
        self.points = torch.from_numpy(points)
        self.points_cuda = self.points.cuda()



    def generate_points(self, stride, size):
        # print('stride',stride,'size',size)
        ori = - (size // 2) * stride # -96
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points


    def _getArgMax(self, r):
        sizes = r.size()
        batch = sizes[0]
        m = r.view(batch, -1).argmax(1).view(-1, 1)
        indices = torch.cat((m // sizes[2], m % sizes[2]), dim=1)
        indices = (indices - (sizes[2]-1)/2) / (sizes[2]-1)/2
        return indices

    def _getSoftArgMax(self, r):
        r = r.squeeze(1)
        sizes = r.size()
        assert len(sizes) == 3
        batch = sizes[0]
        sm = r.view(batch, -1).softmax(1).view(sizes)
        x_ls = torch.linspace(0, sizes[1] - 1, sizes[1])
        y_ls = torch.linspace(0, sizes[2] - 1, sizes[2])
        x, y = torch.meshgrid([x_ls, y_ls])
        indices_x = torch.mul(sm, x.unsqueeze(0).cuda()).sum([1, 2]) / (sizes[1] - 1)
        indices_y = torch.mul(sm, y.unsqueeze(0).cuda()).sum([1, 2]) / (sizes[2] - 1)
        indices = torch.cat((indices_x.view(-1, 1), indices_y.view(-1, 1)), 1)
        return indices

    def test_self_points(self):
        points = self.points
        points = points.permute(1,0)
        plt.scatter(points[0],points[1])

    #4 parameters loc
    def forward(self, cls, loc):
        # self.test_self_points()
        sizes = cls.size()
        batch = sizes[0]
        score = cls.view(batch, cfg.BAN.KWARGS.cls_out_channels, -1).permute(0, 2, 1)
        best_idx = torch.argmax(score[:, :, 1], 1)

        idx = best_idx.unsqueeze(1)
        idx = idx.unsqueeze(2)

        delta = loc.view(batch, 4, -1).permute(0, 2, 1)
        # delta = loc.view(batch, 6, -1).permute(0, 2, 1)
        # delta = loc.view(batch, 2, -1).permute(0, 2, 1)

        dummy = idx.expand(batch, 1, delta.size(2))
        point = self.points.cuda()
        point = point.expand(batch, point.size(0), point.size(1))

        delta = torch.gather(delta, 1, dummy).squeeze(1)
        point = torch.gather(point, 1, dummy[:,:,0:2]).squeeze(1)

        out = torch.zeros(batch, 2).cuda()
        out[:, 0] = (point[:, 0] - delta[:, 0] + point[:, 0] + delta[:, 2]) / 2
        out[:, 1] = (point[:, 1] - delta[:, 1] + point[:, 1] + delta[:, 3]) / 2
        return out



    def get_polar_from_two_para_loc (self, cls, loc):
        # self.test_self_points()
        sizes = cls.size()
        batch = sizes[0]
        score = cls.view(batch, cfg.BAN.KWARGS.cls_out_channels, -1).permute(0, 2, 1)
        best_idx = torch.argmax(score[:, :, 1], 1)

        idx = best_idx.unsqueeze(1)
        idx = idx.unsqueeze(2)
        #fixme use gt
        # delta = loc.view(batch, 4, -1).permute(0, 2, 1)
        # delta = loc.view(batch, 6, -1).permute(0, 2, 1)
        #fixme use  pred_loc
        delta = loc.view(batch, 2, -1).permute(0, 2, 1)

        dummy = idx.expand(batch, 1, delta.size(2))
        point = self.points.cuda()
        point = point.expand(batch, point.size(0), point.size(1))

        delta = torch.gather(delta, 1, dummy).squeeze(1)
        point = torch.gather(point, 1, dummy[:,:,0:2]).squeeze(1)

        out = torch.zeros(batch, 2).cuda()
        out[:, 0] = point[:, 0] - delta[:, 0]
        out[:, 1] = point[:, 1] - delta[:, 1]
        return out


    #shorten the time.
    def get_polar_from_two_para_loc (self, cls, loc):
        sizes = cls.size()
        batch = sizes[0]
        score = cls.view(batch, cfg.BAN.KWARGS.cls_out_channels, -1).permute(0, 2, 1)
        best_idx = torch.argmax(score[:, :, 1], 1)

        idx = best_idx.unsqueeze(1)
        idx = idx.unsqueeze(2)
        delta = loc.view(batch, 2, -1).permute(0, 2, 1)

        dummy = idx.expand(batch, 1, delta.size(2))
        point = self.points_cuda
        point = point.expand(batch, point.size(0), point.size(1))

        delta = torch.gather(delta, 1, dummy).squeeze(1)
        point = torch.gather(point, 1, dummy[:,:,0:2]).squeeze(1)

        out = torch.zeros(batch, 2).cuda()
        out[:, 0] = point[:, 0] - delta[:, 0]
        out[:, 1] = point[:, 1] - delta[:, 1]
        return out