from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import imageio
from hdn.core.config import cfg
from homo_estimator.Deep_homography.Oneline_DLTv1.backbone import get_backbone
from homo_estimator.Deep_homography.Oneline_DLTv1.preprocess import get_pre
import torch
from homo_estimator.Deep_homography.Oneline_DLTv1.utils import transform, DLT_solve
import matplotlib.pyplot as plt

"""
The model_builder we use right now.
"""

criterion_l2 = nn.MSELoss(reduce=True, size_average=True)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False, size_average=False)#anchor p, n
'''
   try to use huber loss to enhance the robustness
    >>> # Custom Distance Function
    >>> def l_infinity(x1, x2):
    >>>     return torch.max(torch.abs(x1 - x2), dim=1).values
    >>>
    >>> triplet_loss = \
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
'''
def triplet_loss_xr(anchor, positive, negative, p_mask, n_mask, mask_ap, margin=0.5):#a, p, n, p_mask, n_mask
    """

    :param anchor:
    :param positive:
    :param negative:
    :param a_mask: anchor weight window
    :param p_mask: positive weight window
    :param n_mask: negative weight window
    :param mask_ap: mask from the mask-generator
    :return: loss (scalar)
    """
    loss_neg = smooth_l1_loss(anchor, negative, p_mask, mask_ap)
    loss_pos = smooth_l1_loss(anchor, positive, n_mask, mask_ap)
    loss = loss_pos - loss_neg + margin
    return loss

def smooth_l1_loss(output, target, o_mask, mask_ap):
    absolute_loss = torch.abs(target - output) * o_mask * mask_ap
    square_loss = 0.5 * (target - output) ** 2 * o_mask * mask_ap
    inds = absolute_loss.lt(1).float()
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    tot = (mask_ap).sum()
    loss = reg_loss.sum() / tot
    return loss





def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def getPatchFromFullimg(patch_size_h, patch_size_w, patchIndices, batch_indices_tensor, img_full):
    num_batch, num_channels, height, width = img_full.size()
    warped_images_flat = img_full.reshape(-1)
    patch_indices_flat = patchIndices.reshape(-1)
    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    mask_patch = torch.gather(warped_images_flat, 0, pixel_indices)
    mask_patch = mask_patch.reshape([num_batch, 1, patch_size_h, patch_size_w])

    return mask_patch


def normMask(mask, strenth=0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    # print('max_value.shape',max_value.shape)
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask / (max_value * strenth)
    mask = torch.clamp(mask, 0, 1)

    return mask

class HomoModelBuilder(nn.Module):
    def __init__(self, pretrained = False):
        super(HomoModelBuilder, self).__init__()

        # build head
        self.ShareFeature = get_pre('PreShareFeature')
        model_name = cfg.BACKBONE_HOMO.TYPE
        print('pretrained:',pretrained)
        self.backbone = get_backbone(model_name,
                                     pretrained, **cfg.BACKBONE_HOMO.KWARGS)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        print('self.avgpool',self.avgpool)

        if model_name == 'resnet18' or model_name == 'resnet34':
            self.fc = nn.Linear(512, 8)
        elif model_name == 'resnet50':
            self.fc = nn.Linear(2048, 8)


    def forward(self, data):
        org_imgs = data['org_imgs']
        input_tensors = data['input_tensors']
        h4p = data['h4p']
        patch_inds = data['patch_indices']
        _device = 'cuda' if str(org_imgs.device)[:4] =='cuda' else 'cpu'
        # tmp_window = data['template_mask'] #[8,127,127]
        if 'search_windowx' in data: #acturally search_window
            sear_window = data['search_window'].squeeze(1) #[8,127,127]
        else:
            sear_window = torch.ones([input_tensors.shape[0], 127,127]).to(_device)
        if 'if_pos' in data:
            if_pos = data['if_pos']
        else:
            if_pos = torch.ones([input_tensors.shape[0],1, 127,127]).float().to(_device)
        if 'if_unsup' in data:
            if_unsup = data['if_unsup']
        else:
            if_unsup = torch.ones([input_tensors.shape[0],1, 127,127]).float().to(_device)
        batch_size, _, img_h, img_w = org_imgs.size()
        _, _, patch_size_h, patch_size_w = input_tensors.size()
        y_t = torch.arange(0, batch_size * img_w * img_h,
                           img_w * img_h)
        batch_inds_tensor = y_t.unsqueeze(1).expand(y_t.shape[0], patch_size_h * patch_size_w).reshape(-1)
        w_h_scala = torch.tensor(63.5)
        M_tensor = torch.tensor([[w_h_scala, 0., w_h_scala],
                                 [0., w_h_scala, w_h_scala],
                                 [0., 0., 1.]])
        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()
            batch_indices_tensor = batch_inds_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, M_tensor.shape[-2], M_tensor.shape[-1])
        # Inverse of M
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, M_tensor_inv.shape[-2],
                                                      M_tensor_inv.shape[-1])

        #original feature
        patch_1 = self.ShareFeature(input_tensors[:, :1, ...])
        patch_2 = self.ShareFeature(input_tensors[:, 1:, ...])

        #feature normed
        patch_1_res = patch_1
        patch_2_res = patch_2

        x = torch.cat((patch_1_res, patch_2_res), dim=1)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)#[bsz, 8]
        H_mat = DLT_solve(h4p, x).squeeze(1)  #H: search -> template
        # 'DLT_solve'
        pred_I2 = transform(patch_size_h, patch_size_w, M_tile_inv, H_mat, M_tile,
                            org_imgs[:, :1, ...], patch_inds, batch_indices_tensor)

        pred_I2_CnnFeature = self.ShareFeature(pred_I2)

        ## handle the negative samples loss
        neg_ids = if_pos.eq(0).nonzero().squeeze(1)
        #only unsupervised homo
        pos_ids = (if_pos*if_unsup).eq(1).nonzero().squeeze(1)
        #add center mask
        mask_sear = sear_window.gt(0).unsqueeze(1).float()
        #do not use mask at all
        patch_1_m = patch_1
        patch_2_m = patch_2
        pred_I2_CnnFeature_m = pred_I2_CnnFeature
        ## use neg samples, it seems loss doesn't descend
        if neg_ids.shape[0] != 0:
            tmp_pos = patch_1_m[pos_ids]
            sear_pos = patch_2_m[pos_ids]
            pred_pos = pred_I2_CnnFeature_m[pos_ids]
            #only use the pos samples
            tmp_replace = tmp_pos
            sear_replace = sear_pos
            pred_replace = pred_pos
        else:
            mask_num = mask_sear.nonzero().shape[0]
            tmp_replace = patch_1_m
            sear_replace = patch_2_m
            pred_replace = pred_I2_CnnFeature_m
        feature_loss_mat = triplet_loss(sear_replace, pred_replace, tmp_replace)

        feature_loss = torch.sum(feature_loss_mat) / pos_ids.shape[0] /(127*127)
        feature_loss = torch.unsqueeze(feature_loss, 0)
        #neg loss
        cur_device = feature_loss.device
        homo_neg_loss = torch.tensor(0.0).to(cur_device)
        if neg_ids.shape[0] > 0:
            homo_neg_loss = torch.sum(torch.norm(x[neg_ids,:], p=2, dim=1)) / neg_ids.shape[0]


        pred_I2_d = pred_I2[:1, ...]
        patch_2_res_d = patch_2_res[:1, ...]
        pred_I2_CnnFeature_d = pred_I2_CnnFeature[:1, ...]

        out_dict = {}

        out_dict.update(feature_loss=feature_loss, pred_I2_d=pred_I2_d, x=x, H_mat=H_mat, patch_2_res_d=patch_2_res_d,
                        pred_I2_CnnFeature_d=pred_I2_CnnFeature_d, homo_neg_loss=homo_neg_loss)

        return out_dict
