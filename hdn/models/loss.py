#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hdn.core.config import cfg
from hdn.models.iou_loss import linear_iou
from torch.autograd import Variable


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)



def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_xr_focal_fuse_smooth_l1_loss_top_k(pred_cls, label_cls,delta_weight=0.1):
    """
    smooth_l1_loss, we only choose the top_K neg_loss as neg_loss cause too many neg points pull the loss down.
    :param pred_cls:
    :param label_cls:
    :param delta_weight:
    :return:
    """
    batch_size = label_cls.shape[0]
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.clone()
    pred_cls = pred_cls.view(-1,2)
    neg = label_cls.data.eq(0).nonzero().squeeze().cuda()
    pos = label_cls.data.gt(0).nonzero().squeeze().cuda()
    cur_device = pred_cls.device
    zero_loss = torch.tensor(0.0).to(cur_device)

    if len(pos.size()) == 0 or \
            pos.size() == torch.Size([0]):
        pos_loss = zero_loss
    else:
        pred_cls_pos = torch.index_select(pred_cls, 0, pos)[:, 1]
        absolute_loss_pos = torch.abs(label_cls_new[pos] - pred_cls_pos)
        reg_loss_pos = absolute_loss_pos# use l1 loss
        pos_loss = reg_loss_pos.sum()/ (reg_loss_pos.shape[0]+1)

    if len(neg.size()) == 0 or \
            neg.size() == torch.Size([0]):
        neg_loss = zero_loss  #problem here
    else:
        pred_cls_neg = torch.index_select(pred_cls, 0, neg)[:, 1]
        pred_cls_neg = pred_cls_neg.clamp(min=0.000001, max=0.9999999)
        reg_loss_neg = - torch.log(1 - pred_cls_neg)
        reg_loss_neg = torch.topk(reg_loss_neg, batch_size*100).values
        neg_loss = reg_loss_neg.sum() / (reg_loss_neg.shape[0]+1)
    reg_loss = pos_loss + neg_loss

    return reg_loss

def select_xr_focal_fuse_smooth_l1_loss(pred_cls, label_cls,delta_weight=0.1):
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.clone()
    pred_cls = pred_cls.view(-1,2)
    neg = label_cls.data.eq(0).nonzero().squeeze().cuda()
    pos = label_cls.data.gt(0).nonzero().squeeze().cuda()
    pos_loss = 0
    neg_loss = 0
    if len(pos.size()) == 0 or \
            pos.size() == torch.Size([0]):
        reg_loss_pos = 0
        neg_loss = 0
    else:
        pred_cls_pos = torch.index_select(pred_cls, 0, pos)[:, 1]
        absolute_loss_pos = torch.abs(label_cls_new[pos] - pred_cls_pos)
        square_loss_pos =  0.5 * ((label_cls_new[pos] - pred_cls_pos)) ** 2
        inds_pos = absolute_loss_pos.le(1).float()
        reg_loss_pos = ( inds_pos * square_loss_pos + (1 - inds_pos) * (absolute_loss_pos - 0.5))
        pos_loss = reg_loss_pos.sum()/ (reg_loss_pos.shape[0]+1)

    if len(neg.size()) == 0 or \
            neg.size() == torch.Size([0]):
        reg_loss_neg = 0  #problem here
        pos_loss = 0
    else:
        pred_cls_neg = torch.index_select(pred_cls, 0, neg)[:, 1]
        pred_cls_neg = pred_cls_neg.clamp(min=0.000001, max=0.9999999)
        absolute_loss_neg = torch.abs(label_cls_new[neg] - pred_cls_neg)
        reg_loss_neg = -0.5*absolute_loss_neg * torch.log(1 - pred_cls_neg)
        neg_loss = reg_loss_neg.sum() / (reg_loss_neg.shape[0]+1)
    reg_loss = pos_loss + neg_loss
    return reg_loss



def select_xr_focal_fuse_smooth_l1_loss(pred_cls, label_cls,delta_weight=0.1):
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.clone()
    pred_cls = pred_cls.view(-1,2)
    neg = label_cls.data.eq(0).nonzero().squeeze().cuda()
    pos = label_cls.data.gt(0).nonzero().squeeze().cuda()
    cur_device = pred_cls.device
    zero_loss = torch.tensor(0.0).to(cur_device)

    pos_loss = zero_loss
    neg_loss = zero_loss
    if len(pos.size()) == 0 or \
            pos.size() == torch.Size([0]):
        pos_loss = zero_loss

    else:
        pred_cls_pos = torch.index_select(pred_cls, 0, pos)[:, 1]
        absolute_loss_pos = torch.abs(label_cls_new[pos] - pred_cls_pos)
        square_loss_pos =  0.5 * ((label_cls_new[pos] - pred_cls_pos)) ** 2
        inds_pos = absolute_loss_pos.le(1).float()
        reg_loss_pos = ( inds_pos * square_loss_pos + (1 - inds_pos) * (absolute_loss_pos - 0.5))
        pos_loss = reg_loss_pos.sum()/ (reg_loss_pos.shape[0]+1)

    if len(neg.size()) == 0 or \
            neg.size() == torch.Size([0]):
        neg_loss = zero_loss  #problem here
    else:
        pred_cls_neg = torch.index_select(pred_cls, 0, neg)[:, 1]
        pred_cls_neg = pred_cls_neg.clamp(min=0.000001, max=0.9999999)
        absolute_loss_neg = torch.abs(label_cls_new[neg] - pred_cls_neg)
        reg_loss_neg = -0.5*absolute_loss_neg * torch.log(1 - pred_cls_neg)
        neg_loss = reg_loss_neg.sum() / (reg_loss_neg.shape[0]+1)
    reg_loss = pos_loss + neg_loss
    return reg_loss



def select_l1_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.gt(0).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)
    return kalyo_l1_loss(pred_loc, label_loc) #+ 0.5 * kalyo_l1_loss(pred_loc_add, label_loc_add)

def select_l1_loss_c(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    max_c = torch.max(label_cls.data)
    pos = label_cls.data.gt(max_c - 0.2).nonzero().squeeze().cuda()
    cur_device = pred_loc.device
    zero_loss = torch.tensor(0.0).to(cur_device)
    if len(pos.size()) == 0 or \
            pos.size() == torch.Size([0]):
        loss_pos = zero_loss
        return loss_pos
    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 2)
    pred_loc = torch.index_select(pred_loc, 0, pos)
    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 2)
    label_loc = torch.index_select(label_loc, 0, pos)
    absolute_loss = torch.abs(pred_loc - label_loc)
    square_loss = 0.5 * ((label_loc - pred_loc)) ** 2
    inds = absolute_loss.lt(1).float()
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    tsz = label_loc.size()[0] * label_loc.size()[1]+1
    reg_loss = reg_loss.sum()/tsz#weighted loss
    return reg_loss


def select_l1_loss_lp(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.clone()
    pos = label_cls_new.data.gt(0).nonzero().squeeze().cuda()
    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)
    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)
    absolute_loss = torch.abs(pred_loc - label_loc)
    square_loss = 0.5 * ((label_loc - pred_loc)) ** 2
    inds = absolute_loss.lt(1).float()
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    reg_loss = (reg_loss[:,1]).sum()/(pos.sum()) #weighted loss
    reg_loss = (reg_loss.sum())/(pos.sum()) #weighted loss
    tsz = label_loc.size()[0] * label_loc.size()[1]+1
    reg_loss = (reg_loss.sum())/tsz #weighted loss
    return reg_loss


def kalyo_l1_loss(output, target, norm=False):
    tsz = output.size()[0] * output.size()[1]+1
    # w, h
    absolute_loss = torch.abs(target - output)
    square_loss = 0.5 * (target - output) ** 2
    if norm:
        absolute_loss = absolute_loss / (target[:, :2] + 1e-10)
        square_loss = square_loss / (target[:, :2] + 1e-10) ** 2
    inds = absolute_loss.lt(1).float()
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    return reg_loss.sum()/tsz

