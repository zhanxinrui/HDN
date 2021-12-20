#Copyright 2021, XinruiZhan
# A distribute version of training
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from hdn.utils.lr_scheduler import build_lr_scheduler
from hdn.utils.log_helper import init_log, print_speed, add_file_handler
from hdn.utils.distributed import dist_init, DistModule, reduce_gradients, \
    average_reduce, get_rank, get_world_size
from hdn.utils.model_load import load_pretrain, restore_from
from hdn.utils.average_meter import AverageMeter
from hdn.utils.misc import describe, commit
from hdn.models.model_builder_e2e_unconstrained_v2 import ModelBuilder
# from hdn.datasets.dataset.semi_supervised_dataset import BANDataset
from hdn.datasets.dataset import get_dataset

from hdn.core.config import cfg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.optim as optim
# from torchviz import make_dot
import cv2
import time
logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=0,1,2,3




def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    if cfg.BAN.BAN:
        print('cfg.DATASET.TYPE',cfg.DATASET.TYPE)
        train_dataset =get_dataset(cfg.DATASET.TYPE)
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    print('num_worker',cfg.TRAIN.NUM_WORKERS)
    #we don't have enough memory
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=False,
                              sampler=train_sampler)
    return train_loader

def build_opt_lr(model, current_epoch=0):
    model.train()
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        if cfg.TRAIN.OBJ == 'LP':
            for param in model.neck.parameters():
                param.requires_grad = False
            trainable_params += [{'params': model.neck_lp.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'NM':
            for param in model.neck_lp.parameters():
                param.requires_grad = False
            trainable_params += [{'params': model.neck.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'SIM' or cfg.TRAIN.OBJ == 'ALL':
            trainable_params += [{'params': model.neck_lp.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]
            trainable_params += [{'params': model.neck.parameters(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'HOMO':
            for param in model.neck.parameters():
                param.requires_grad = False
            for param in model.neck_lp.parameters():
                param.requires_grad = False
    # neck & head
    if cfg.TRAIN.OBJ == 'LP':
        trainable_params += [{'params': model.head_lp.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        for param in model.head.parameters():
            param.requires_grad = False
    elif cfg.TRAIN.OBJ == 'NM':
        trainable_params += [{'params': model.head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        for param in model.head_lp.parameters():
            param.requires_grad = False
    elif cfg.TRAIN.OBJ == 'SIM':
        trainable_params += [{'params': model.head_lp.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.hm_net.parameters(),
                              'lr': cfg.TRAIN.HOMO_START_LR}]
    elif cfg.TRAIN.OBJ == 'ALL':
        trainable_params += [{'params': model.head_lp.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.hm_net.parameters(),
                              'lr': cfg.TRAIN.HOMO_START_LR * cfg.TRAIN.HOMO_LR_RATIO}]

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.TRAIN.BASE_LR, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        # print('weight', weights[k])
        weight = weights[k]
        w_norm = weight.norm(2)
        w_norm_avg = weight.norm(1)/weight.shape[0]
        # print('w_norm',w_norm)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'),
                             w_norm / (1e-20 + _norm), tb_index)
        tb_writer.add_scalar('w_norm_avg' + k.replace('.', '/'),
                             w_norm_avg, tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)



def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    rank = get_rank()
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
                    cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch
    print('start epoch', cfg.TRAIN.START_EPOCH)
    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    print('num_per_epoch', num_per_epoch)

    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    cfg.TRAIN.SNAPSHOT_DIR + '/got_e2e_%s_e%d.pth' % (cfg.TRAIN.OBJ, epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))
            lr_scheduler.step(epoch)
            logger.info('epoch: {}'.format(epoch + 1))
        tb_idx = idx + start_epoch * num_per_epoch
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx + 1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        # with torch.autograd.detect_anomaly():
        optimizer.zero_grad()


        outputs = model(data)

        loss = outputs['total_loss']
        if is_valid_number(loss.data.item()):
            loss.backward()
            reduce_gradients(model)
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)

        for k, v in sorted(outputs.items()):
            # pass
            batch_info[k] = average_reduce(v.data.item())
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)
            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                    epoch + 1, (idx + 1) % num_per_epoch,
                    num_per_epoch, lr_scheduler.get_lr()[1])
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                            getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                            getattr(average_meter, k))
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
                if cfg.TRAIN.LOG_GRADS:
                    log_grads(model.module, tb_writer, tb_idx)
        end = time.time()



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
    rank, world_size = dist_init()
    print('rank', rank, 'world_size', world_size)
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        print('pretrained path', backbone_path)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()
    start_epoch = cfg.TRAIN.START_EPOCH
    print('start_epoch',start_epoch)
    #build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)
    # resume training
    RESUME_PATH = cfg.BASE.PROJ_PATH + cfg.TRAIN.RESUME
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(RESUME_PATH))
        assert os.path.isfile(RESUME_PATH), \
            '{} is not a valid file.'.format(RESUME_PATH)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, RESUME_PATH)
    # # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        print('if cfg.TRAIN.PRETRAINED')
        load_pretrain(model, cfg.TRAIN.PRETRAINED)


    dist_model = DistModule(model)
    logger.info(lr_scheduler)
    logger.info("model prepare done")
    cfg.TRAIN.START_EPOCH = start_epoch

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    # seed_torch(args.seed)
    # import heartrate
    # from heartrate import trace, files
    # heartrate.trace(browser=True,host='10.214.241.12', port=4235, files=files.path_contains('model_builder_e2e_unconstrained', 'train_e2e_unconstrained_dist','unconstrained_dataset'))
    seed_torch(args.seed)
    main()
