# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
from memory_profiler import profile


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

# @profile  # no memory increment
def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    print('device',device)
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')


    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if  'rf' not in k}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if  'rf_lp' not in k}

    pretrained_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# @profile
def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = 0#ckpt['epoch']
    model_state_dict = model.state_dict()
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    # ckpt_model_dict = remove_parameter(ckpt_model_dict, 'head_lp.')

    check_keys(model, ckpt_model_dict)
    # model.load_state_dict(ckpt_model_dict, strict=False)

    # 1. filter out unnecessary keys
    pretrained_dict = ckpt_model_dict
    # pretrained_dict = {k: v for k, v in cy"kpt_model_dict.items() if  '_lp.' not in k}

    # pretrained_dict = {k: v for k, v in ckpt_model_dict.items() if  'head' not in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if  ('head.' != k[:5] and 'neck.'!=k[:5])}
    # pretrained_dict = {k: v for k, v in ckpt_model_dict.items()}

    print('pretrain_dict',pretrained_dict.keys())

    model_state_dict.update(pretrained_dict)

    # 3. load the new state dict
    # model.load_state_dict(ckpt_model_dict)
    model.load_state_dict(model_state_dict)

    # check_keys(optimizer, ckpt['optimizer'])
    # optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
