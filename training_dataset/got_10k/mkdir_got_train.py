import cv2
import os
import argparse
import glob
import numpy as np
from os.path import join
from os import listdir
from hdn.core.config import cfg

'''This script is using for create GOT dataset for training.'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='GOT10k', help='your got_10k data dir')
args = parser.parse_args()
# got10k_base_path = args.dir
got10k_base_path = cfg.BASE.DATA_PATH + 'GOT10k'
# if not os.path.exists(got10k_base_path):
#     os.makedirs(got10k_base_path)
sub_sets = sorted({'train_data', 'val_data'})
got10k = []


if __name__ == "__main__":
    import os, shutil
    video_type_num = 1 #select num of trans types
    # start_type = 1
    start_v = 1#1 16
    # end_v = 5
    end_v = 1000000# 31 17
    # end_v = 30# 31 17

    for sub_set in sub_sets:
        sub_set_base_path = join(got10k_base_path, sub_set)
        for video_set in sorted(listdir(sub_set_base_path)):
            v_idx = int(video_set[-6:])
            if  v_idx > end_v or v_idx < start_v:
                continue
            video = join(sub_set_base_path, video_set)
            s = []
            # for vi, video in enumerate(videos):
            print('subset: {}, video_set: {}, video id: {:04d} / {:04d}'.format(sub_set, video_set, 1, 1))
            v = dict()
            video_base_path = join(sub_set_base_path, video_set)
            gts_path = join(video_base_path, 'groundtruth.txt')
            target_base_path = join('./GOT_10k',sub_set, video_set)
            if not os.path.exists(target_base_path):
                os.makedirs(target_base_path)

            target_path = join('./GOT_10k',sub_set, video_set, 'V01')
            if not os.path.exists(target_path):
                os.symlink(video_base_path,target_path,True)