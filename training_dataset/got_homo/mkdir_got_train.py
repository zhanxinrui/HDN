import cv2
import os
import argparse
import glob
import numpy as np
from os.path import join
from os import listdir
'''This script is using for create GOT dataset for training.'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='GOT10k', help='your got_10k data dir')
args = parser.parse_args()
got10k_base_path = args.dir
sub_sets = sorted({'train_data', 'val_data'})
got10k = []


if __name__ == "__main__":
    import os, shutil
    video_type_num = 1 #select num of trans types
    # start_type = 1
    start_v = 1#1 16
    # end_v = 5
    end_v = 1000000# 31 17
    # end_v = 1000# 31 17
    # end_v = 30# 31 17

    # video2img(video_path, frame_save_dir)
    # result_path = cfg.BASE.BASE_PATH + 'hdn/experiments/hdn_r50_l234_pot/results/POT/model_otb'
    # result_path = cfg.BASE.BASE_PATH + 'hdn-map/experiments/hdn_r50_l234_pot/results/POT/model_vot'
    # result_path = cfg.BASE.DATA_ROOT + 'hdn_liyang/hdn/experiments/hdn_r50_l234_pot/results/POT/checkpoint_e10'
    # base_path =   cfg.BASE.DATA_PATH + 'POT_train/'
    # base_path = cfg.BASE.DATA_ROOT + "SOT/POT/POT_annotation/"
    # anno_path = cfg.BASE.DATA_ROOT + "SOT/POT/POT_annotation/annotation/"
    # txt_Results = cfg.BASE.DATA + 'experiments/hdn_r50_l234_otb/results/OTB100/model_otb'
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
            # v['base_path'] = join(sub_set, video_set)
            # v['frame'] = []
            video_base_path = join(sub_set_base_path, video_set)
            gts_path = join(video_base_path, 'groundtruth.txt')
            # gts_file = open(gts_path, 'r')
            # gts = gts_file.readlines()
            # gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

            # get image size
            # im_path = join(video_base_path, '00000001.jpg')
            # im = cv2.imread(im_path)
            target_base_path = join('./GOT_10k',sub_set, video_set)
            if not os.path.exists(target_base_path):
                os.makedirs(target_base_path)

            target_path = join('./GOT_10k',sub_set, video_set, 'V01')
            if not os.path.exists(target_path):
                os.symlink(video_base_path,target_path,True)