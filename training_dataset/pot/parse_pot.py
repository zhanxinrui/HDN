# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default=cfg.BASE.BASE_PATH + 'hdn_liyang/hdn/training_dataset/pot/POT_train/', help='your POT data dir')
args = parser.parse_args()

pot_base_path = args.dir

pot = []
for video_set in sorted(listdir(pot_base_path)):
    if 'txt' not in video_set:
        videos = sorted(listdir(join(pot_base_path,video_set)))
        s = []
        for vi, video in enumerate(videos):
            print('video_set: {}, video id: {:04d} / {:04d}'.format(video_set, vi, len(videos)))
            v = dict()
            v['base_path'] = join(video_set, video)
            v['frame'] = []
            video_base_path = join(pot_base_path, video_set,video)
            gts_path = join(video_base_path, video+'_gt_points.txt')

            # gts_file = open(gts_path, 'r')
            # gts = gts_file.readlines()
            gts = np.genfromtxt(open(gts_path, "rb"), delimiter=' ',invalid_raise=False)

            # get image size
            im_path = join(video_base_path, 'img', '0001.jpg')
            im = cv2.imread(im_path)
            size = im.shape  # height, width
            frame_sz = [size[1], size[0]]  # width,height

            # get all im name
            jpgs = sorted(glob.glob(join(video_base_path, 'img', '*.jpg')))

            f = dict()
            for idx, img_path in enumerate(jpgs):
                if idx==0 or idx%2==1:
                    f['frame_sz'] = frame_sz
                    f['img_path'] = img_path.split('/')[-1]

                    gt = gts[idx]
                    poly = [int(g) for g in gt]
                    # print('poly',poly)
                    poly_mat = np.array(poly).reshape(4,2)
                    # we use outer bounding box here
                    max_p = np.max(poly_mat, 0)
                    min_p = np.min(poly_mat, 0)
                    bbox = [int(min_p[0]), int(min_p[1]),
                                   int(max_p[0])-int(min_p[0]), int(max_p[1]-min_p[1])]
                    # bbox = [int(g) for g in gt]   # (x,y,w,h)
                    f['bbox'] = bbox
                    f['poly'] = poly
                    v['frame'].append(f.copy())
            s.append(v)
        pot.append(s)
print('save json (raw pot info), please wait 1 min~')
json.dump(pot, open('pot.json', 'w'), indent=4, sort_keys=True)
print('pot.json has been saved in ./')
