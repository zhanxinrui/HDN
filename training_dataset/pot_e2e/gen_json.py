from os.path import join
from os import listdir
import json
import numpy as np
import cv2
import math
# In this version (pot_e2e), I neglect the rotation angle. cause for e2e training, all the pairs were generate from one image.
print('loading json (raw pot info), please wait 20 seconds~')
pot = json.load(open('pot.json', 'r'))


def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok


with open(join('./POT_train_e2e/testing_set.txt')) as f:
    test_set = f.read().splitlines()
print('test_set', test_set)
train_snippets = dict()
val_snippets = dict()

n_videos = 0
count = 0
for subset in pot:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        snippet = dict()
        for f, frame in enumerate(frames):
            frame_sz = frame['frame_sz']
            bbox = frame['bbox']  # (x_minx, y_min, w, h)

            # print('poly',frame['poly'])
            # print('poly',poly)
            if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0 or (bbox[0] + bbox[2]) > frame_sz[0] or (
                    bbox[1] + bbox[3]) > frame_sz[1]:
                count += 1
                # print("count, [w, h], [x_min, y_min, x_max, y_max], frame_sz: ",
                #       count, [bbox[2], bbox[3]], [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], frame_sz)
                continue
            w = bbox[2]
            h = bbox[3]
            theta = 0
            aligned_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            center_x = (aligned_bbox[0] + aligned_bbox[2]) / 2
            center_y = (aligned_bbox[1] + aligned_bbox[3]) / 2
            snippet['{:06d}'.format(f)] = [aligned_bbox,
                                           [float(w), float(h), theta, w, h, center_x,center_y],
                                           frame['poly']]


        if video['base_path'].split("/")[-1] in test_set:
            print('test:',video['base_path'].split("/")[-1])
            val_snippets[video['base_path']] = dict()
            val_snippets[video['base_path']]['{:02d}'.format(0)] = snippet.copy()
        else:
            train_snippets[video['base_path']] = dict()
            train_snippets[video['base_path']]['{:02d}'.format(0)] = snippet.copy()

json.dump(train_snippets, open('train.json', 'w'), indent=4, sort_keys=True)
json.dump(val_snippets, open('val.json', 'w'), indent=4, sort_keys=True)
print('video: {:d}'.format(n_videos))
print('done!')
