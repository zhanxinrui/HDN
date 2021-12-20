from os.path import join
from os import listdir
import json
import numpy as np
import math

print('loading json (raw got10k info), please wait 20 seconds~')
got10k = json.load(open('got10k.json', 'r'))


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


snippets = dict()

n_videos = 0
count = 0
for subset in got10k:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        snippet = dict()
        snippets[video['base_path']] = dict()
        for f, frame in enumerate(frames):
            frame_sz = frame['frame_sz']
            bbox = frame['bbox']  # (x,y,w,h)
            if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0 or (bbox[0] + bbox[2]) > frame_sz[0] or (
                    bbox[1] + bbox[3]) > frame_sz[1]:
                count += 1
                # print("count, [w, h], [x_min, y_min, x_max, y_max], frame_sz: ",
                #       count, [bbox[2], bbox[3]], [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], frame_sz)
                continue
            if bbox[2] > bbox[3]:
                w = bbox[2]
                h = bbox[3]
                theta = 0
            else:
                w = bbox[3]
                h = bbox[2]
                theta = math.pi / 2
            # aligned_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            # center_x = (bbox[0] + bbox[2]) / 2
            # center_y = (bbox[1] + bbox[3]) / 2
            # aligned_bbox = [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]
            aligned_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            center_x = (aligned_bbox[0] + aligned_bbox[2]) / 2
            center_y = (aligned_bbox[1] + aligned_bbox[3]) / 2

            snippet['{:06d}'.format(f)] = [aligned_bbox, [float(w), float(h), theta, w, h, center_x,
                                                          center_y]]  # (xmin, ymin, xmax, ymax)

        snippets[video['base_path']]['{:02d}'.format(0)] = snippet.copy()

train = {k: v for (k, v) in snippets.items() if 'train' in k}
val = {k: v for (k, v) in snippets.items() if 'val' in k}

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)
print('video: {:d}'.format(n_videos))
print('done!')
