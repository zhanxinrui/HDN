from os.path import join
from os import listdir
import json
import numpy as np
import cv2
import math
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


with open(join('./POT_train/testing_set.txt')) as f:
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
            print('poly',frame['poly'])
            poly = np.array(frame['poly']).reshape(-1,2)
            # print('poly',poly)
            if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0 or (bbox[0] + bbox[2]) > frame_sz[0] or (
                    bbox[1] + bbox[3]) > frame_sz[1]:
                count += 1
                # print("count, [w, h], [x_min, y_min, x_max, y_max], frame_sz: ",
                #       count, [bbox[2], bbox[3]], [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], frame_sz)
                continue
            rot_rect = cv2.minAreaRect(poly)
            rot_points = cv2.boxPoints(rot_rect)#4*2
            # print('rot_points',rot_points)
            # print('rot',rot_rect)

            # snippet['{:06d}'.format(f)] = [bbox[0], bbox[1], bbox[0] + bbox[2],
            #                                bbox[1] + bbox[3]]  # (x_min, y_min, x_max, y_max)

            #
            center_x = (rot_points[0][0]+rot_points[2][0])/2
            center_y = (rot_points[0][1]+rot_points[2][1])/2
            axis1_long = np.linalg.norm([rot_points[1][0]-rot_points[0][0], rot_points[1][1]-rot_points[0][1]])
            axis2_long = np.linalg.norm([rot_points[2][0]-rot_points[1][0], rot_points[2][1]-rot_points[1][1]])

            if (axis1_long > axis2_long):
                if abs(rot_points[1][0] - rot_points[0][0]) < 0.5:
                    theta = math.pi / 2
                elif  abs(rot_points[1][1]-rot_points[0][1]) < 0.5:
                    theta = 0
                    # theta = math.pi / 2

                else:
                    theta = math.atan((rot_points[1][1]-rot_points[0][1]) / (rot_points[1][0]-rot_points[0][0]))
                w = axis1_long
                h = axis2_long
                # w1 = math.sqrt(math.pow(rot_points[1][1]-rot_points[0][1],2) + math.pow(rot_points[1][0]-rot_points[0][0],2))
                # h1 = math.sqrt(math.pow(rot_points[2][1]-rot_points[1][1],2) + math.pow(rot_points[2][0]-rot_points[1][0],2))
            else:
                if abs(rot_points[2][0]-rot_points[1][0]) < 0.5:
                    theta = math.pi / 2
                elif abs(rot_points[2][1]-rot_points[1][1]) < 0.5:#here problem
                    # print('trans',rot_points)
                    # print('rot_points',poly)
                    theta = 0
                    # theta = math.pi / 2

                else:
                    theta = math.atan((rot_points[2][1]-rot_points[1][1]) / (rot_points[2][0]-rot_points[1][0]))
                w = axis2_long
                h = axis1_long
                # h1 = math.sqrt(
                #     math.pow(rot_points[1][1] - rot_points[0][1], 2) + math.pow(rot_points[1][0] - rot_points[0][0], 2))
                # w1 = math.sqrt(
                #     math.pow(rot_points[2][1] - rot_points[1][1], 2) + math.pow(rot_points[2][0] - rot_points[1][0], 2))
            # print('w',w,'w1',w1)
            # print('h',h,'h1',h1)
            # print('theta',theta)
            if theta > math.pi/2:
                theta = theta - math.pi
            elif theta < -math.pi/2:
                theta = math.pi - theta
            # print('theta',theta)
            aligned_w = max(bbox[2], bbox[3])
            aligned_h = min(bbox[2], bbox[3])
            # print('w,h',aligned_w,aligned_h)
            # snippet['{:06d}'.format(f)] = [[bbox[0], bbox[1], bbox[0] + aligned_w,
            #                                 bbox[1] + aligned_h],
            #                                [float(w), float(h), theta,center_x,center_y]]  # (x_min, y_min, x_max, y_max) [w,h,theta]
            snippet['{:06d}'.format(f)] = [[center_x-aligned_w/2, center_y-aligned_h/2, \
                                            center_x+aligned_w/2, center_y+aligned_h/2],
                                           [float(w), float(h), theta, center_x, center_y],
                                           ]  # (x_min, y_min, x_max, y_max) [w,h,theta]

            # snippet['{:06d}'.format(f)] = [[bbox[0], bbox[1], bbox[0] + bbox[2],
            #                                 bbox[1] + bbox[3]],
            #                                [rot_rect[0][0], rot_rect[0][1], rot_rect[1][0], rot_rect[1][1],
            #                                 theta]]  # (x_min, y_min, x_max, y_max) [x1,y1,w,h,theta]
            # snippet['{:06d}'.format(f)] = [[bbox[0], bbox[1], bbox[0] + bbox[2],
            #                                bbox[1] + bbox[3]],[rot_rect[0][0],rot_rect[0][1],rot_rect[1][0],rot_rect[1][1],rot_rect[2]]]  # (x_min, y_min, x_max, y_max),(cx,cy,w,h,,theta)
            # snippet['{:06d}'.format(f)] = poly # poly 4 points
        # print('test:', video['base_path'].split("/")[-1])

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
