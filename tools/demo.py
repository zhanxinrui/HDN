#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from hdn.core.config import cfg
from hdn.models.model_builder_e2e_unconstrained_v2 import ModelBuilder
from hdn.tracker.tracker_builder import build_tracker
from hdn.utils.model_load import load_pretrain
from hdn.utils.bbox import get_axis_aligned_bbox, get_min_max_bbox, get_w_h_from_poly, get_points_from_xywh, poly2mask, xywh2xyxy
import matplotlib.pyplot as plt
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save', action='store_true',
                    help='whether visualzie result')

parser.add_argument('--mode', help='demo mode', default='tracking')
parser.add_argument('--img_insert', help='img for replacing', default='None')
parser.add_argument('--video_insert', help='video for replacing', default='None')
parser.add_argument('--mosiac_range', help='the range for pixels to be the same for mosiac', default=30)
args = parser.parse_args()
from shapely.geometry import Polygon

pts = []  # save points in init image


def draw_roi(event, x, y, flags,param):
    img2 = param[0].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        pts.pop()

    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow(param[1], img2)
#


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4') or \
            video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame




def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video:
        tmp = args.video.split('/')
        video_name = tmp[-1].split('.')[0]
        if video_name is '':
            video_name = tmp[-2]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    print('video_name', video_name, args.video)

    if  args.mode == 'video_replace':
        #prepare img names
        video_path = args.video_insert
        # insert_img_names = sorted(os.listdir(video_path))
        insert_img_names = sorted(os.listdir(video_path),  key=lambda x: int(x[:-4]) )

    for fr_idx, frame in enumerate(get_frames(args.video)):
        if first_frame:
            print('if_first_frame', first_frame)
            # build video writer
            if args.save:
                if args.video.endswith('avi') or \
                        args.video.endswith('mp4') or \
                        args.video.endswith('mov'):
                    cap = cv2.VideoCapture(args.video)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30

                # save_video_path = args.video.split(video_name)[0] + video_name + '_tracking.mp4'
                save_video_path = args.video.split(video_name)[0] + video_name +'-'+ args.mode + '.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)

            cv2.setMouseCallback(video_name, draw_roi, param=[frame, video_name])
            print('replace with image/video only support 4 points for now')
            print("[INFO] click left mouse button: choose point，click right mouse button: remove the last selected point")
            print("[INFO] press 's' to complete object selection ")
            print("[INFO] ESC to exit")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("s"):
                    break
            poly = np.array(pts).reshape(-1,2)
            gt_points = poly.reshape(-1)
            first_point = pts[0]
            geo_poly = Polygon(poly)
            cx, cy =  geo_poly.centroid.x, geo_poly.centroid.y

            max_p = np.max(poly, 0)
            min_p = np.min(poly, 0)
            align_bbox = [min_p[0], min_p[1],
                          max_p[0] - min_p[0], max_p[1] - min_p[1]]
            gt_rect = [cx, cy, align_bbox[2], align_bbox[3],0]
            tracker.init(frame, align_bbox, gt_rect, gt_points, first_point)
            first_frame = False
            cv2.destroyAllWindows()
        else:
            outputs = tracker.track_new(fr_idx, frame)
            bbox = list(map(int, outputs['bbox']))
            if frame.shape[0] < frame.shape[1]:
                w_rsz = 1080
                h_rsz = int(frame.shape[0] / frame.shape[1] *1080)
            else:
                h_rsz = 720
                w_rsz = int(frame.shape[1]/ frame.shape[0] * 720)
            if args.mode == 'tracking':
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon],
                                  True, (0, 255, 0), 2)
                    if 'mask' in outputs:
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (0, 255, 0), 3)
                frame_rsz = cv2.resize(frame, (w_rsz, h_rsz))/255
                cv2.imshow(video_name, frame_rsz)
                cv2.waitKey(40)
            #------------App 1. replace by a single img or a video-----------------------
            #This can be used to add static AD poster, video AD or add decoration for img (although we do not track the surface). and it's a toy now.
            #more you can do: replace one video object with object tracked in another video.
            #more accurate results needs align the decorations precisely for first frame.
            if args.mode == 'img_replace' or args.mode == 'video_replace':
                if not args.img_insert:
                    raise Exception('image for replacing not found')
                if args.mode == 'img_replace':
                    I_ins = cv2.imread(args.img_insert)
                if args.mode == 'video_replace':
                    if len(insert_img_names) > fr_idx:
                        img_name = os.path.join(video_path, insert_img_names[fr_idx])
                    else:
                        img_name = os.path.join(video_path, insert_img_names[-1])
                    I_ins = cv2.imread(img_name)
                # #for similarity, we lower the quality of the inserted image.
                # I_ins = cv2.resize(I_ins, (int(box_w), int(box_h)))
                I_ins_w, I_ins_h = I_ins.shape[1], I_ins.shape[0]
                I_ins_4pts = [0, 0, I_ins_w, 0, I_ins_w, I_ins_h, 0, I_ins_h]
                I_ins_4pts = np.array(I_ins_4pts).reshape(4, 2).astype(np.float32)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                else:
                    raise Exception('No polygon ouput')
                H = cv2.getPerspectiveTransform(I_ins_4pts, polygon.astype(np.float32))
                mask = np.zeros([I_ins_h, I_ins_w]).astype('float')
                if args.img_insert.endswith('.png'):
                    I_ins_w_alpha = cv2.imread(args.img_insert, cv2.IMREAD_UNCHANGED)
                    I_alpha =  I_ins_w_alpha[:,:,3]
                    I_alpha_mask = (I_alpha==0).astype('int')
                    I_alpha_mask_rev = (I_alpha!=0).astype('int')
                    mask = mask + I_alpha_mask
                    I_alpha_mask_rev = np.repeat(np.expand_dims(I_alpha_mask_rev,2), 3, axis=2)
                    I_ins = I_ins * (I_alpha_mask_rev.astype(np.int8))
                I_ins_warped =  cv2.warpPerspective(I_ins, H, (frame.shape[1], frame.shape[0]),
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask_warped = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=1)
                mask_warped_c3 = np.repeat(np.expand_dims(mask_warped,2), 3, axis=2)
                mask_warped = mask_warped_c3[:frame.shape[0],:frame.shape[1], :]

                frame = (mask_warped * frame + I_ins_warped).astype('uint8')
                frame_rsz = cv2.resize(frame, (w_rsz, h_rsz))
                cv2.imshow(video_name, frame_rsz)
                cv2.waitKey(40)

            #------------App 2: add video mosiac--------------------------#
            #For video editing
            if args.mode == 'mosiac':
                poly = outputs['polygon']
                poly_mask = poly2mask((frame.shape[0], frame.shape[1]), [poly])
                poly_mask_rev = (poly_mask==0).astype(np.uint8)
                frame_mosiac = frame[::args.mosiac_range, ::args.mosiac_range]
                frame_mosiac = cv2.resize(frame_mosiac, (frame.shape[1], frame.shape[0] ))
                poly_mask = np.repeat(np.expand_dims(poly_mask,2), 3, axis=2)
                poly_mask_rev = np.repeat(np.expand_dims(poly_mask_rev,2), 3, axis=2)
                frame = frame_mosiac * poly_mask + poly_mask_rev * frame
                frame_rsz = cv2.resize(frame, (w_rsz, h_rsz))/255
                cv2.imshow(video_name, frame_rsz)
                cv2.waitKey(40)

        if args.save:
            video_writer.write(frame)

    if args.save:
        video_writer.release()


if __name__ == '__main__':
    main()
