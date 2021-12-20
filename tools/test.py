#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from hdn.core.config import cfg
from hdn.tracker.tracker_builder import build_tracker
from hdn.utils.bbox import get_axis_aligned_bbox, get_min_max_bbox, get_w_h_from_poly, get_points_from_xyxy
from hdn.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from hdn.models.model_builder_e2e_unconstrained_v2 import ModelBuilder


parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--config', default='', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
parser.add_argument("--gpu_id", default="not_set", type=str,
                    help="gpu id")
parser.add_argument('--img_w', type=int, default=640)
parser.add_argument('--img_h', type=int, default=360)
parser.add_argument('--patch_size_h', type=int, default=315)
parser.add_argument('--patch_size_w', type=int, default=560)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')

parser.add_argument('--model_name', type=str, default='resnet34')
parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained waights?')

parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.set_num_threads(1)

def main():
    # load config
    print('args.config',args.config)
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_name = args.dataset
    if args.dataset.startswith('POT'):
        dataset_name = 'POT'
    dataset_root = os.path.join(cur_dir, '../testing_dataset', dataset_name)

    # create model
    model = ModelBuilder()
    # load model
    print('args.snapshot',args.snapshot)

    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)
    exp_name = cfg.BASE.PROJ_PATH+'hdn/'
    device = torch.cuda.current_device()

    # create dataset

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    for v_idx, video in enumerate(dataset):
        print('video_name', video.name)
        # test one special video  series
        # print('video.attr[0]', video.attr[0])
        # if video.attr[0] not in  args.video_attr:
        #     continue
        # print('visibledevice',os.environ["CUDA_VISIBLE_DEVICES"])
        # if os.environ["CUDA_VISIBLE_DEVICES"] == "0":
        #     if v_idx > 55 :
        #         continue
        # elif os.environ["CUDA_VISIBLE_DEVICES"] == "1":
        #     if v_idx > 110 or  v_idx < 55:
        #         continue
        # if os.environ["CUDA_VISIBLE_DEVICES"] == "2":
        #     if v_idx > 165 or  v_idx < 110:
        #         continue
        # elif os.environ["CUDA_VISIBLE_DEVICES"] == "3":
        #     # if v_idx > 210 or  v_idx < 165:
        #     if v_idx < 165:
        #         continue

        if args.video != '':
            # test one special video
            if video.name not in  args.video:
                continue
        print('v_idx',v_idx)
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        isPolygon=False
        for idx, (img, gt_bbox) in enumerate(video):
            ###slow down
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_min_max_bbox(np.array(gt_bbox))
                if(len(gt_bbox)==8):
                    gt_points = gt_bbox
                    gt_poly = get_w_h_from_poly(np.array(gt_bbox))
                else:
                    gt_points = get_points_from_xyxy(np.array(gt_bbox))
                    gt_poly = [cx, cy, w, h, 0]

                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                first_point = np.array([gt_bbox[:2]])

                tracker.init(img, gt_bbox_, gt_poly, gt_points, first_point)
                pred_bbox = gt_bbox_
                #fixme
                if dataset_name in ['POT', 'UCSB', 'POIC']:
                    pred_bbox = np.array(gt_bbox)
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                if idx%2==1:
                    if (len(gt_bbox) == 8):
                        gt_points = gt_bbox
                        gt_poly = get_w_h_from_poly(np.array(gt_bbox))
                        cx, cy, w, h = gt_poly[0],gt_poly[1], gt_poly[2], gt_poly[3]
                    else:
                        gt_points = get_points_from_xyxy(np.array(gt_bbox))
                        cx, cy, w, h = get_min_max_bbox(np.array(gt_bbox))
                        gt_poly = [cx, cy, w, h, 0]

                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

                outputs = tracker.track_new(idx, img, gt_bbox_, gt_poly, gt_points)

                #fixme
                if dataset_name in ['POT', 'UCSB', 'POIC']:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    isPolygon = True
                    pred_bbox = np.array(outputs['polygon']).astype(np.float32).reshape(1,-1)[0]
                    bbox_align = outputs['bbox_aligned']
                elif 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.float32)
                    max_p = np.max(polygon, 0)
                    min_p = np.min(polygon, 0)
                    pred_bbox = [min_p[0], min_p[1],
                                 max_p[0]-min_p[0], max_p[1]-min_p[1]]
                    isPolygon = True
                    bbox_align = outputs['bbox_aligned']
                else:
                    pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if dataset_name in ['POT', 'UCSB', 'POIC']:
                    cv2.polylines(img, [np.array(gt_bbox).reshape(4,2).astype(np.int32)],
                                  True, (0, 255, 255), 2)
                else:
                    gt_box = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_box[0], gt_box[1]),
                                  (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
                if isPolygon:
                    cv2.polylines(img, [polygon],
                                  True, (0, 255, 0), 2)
                    cv2.rectangle(img, (int(bbox_align[0]), int(bbox_align[1])), (int(bbox_align[0])+int(bbox_align[2]), int(bbox_align[1])+int(bbox_align[3])), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 2)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        if 'VOT2018-LT' == args.dataset:
            video_path = os.path.join('results', args.dataset, model_name,
                                      'longterm', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path,
                                       '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            result_path = os.path.join(video_path,
                                       '{}_001_confidence.value'.format(video.name))
            with open(result_path, 'w') as f:
                for x in scores:
                    f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
            result_path = os.path.join(video_path,
                                       '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        elif 'GOT-10k' == args.dataset:
            video_path = os.path.join('results', args.dataset, model_name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            result_path = os.path.join(video_path,
                                       '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))

        else:
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                if dataset_name in ['POT', 'UCSB', 'POIC']:
                    for x in pred_bboxes:
                        f.write(' '.join([str(i) for i in x]) + '\n')
                else:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
