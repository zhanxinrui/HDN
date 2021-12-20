import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import cv2
import os.path as osp
from hdn.core.config import cfg
def plotPOT(im_root, res_root, anno_root, plot_root,tracker):
    for i in range(1, 31):
        for j in range(1,8):
            seq_name_prefix = "V%02d"%i
            seq_name = "V%02d_"%i+"%d"%j
            print('seq_name',seq_name)
            im_dir = osp.join(im_root,seq_name_prefix,seq_name)
            anno_txt = osp.join(anno_root,seq_name+'_gt_points.txt')
            res_txt = osp.join(res_root,seq_name+'_HDN'+".txt")
            print('res_txt',res_txt)
            plot_dir = osp.join(plot_root,seq_name)#V25_3/0001.jpg
            if not osp.exists(plot_dir):
                os.makedirs(plot_dir)
            if len(os.listdir(plot_dir)) >=501:
                continue
            anno = open(anno_txt,"r")
            res = open(res_txt,'r')
            anno_reader = csv.reader(anno, delimiter=' ')
            res_reader = csv.reader(res, delimiter=' ')
            anno_list = []
            res_list = []
            for row in anno_reader:
                anno_list.append(row)
            for row in res_reader:
                res_list.append(row)
            for idx in range(1,502):
                img_path = im_dir+'/'+"%04d"%idx+'.jpg'
                img = cv2.imread(img_path)
                anno_poly = np.array([float(x) for x in anno_list[idx-1][:8]]).astype(np.int32).reshape(-1,2)
                res_poly  = np.array([float(x) for x in res_list[idx-1][:8]]).astype(np.int32).reshape(-1,2)
                cv2.polylines(img,[anno_poly],True, (0, 255, 255), 2)
                cv2.polylines(img,[res_poly],True, (255, 255), 3)
                save_path = osp.join(plot_dir,'%04d'%idx+'.jpg')
                cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 3])
            anno.close()
            res.close()


def plotVOT18(im_root, res_root, anno_root, plot_root,tracker):
    seq_names = os.listdir(im_root)
    for seq_name in seq_names:
        if '.' in seq_name:
            continue
        im_dir =  osp.join(im_root,seq_name,'color')
        anno_txt = osp.join(anno_root, seq_name,'groundtruth.txt')

        res_txt = osp.join(res_root, seq_name,seq_name + '_001.txt')
        plot_dir = osp.join(plot_root, seq_name)
        print('plot_dir', plot_dir)
        if not osp.exists(plot_dir):
            os.makedirs(plot_dir)
        if len(os.listdir(plot_dir)) >= 10:
            print('pass')
            continue
        anno = open(anno_txt, "r")
        res = open(res_txt, 'r')
        anno_reader = csv.reader(anno, delimiter=',')
        res_reader = csv.reader(res, delimiter=',')
        anno_list = []
        res_list = []
        for row in anno_reader:
            anno_list.append(row)
        for row in res_reader:
            res_list.append(row)
        for idx in range(1,len(os.listdir(im_dir))+1):
            img_path = os.path.join(im_dir,'%08d.jpg'%idx)
            img = cv2.imread(img_path)
            anno_poly = np.array([float(x) for x in anno_list[idx - 1][:8]]).astype(np.int32).reshape(-1, 2)
            res_poly = []
            res_rect = []
            is_poly = False
            is_rect = False
            if len(res_list[idx-1]) >= 8:
                res_poly = np.array([float(x) for x in res_list[idx - 1][:8]]).astype(np.int32).reshape(-1, 2)
                is_poly = True
            elif len(res_list[idx-1]) >=4:
                res_rect = np.array([float(x) for x in res_list[idx - 1][:4]]).astype(np.int32)
                is_rect = True
            cv2.polylines(img, [anno_poly], True, (0, 255, 255), 2)
            if is_poly:
                cv2.polylines(img, [res_poly], True, (0,0, 255), 3)
            elif is_rect:
                cv2.rectangle(img, (res_rect[0], res_rect[1]),(res_rect[0] + res_rect[2], res_rect[1] + res_rect[3]), (0, 0, 255), 3)
            save_path = osp.join(plot_dir, '%08d' % idx + '.png')
            cv2.imwrite(save_path, img,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        anno.close()
        res.close()
def main():
    path_out = "data/trackrcnn_aligned_data"
    path_in = 'data/overlap_mots'
    tracker = 'HDN'


    # tracker = "hdn"
    POT_img_root = cfg.BASE.DATA_ROOT + "SOT/POT/POT-280/"  # /home/USERNAME/Downloads/Dataset/SOT/POT/POT/V29_3/img
    POT_result_root = cfg.BASE.BASE_PATH + "POT/results/"+tracker  # /home/USERNAME/SOT/POT/results/LDESNEW/V25_3_LDESNEW.txt
    POT_anno_root = cfg.BASE.DATA_ROOT + "SOT/POT/POT_annotation-280/"  # V01_1/V01_1_gt_points.txt
    POT_plot_root = cfg.BASE.BASE_PATH + "POT/results_plot/"+tracker
    plotPOT(POT_img_root, POT_result_root, POT_anno_root, POT_plot_root, tracker)



    # tracker = "hdn"
    # VOT18_img_root = cfg.BASE.DATA_ROOT + "SOT/VOT18/"#/home/USERNAME/Downloads/Dataset/SOT/VOT18/ants1/color/00000001.jpg
    # VOT18_result_root = cfg.BASE.BASE_PATH + "VOT/VOT18/results/"+tracker+'/baseline'  #/home/USERNAME/SOT/VOT/VOT18/results/siamldes/baseline/ants1
    # VOT18_anno_root = cfg.BASE.DATA_ROOT + "SOT/VOT18/"  # V01_1/V01_1_gt_points.txt
    # VOT18_plot_root = cfg.BASE.BASE_PATH + "VOT/VOT18/results_plot/"+tracker
    # #plotVOT18(VOT18_img_root,VOT18_result_root,VOT18_anno_root,VOT18_plot_root,tracker)
    #

if __name__ == "__main__":
    main()
