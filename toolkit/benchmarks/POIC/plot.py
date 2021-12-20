import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import cv2
import os.path as osp
from hdn.core.config import cfg
"""
gt format
frame ulx uly urx ury lrx lry llx lly
frame00001.jpg 207.0010 134.0020 557.9950 122.0010 562.0030 485.0090 198.9940 485.9930 
frame00002.jpg 206.9468 134.5056 556.9513 121.9893 561.7339 484.9492 200.7426 485.0564 
frame00003.jpg 206.7172 135.3556 557.6222 121.8158 562.5117 484.0848 200.3722 486.7005 
frame00004.jpg 206.1037 136.7816 556.9271 120.5655 562.4001 483.8418 200.5525 487.9390 
"""

def plotPOIC(im_root, res_root, anno_root, plot_root,tracker):
    seq_names = os.listdir(res_root)
    for seq_name in seq_names:
        seq_name = seq_name[:-4]
        im_dir = osp.join(im_root,seq_name)
        img_names = [x for x in os.listdir(im_dir) if '.jpg' in x]
        anno_txt = osp.join(anno_root,seq_name + '.txt')
        res_txt = osp.join(res_root,seq_name+".txt")# '_hdn', '_Ferns' '_SIFT'
        plot_dir = osp.join(plot_root,seq_name)#V25_3/0001.jpg
        if not osp.exists(plot_dir):
            os.makedirs(plot_dir)
        else:
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
        if (len(anno_list) - len(res_list)) > 1:
            for lk in range(len(anno_list) - len(res_list)-1):
                res_list.append([0,0,0,0,0,0,0,0])

        for idx in range(1,len(img_names)):
            img_path = im_dir+'/'+"frame%05d"%idx+'.jpg'
            img = cv2.imread(img_path)
            anno_poly = np.array([float(x) for x in anno_list[idx][1:9]]).astype(np.int32).reshape(-1,2)# start from the second line
            res_poly  = np.array([float(x) for x in res_list[idx-1][0:8]]).astype(np.int32).reshape(-1,2)
            cv2.polylines(img,[anno_poly],True, (0, 255, 255), 2)
            cv2.polylines(img,[res_poly],True, (255, 0, 255), 2)
            save_path = osp.join(plot_dir,'%05d'%idx+'.jpg')
            cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        anno.close()
        res.close()

def main():
    path_out = "data/trackrcnn_aligned_data"
    path_in = 'data/overlap_mots'
    tracker = 'got_e2e_ALL_e30_rm_unsup_simi_loss'#hdn 'FERNS', 'SIFT'
    POIC_img_root = cfg.BASE.DATA_ROOT + "SOT/POIC/sequences/"  # /home/USERNAME/Downloads/Dataset/SOT/POIC/POIC/V29_3/img
    POIC_result_root = cfg.BASE.BASE_PATH + "benchmark_results/POIC-standard/results/"+tracker  # /home/USERNAME/SOT/POIC/results/LDESNEW/V25_3_LDESNEW.txt
    POIC_anno_root = cfg.BASE.DATA_ROOT + "SOT/POIC/gt/"  # V01_1/V01_1_gt_points.txt
    POIC_plot_root =  cfg.BASE.BASE_PATH + "benchmark_results/POIC-standard/results_plot/"+tracker

    plotPOIC(POIC_img_root, POIC_result_root, POIC_anno_root, POIC_plot_root, tracker)

if __name__ == "__main__":
    main()
