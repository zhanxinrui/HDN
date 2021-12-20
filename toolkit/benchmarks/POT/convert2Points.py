'''
convert the homography reuslts to the polygon form

'''

import cv2
import sys,os
import glob
import numpy as np
from hdn.core.config import cfg
import csv

inDir = cfg.BASE.BASE_PATH + "POT/results/TSA-ESM-original-H/"
outDir = cfg.BASE.BASE_PATH + "POT/results/TSA-ESM/"
anno_base_path = cfg.BASE.DATA_ROOT + "SOT/POT/POT_annotation" #"/home/username/data/POT_annotation"/home/hook/Downloads/Dataset/SOT/POT

if __name__ == "__main__":
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    p_name = os.listdir(inDir)

    for infile_base in p_name:
        seq = infile_base[:5]
        anno_path = os.path.join(anno_base_path, seq, seq +'_gt_points.txt')
        gt_points_list = []
        with open(anno_path,"r") as inf:
            reader = csv.reader(inf,delimiter=' ')
            row_count = 0
            init_points = inf.readline()
        refPoints = np.float32([float(t) for t in init_points.strip().split(" ")]).reshape(4,2)

        in_f = os.path.join(inDir,infile_base)
        out_f = os.path.join(outDir, seq + '.txt')

        with open(in_f,"r") as f:
            lines = f.readlines()
        with open  (out_f, "w") as f:
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                else:
                    homo_mat = np.float32([float(t) for t in line.strip().split(" ")]).reshape(3,3)
                    warped_points = cv2.perspectiveTransform(np.expand_dims(refPoints,0), homo_mat)
                    pts = warped_points.reshape(8)
                    f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"%(pts[0],pts[1],pts[2],pts[3],pts[4],pts[5],pts[6],pts[7]))
        # break





