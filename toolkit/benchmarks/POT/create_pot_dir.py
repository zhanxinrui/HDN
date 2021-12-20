import cv2
import os
from hdn.core.config import cfg

if __name__ == "__main__":
    import os, shutil
    anno_path = cfg.BASE.DATA_PATH + "POT_annotation/"
    base_path = cfg.BASE.DATA_PATH + "POT_annotation_280/"

    for i in range(1,31):
        for j in range(1,8):
            new_v_dir = base_path+'V%02d_%d'%(i,j)
            os.mkdir(new_v_dir)
            flag_dir = anno_path+'V%02d_%d'%(i,j)+'_flag.txt'
            gt_points = anno_path+'V%02d_%d'%(i,j)+'_gt_points.txt'
            gt_homography = anno_path+'V%02d_%d'%(i,j)+'_gt_homography.txt'
            shutil.move(flag_dir,new_v_dir)
            shutil.move(gt_points,new_v_dir)
            shutil.move(gt_homography,new_v_dir)


