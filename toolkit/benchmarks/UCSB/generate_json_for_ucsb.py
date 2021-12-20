"""
generate the JSON file for UCSB dataset. UCSB is our testing dataset.  not including the homo anno
"""
import os
import csv
import json
import re
from hdn.core.config import cfg
if __name__ == "__main__":
    UCSB_Path = cfg.BASE.DATA_PATH + 'UCSB'
    seq_dirs = os.listdir(UCSB_Path)
    json_obj = {}
    for seq_dir in seq_dirs:
        if seq_dir[-3:] == 'txt' or seq_dir[-3:] == 'cpp' or seq_dir[-3:] == 'exe' or seq_dir[-2:] == '.o' or seq_dir[-4:] == 'xlsx' or seq_dir == 'groundtruth_warps' \
                or seq_dir == 'SSM' or seq_dir == 'ReinitGT' or seq_dir[-3:] == 'zip' or seq_dir == 'OptGT' or seq_dir[-2:]=='.h' or seq_dir[-4:]=='json':
            continue
        p = os.path.join(UCSB_Path,seq_dir)#　/home/username/data/UCSB/dl_bus
        v_full_dir = os.path.join(UCSB_Path,seq_dir)#/home/username/data/UCSB/v01/v01_1
        json_obj[seq_dir] = {}
        json_obj[seq_dir]['video_dir'] = seq_dir
        img_name_list = [os.path.join(seq_dir,x) for x in sorted(os.listdir((v_full_dir)))]#img path
        json_obj[seq_dir]['img_names'] = img_name_list


        #annotation to obj
        anno_path = os.path.join(UCSB_Path, seq_dir+'.txt') #"/home/username/data/UCSB/dl_bus.txt "
        with open(anno_path,"r") as inf:
            lines = inf.readlines()
            row_count = 0
            gt_points_list = []
            init_rect = []
            for row in lines:
                row = re.split(r"[ (\t)*(\n)*]+", row)
                if row_count == 0:
                    row_count += 1
                    continue
                else:
                    if row_count == 1:
                        init_rect = []
                        for col in range(1,9):
                            init_rect.append(float(row[col]))
                    gt_points = []
                    for col in range(1,9):
                        gt_points.append(float(row[col]))
                    gt_points_list.append(gt_points)
                row_count += 1

            json_obj[seq_dir]['init_rect'] = init_rect
            json_obj[seq_dir]['gt_rect'] = gt_points_list


    anno_json_path = cfg.BASE.PROJ_PATH + "testing_dataset/UCSB/UCSB.json"
    anno_json_dir = cfg.BASE.PROJ_PATH + "testing_dataset/UCSB"
    if not os.path.exists(anno_json_dir):
        os.mkdir(anno_json_dir)
    with open(anno_json_path, 'w') as file_obj:
        json.dump(json_obj, file_obj)
