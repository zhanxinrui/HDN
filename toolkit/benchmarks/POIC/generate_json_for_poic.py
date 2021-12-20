"""
generate the JSON file for POIC dataset. POIC is our testing dataset.
"""
import os
import csv
import json
import re
import os.path as osp
from hdn.core.config import cfg
if __name__ == "__main__":
    POIC_Path = cfg.BASE.DATA_PATH + 'POIC'
    POIC_Path_seqs = osp.join(POIC_Path, 'sequences')
    POIC_Path_gts = osp.join(POIC_Path, 'gt')
    seq_dirs = os.listdir(POIC_Path_seqs)
    json_obj = {}
    for seq_dir in seq_dirs:
        if seq_dir[-3:] == 'txt' or seq_dir=='OptGT' or seq_dir=='tags' or seq_dir[-4:]=='json':
            continue
        p = os.path.join(POIC_Path_seqs,seq_dir)#　/home/username/data/POIC/dl_bus
        v_full_dir = os.path.join(POIC_Path_seqs,seq_dir)#/home/username/data/POIC/v01/v01_1
        json_obj[seq_dir] = {}
        json_obj[seq_dir]['video_dir'] = seq_dir
        img_name_list = [os.path.join('sequences',seq_dir,x) for x in sorted(os.listdir((v_full_dir))) if '.jpg' in x]#img path
        json_obj[seq_dir]['img_names'] = img_name_list


        #annotation to obj
        anno_path = os.path.join(POIC_Path_gts, seq_dir+'.txt') #"/home/username/data/POIC/dl_bus.txt "
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

    anno_json_path = cfg.BASE.PROJ_PATH + "testing_dataset/POIC/POIC.json"
    anno_json_dir = cfg.BASE.PROJ_PATH + "testing_dataset/POIC"
    if not os.path.exists(anno_json_dir):
        os.mkdir(anno_json_dir)
    with open(anno_json_path, 'w') as file_obj:
        json.dump(json_obj, file_obj)
