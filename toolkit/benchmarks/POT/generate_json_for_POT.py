import os
import csv
import json
from hdn.core.config import cfg
import argparse

parser = argparse.ArgumentParser(description='POT json')
parser.add_argument('--dataset', default='POT210', type=str, help='POT210 or POT280')
args = parser.parse_args()

if __name__ == "__main__":
    anno_path = cfg.BASE.DATA_PATH + "POT_annotation-280" #"/home/username/data/POT_annotation"
    if args.dataset == 'POT210':
        max_v = 30
    else:
        max_v = 40
    v_list = [('V'+"%02d"%i)for i in range(1, max_v+1)]
    POT_Path = cfg.BASE.DATA_PATH + 'POT'
    seq_dirs = os.listdir(POT_Path)
    json_obj = {}
    for seq_dir in seq_dirs:
        if seq_dir not in v_list or seq_dir.endswith('.json'):
            continue
        p = os.path.join(POT_Path,seq_dir)#　/home/username/data/POT/v01
        videos_dir = os.listdir(p)
        for v_dir in videos_dir:
            v_full_dir = os.path.join(POT_Path,seq_dir,v_dir)#/home/username/data/POT/v01/v01_1
            print('v_dir',v_dir)
            json_obj[v_dir] = {}
            json_obj[v_dir]['video_dir'] = v_dir
            img_name_list = [os.path.join(seq_dir,v_dir,x)for x in sorted(os.listdir((v_full_dir)))]
            json_obj[v_dir]['img_names'] = img_name_list


    '''Save annotation to obj'''

    # print('v_list', max_v, v_list)

    for v in v_list:
        for v_type in range(1,8):
            v_name = v+'_'+str(v_type)#V01_1
            print('v_name',v_name)
            v_flag = os.path.join(anno_path,v_name+'_flag.txt')
            v_gt_homo = os.path.join(anno_path,v_name+'_gt_homography.txt')
            v_gt_points = os.path.join(anno_path,v_name+'_gt_points.txt')
            flag_list = []
            gt_homo_list = []
            gt_points_list = []
            init_rect = []
            with open(v_flag,"r") as inf:
                reader = csv.reader(inf,delimiter=' ')
                row_count = 0
                for row in reader:
                    if row_count%2 != 0 or row_count == 0:
                        flag_list.append(row[0])
                    row_count+=1
            with open(v_gt_homo,"r") as inf:
                reader = csv.reader(inf,delimiter=' ')
                row_count = 0
                for row in reader:
                    gt_homo = []
                    for col in range(0,9):
                        gt_homo.append(float(row[col]))
                    gt_homo_list.append(gt_homo)
                    row_count += 1

            with open(v_gt_points,"r") as inf:
                reader = csv.reader(inf,delimiter=' ')
                row_count = 0
                for row in reader:
                    if row_count == 0:
                        init_rect = []
                        for col in range(0,8):
                            init_rect.append(float(row[col]))
                    gt_points = []
                    for col in range(0,8):
                        gt_points.append(float(row[col]))
                    gt_points_list.append(gt_points)
                    row_count += 1
            json_obj[v_name]['init_rect'] = init_rect
            json_obj[v_name]['gt_rect'] = gt_points_list
            json_obj[v_name]['flag'] = flag_list
            json_obj[v_name]['homography'] = gt_homo_list


    anno_json_path = cfg.BASE.PROJ_PATH + "testing_dataset/POT/" + args.dataset + ".json"
    with open(anno_json_path, 'w') as file_obj:
        json.dump(json_obj, file_obj)


