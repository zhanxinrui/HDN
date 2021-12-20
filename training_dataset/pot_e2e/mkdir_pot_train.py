import cv2
import os
from hdn.core.config import cfg
'''This script is using for create POT dataset for training.'''

if __name__ == "__main__":

    import os, shutil
    video_type_num = 7 #select num of trans types
    start_type = 1
    start_v = 1#1 16
    end_v = 31# 31 17
    # video2img(video_path, frame_save_dir)
    # result_path = cfg.BASE.BASE_PATH + 'hdn/experiments/hdn_r50_l234_pot/results/POT/model_otb'
    # result_path = cfg.BASE.BASE_PATH + 'hdn-map/experiments/hdn_r50_l234_pot/results/POT/model_vot'
    # result_path = cfg.BASE.BASE_PATH + 'hdn_liyang/hdn/experiments/hdn_r50_l234_pot/results/POT/checkpoint_e10'
    base_path =   cfg.BASE.DATA_PATH +'POT_train_e2e/'

    # base_path = cfg.BASE.DATA_ROOT + 'SOT/POT/POT_annotation/"
    # anno_path = cfg.BASE.DATA_ROOT + 'SOT/POT/POT_annotation/annotation/"
    # txt_Results = cfg.BASE.BASE+ 'hdn/experiments/hdn_r50_l234_otb/results/OTB100/model_otb'
    for i in range(start_v,end_v):
        for j in range(start_type,video_type_num+1):
            if j != 3 and j != 7:
                continue
            v_path = os.path.join(base_path,'V%02d'%i,'V%02d_%d'%(i,j))
            print('v_path',v_path)
            if not os.path.exists(v_path):
                os.makedirs(v_path)
            new_img_path = os.path.join(v_path,'img')
            # print('new_img_path',new_img_path)
            # if not os.path.exists(new_img_path):
            #     os.makedirs(new_img_path)
            ori_img_src = os.path.join(cfg.BASE.DATA_PATH + '/POT/','V%02d'%i,'V%02d_%d'%(i,j))
            print('ori_img_src',ori_img_src)
            try:
                os.symlink(ori_img_src,new_img_path,True)
                ori_gt_src = os.path.join(cfg.BASE.DATA_PATH + '/POT_annotation/','V%02d_%d_gt_points.txt'%(i,j))
                shutil.copy(ori_gt_src, v_path)
                print('ori_gt_src',ori_gt_src)
            except:
                pass

    # for i in os.listdir(result_path):
    #     # if '-' in i:
    #     origin_name = os.path.join(result_path, i)
    #     new_name = origin_name[:-4]+'_siamldes_rot_v3'+origin_name[-4:]
    #     # new_name = origin_name.replace('-','.')
    #     print('orgin_name',origin_name)
    #     print('new_name',new_name)
    #     os.rename(origin_name,new_name)

    # for i in range(1,31):
    #     for j in range(1,8):
    #         new_v_dir = base_path+'V%02d_%d'%(i,j)
    #         os.mkdir(new_v_dir)
    #         flag_dir = anno_path+'V%02d_%d'%(i,j)+'_flag.txt'
    #         gt_points = anno_path+'V%02d_%d'%(i,j)+'_gt_points.txt'
    #         gt_homography = anno_path+'V%02d_%d'%(i,j)+'_gt_homography.txt'
    #         print('new_v_dir',new_v_dir)
    #         print('flag_dir',flag_dir)
    #         print('gt_points',gt_points)
    #         print('gt_homography',gt_homography)
    #
    #         # shutil.move(flag_dir,new_v_dir)
    #
