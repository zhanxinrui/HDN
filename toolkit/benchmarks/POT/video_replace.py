import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import cv2
import os.path as osp
from hdn.core.config import cfg
plot_list = ['V19_7','V14_7','V11_7','V09_7','V07_7', 'V05_7', 'V04_7','V01_7']
# plot_list = ['V30_7','V02_7','V11_7', 'V29_7']



def plotPOT(im_root, res_root, anno_root, plot_root, ad_img_root, tracker):
    save_idx = 0
    for i in range(1, 31):
        for j in range(1,8):
            seq_name = "V%02d_"%i+"%d"%j
            seq_prefix = "V%02d"%i
            if seq_name not in plot_list:
                continue
            print('seq_name',seq_name)
            im_dir = osp.join(im_root,seq_prefix,seq_name)
            anno_txt = osp.join(anno_root,seq_name,seq_name+'_gt_points.txt')
            res_txt = osp.join(res_root,seq_name+'_siamBAN'+".txt")# '_hdn', '_Ferns' '_SIFT', 'got_e2e_neg_rec_e30'
            ad_name = ad_img_root.split('/')[-2]
            plot_dir = osp.join(plot_root,ad_name, seq_name)#V25_3/0001.jpg
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
                anno_poly = np.array([float(x) for x in anno_list[idx-1][:8]]).astype(np.int32).reshape(-1,2).astype(np.float32)
                res_poly  = np.array([float(x) for x in res_list[idx-1][:8]]).astype(np.int32).reshape(-1,2)
                img_path = im_dir+'/'+"%04d"%idx+'.jpg'
                ad_img_path = ad_img_root + '%d'%(save_idx%1201+1) + '.jpg'
                img = cv2.imread(img_path)#(720, 1280, 3)
                ad_img = cv2.imread(ad_img_path)
                ad_img = cv2.resize(ad_img,[1280, 720])

                ad_w, ad_h = ad_img.shape[1], ad_img.shape[0]
                ad_4points = [0, 0, ad_w, 0, ad_w, ad_h, 0, ad_h]
                ad_4points = np.array(ad_4points).reshape(4, 2).astype(np.float32)
                H = cv2.getPerspectiveTransform(ad_4points, res_poly.astype(np.float32))
                ad_warped = cv2.warpPerspective(ad_img, H, (ad_w, ad_h),
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                ad_warped = ad_warped[:img.shape[0],:img.shape[1],:]
                mask = np.zeros([ad_h, ad_w]).astype('float')
                mask_warped = cv2.warpPerspective(mask, H, (ad_w, ad_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=1)
                mask_warped_c3 = np.repeat(np.expand_dims(mask_warped,2), 3, axis=2)
                mask_warped = mask_warped_c3[:img.shape[0],:img.shape[1], :]
                sythesis = mask_warped * img + ad_warped
                save_path = osp.join(plot_dir,'%04d'%idx+'.jpg')
                save_idx+=1
                cv2.imwrite(save_path, sythesis, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            anno.close()
            res.close()



def main():
    tracker = 'siamBAN' #hdn 'FERNS', 'SIFT' 'got_e2e_neg_rec_e30'
    # ad_img_root = cfg.BASE.PROJ_PATH + 'hdn/imgs/experiments/video_replace/ad_chanel/' # 'ad_chanel' 'ad_dior'
    ad_img_root = cfg.BASE.PROJ_PATH + 'demo/t5_videos/replace-video/zju-view/' # 'ad_chanel' 'ad_dior'
    POT_img_root = cfg.BASE.DATA_ROOT + "SOT/POT/POT/"  # /home/USERNAME/Downloads/Dataset/SOT/POT/POT/V29_3/img
    POT_result_root = cfg.BASE.BASE_PATH + "POT/results/"+tracker  # /home/USERNAME/SOT/POT/results/LDESNEW/V25_3_LDESNEW.txt
    POT_anno_root = cfg.BASE.DATA_ROOT + "SOT/POT/POT_annotation/"  # V01_1/V01_1_gt_points.txt
    POT_plot_root = "/media/hook/8106e99d-fd4f-4f8e-bce1-7a2428b0c4bc/hook/PROJ/HDN/siamban_liyang_86/siamban/imgs/experiments/video_replace/processed/"+tracker
    plotPOT(POT_img_root, POT_result_root, POT_anno_root, POT_plot_root, ad_img_root,  tracker)

if __name__ == "__main__":
    main()
