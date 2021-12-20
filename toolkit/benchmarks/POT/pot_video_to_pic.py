import cv2
import os
from hdn.core.config import   cfg
'''This script is used for transform POT dataset from video to img frames.'''
def video2img(video_path, frame_save_dir):
    cap = cv2.VideoCapture(video_path)
    suc = cap.isOpened()
    frame_count = 0
    while suc:
        suc, frame = cap.read()
        frame_count += 1
        if suc:
            save_path = os.path.join(frame_save_dir, "{:04d}.jpg".format(frame_count))  # 格式化命名，不足补零
            cv2.imwrite(save_path, frame)
    cap.release()


if __name__ == "__main__":
    POT_Path = cfg.BASE.DATA_PATH + 'POT_v'
    POT_new_Path = cfg.BASE.DATA_PATH + 'POT'#'/home/username/Downloads/Dataset/SOT/POT/POT'  "/home/zhanxinrui/data/POT"
    tmp_path = POT_Path
    seq_dirs = os.listdir(POT_Path)
    # for seq_dir in ['V15','V16']:
    for seq_dir in seq_dirs:
        p = os.path.join(POT_Path,seq_dir)
        videos_dir = os.listdir(p)
        for v_dir in videos_dir:
            origin_dir = os.path.join(POT_Path,seq_dir,v_dir)
            save_dir = os.path.join(POT_new_Path,seq_dir,v_dir[:-4])
            print('save_dir',save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                continue
            print('origin_dir',origin_dir)
            video2img(origin_dir,save_dir)
