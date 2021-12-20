import cv2
from hdn.core.config import cfg
import os

if __name__ == "__main__":
    # ad_video_root = cfg.BASE.PROJ_PATH + 'demo/videos/dior.mp4' # 'chanel.mp4' 'dior'
    ad_video_root = cfg.BASE.PROJ_PATH + 'demo/t5_videos/replace-video/zju-view.mp4' # 'chanel.mp4' 'dior'[demo/videos/t5_videos/replace-video/21centfox.mp4]
    # img_out_path = cfg.BASE.PROJ_PATH + 'demo/video_replace/frames/dior/' # 'ad_chanel' 'ad_dior'
    img_out_path = cfg.BASE.PROJ_PATH + 'demo/t5_videos/replace-video/zju-view/' # 'ad_chanel' 'ad_dior'
    print('imgout path', img_out_path)
    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)
    print('ad_video_root', ad_video_root)
    print('img_out_path', img_out_path)
    vc=cv2.VideoCapture(ad_video_root)
    c=1
    if vc.isOpened():
        print('yes')
        rval,frame=vc.read()
    else:
        rval=False
    while rval:
        cv2.imwrite(img_out_path+str(c)+'.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
        c=c+1
        cv2.waitKey(1)
        rval,frame=vc.read()
vc.release()