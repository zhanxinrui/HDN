import cv2
import os
from hdn.core.config import cfg
'''This script is using for transform POT dataset from video to img frames.'''

if __name__ == "__main__":
    import os, shutil
    result_path = cfg.BASE.PROJ_PATH + 'experiments/tracker_homo_config/results/POT/HDN'

    for i in os.listdir(result_path):
        origin_name = os.path.join(result_path, i)
        new_name = origin_name[:-4]+'_HDN'+origin_name[-4:]
        print('orgin_name', origin_name)
        print('new_name', new_name)
        os.rename(origin_name, new_name)
