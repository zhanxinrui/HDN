import cv2
import os
from hdn.core.config import cfg

if __name__ == "__main__":
    import os, shutil
    result_path = cfg.BASE.PROJ_PATH + 'experiments/hdn_r50_l234_pot/results/POT/model_otb'
    new_result_path = cfg.BASE.PROJ_PATH + 'experiments/hdn_r50_l234_pot/results/POT/model_otb_convert_with_minus'

    for i in os.listdir(result_path):
        if not os.path.exists(new_result_path):
            os.mkdir(new_result_path)
        origin_name = os.path.join(result_path, i)
        result_name = os.path.join(new_result_path,i)
        with open(origin_name,"r") as f:
            lines = f.readlines()
        with open(result_name,"w") as f:
            for line in lines:
                H = line.strip().split(',')
                H =  [float(i)for i in H]
                print(H)
                f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
                H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7]))
