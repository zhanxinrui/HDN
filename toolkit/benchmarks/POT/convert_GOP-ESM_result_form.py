"""
This script is for convert the results provided by GOP-ESM  to our forms.

GOP-ESM form:
frame ulx uly urx ury lrx lry llx lly
frame00001.jpg 399.0000 150.0000 612.0000 166.0000 595.0000 508.0000 384.0000 504.0000
frame00002.jpg 397.2197 149.7962 610.5286 165.9439 593.6185 508.1770 382.2537 504.2491
frame00003.jpg 395.2773 150.1045 609.1568 166.0726 592.3595 508.6357 380.6726 504.6892

Our form:

237.000 79.000 964.000 79.000 964.000 516.000 237.000 516.000
237.000 79.000 964.000 79.000 964.000 516.000 237.000 516.000
237.000 79.000 964.000 79.000 964.000 516.000 237.000 516.000
236.000 80.000 963.000 80.000 963.000 517.000 236.000 517.000
236.000 80.000 963.000 80.000 963.000 517.000 236.000 517.000
237.000 81.000 964.000 81.000 964.000 518.000 237.000 518.000
"""


import math
import re
import cv2
import os
from hdn.core.config import cfg

import re

if __name__ == "__main__":
    import os, shutil
    result_path = cfg.BASE.BASE_PATH + 'benchmark_results/POT210-GOP-ESM-results/POT/'
    new_result_path = cfg.BASE.BASE_PATH + 'benchmark_results/POT210-standard/POT/'

    for i in os.listdir(result_path):
        tracker = i
        tracker_path = os.path.join(result_path, i)
        new_tracker_path = os.path.join(new_result_path, i)
        print('new_tracker_path', new_tracker_path)
        if not os.path.exists(new_tracker_path):
            os.mkdir(new_tracker_path)
        else:
            continue
        for j in os.listdir(tracker_path):
            origin_name = os.path.join(tracker_path, j)
            result_name = os.path.join(new_tracker_path,j[:-4]+'_'+i+j[-4:])
            with open(origin_name,"r") as f:
                lines = f.readlines()
            print('result_name',result_name)
            with open(result_name,"w") as f:
                for idx, line in enumerate(lines):
                # for line in lines:
                    if idx == 0:
                        continue
                    P = re.split(r"[ (\n)*]+", line)

                    try:
                        P =  [float(i)for i in P[1:9]]
                        if math.isnan(P[0]):
                            P = [0,0,0,0,0,0,0,0]
                    except:
                        P = [0,0,0,0,0,0,0,0]
                    f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
                        P[0],P[1], P[2], P[3], P[4], P[5], P[6], P[7]))
