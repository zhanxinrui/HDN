import os

import cv2
import numpy as np

from hdn.core.config import cfg

inDir = cfg.BASE.BASE_PATH + 'siamban_liyang_86/hdn/experiments/tracker_homo_config/results/POT/HDN'
outDir = cfg.BASE.BASE_PATH + 'POT/results/HDNHomography/'

if __name__ == "__main__":
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    p_name = os.listdir(inDir)
    for infile_base in p_name:
        infile = os.path.join(inDir,infile_base)
        with open(infile,"r") as f:
            lines = f.readlines()
        refPoints = np.float32([float(t) for t in lines[0].strip().split(" ")]).reshape(4,2)
        outfile = outDir + os.path.basename(infile)[:-4] + "_homography.txt"
        f = open(outfile, "w")
        for line in lines:
            x1, y1, x2, y2, x3, y3, x4, y4 = [float(t) for t in line.strip().split(" ")]
            curPoints = np.float32([float(t) for t in line.strip().split(" ")]).reshape(4, 2)
            H, mask= cv2.findHomography(refPoints, curPoints, cv2.RANSAC)
            try:
                H = H.reshape(1,9).tolist()[0]
            except:
                H = [1,0,0, 0,1,0, 0,0,1]
            f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"%(H[0],H[1],H[2],H[3],H[4],H[5],H[6],H[7],H[8]))
        f.close()

