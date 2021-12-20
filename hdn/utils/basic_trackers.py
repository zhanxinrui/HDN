import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from math import sin, cos, atan2, sqrt, degrees
from hdn.core.config import  cfg
sift = cv2.xfeatures2d.SIFT_create()

def find_homo_by_imgs_opencv_ORB_ransac(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype('uint8')
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype('uint8')
    orb = cv2.ORB_create(MAX_FEATURES)

    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)


    return h


def find_homo_by_imgs_opencv_SIFT_ransac_ori(im1, im2, idx=0, keypoints1=None, descriptors1=None):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    MIN_MATCH_COUNT = 10

    im1Gray = im1
    im2Gray = im2
    if keypoints1 == None:
        keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)



    FLANN_INDEX_KDTREE = 0  # kd树
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if type(descriptors2) is np.ndarray and type(descriptors1) is np.ndarray:
        if descriptors2.shape[0] >=10:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        else:
            M = np.identity(3).astype('float')
            return M
    else:
        M = np.identity(3).astype('float')
        # print('1',M)
        return M

    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:#original
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if  type(M) is not np.ndarray:
            M = np.identity(3).astype('float')
        M_i = np.identity(3).astype('float')
    else:
        M = np.identity(3).astype('float')

    return  M


def find_homo_by_imgs_opencv_SIFT_ransac(im1, im2, idx=0):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    MIN_MATCH_COUNT = 10

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype('uint8')
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype('uint8')

    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)
    FLANN_INDEX_KDTREE = 0  # kd树
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if type(descriptors2) is np.ndarray and type(descriptors1) is np.ndarray:
        if descriptors2.shape[0] >=10:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        else:
            M = np.identity(3).astype('float')
            return M
    else:
        M = np.identity(3).astype('float')
        return M

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:#original

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if  type(M) is not np.ndarray:
            M = np.identity(3).astype('float')
        M_i = np.identity(3).astype('float')
    else:
        M = np.identity(3).astype('float')

    return  M


