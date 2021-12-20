#Copyright 2021, XinruiZhan
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
def img_padding(img, sx, sy):
    """
    add padding to an image [w,h] => [w+sx*2, h+sy*2]
    :param img:
    :param sx:
    :param sy:
    :return:
    """
    padd_w = img.shape[1] + sx*2
    padd_h = img.shape[0] + sy*2

    mapping_shift = np.array([[1, 0, sx],
                              [0, 1, sy],
                              [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    sh_img = cv2.warpAffine(img, project @ mapping_shift, (padd_h, padd_w), \
                            flags=2, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return sh_img
def img_shift(img, sx, sy):
    """
    shifting the image
    :param img:
    :param sx:  shift x
    :param sy:  shift y
    :return:
    """
    mapping_shift = np.array([[1, 0, sx],
                            [0, 1, sy],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    sh_img = cv2.warpAffine(img, project @ mapping_shift, (img.shape[0], img.shape[1]), \
                            flags=2, borderMode=cv2.BORDER_REPLICATE)
    return sh_img

def img_shift_crop_w_h(img, sx, sy, nw, nh):
    """
    template obj center is placed on image center.
    use [nw,nh] to corp the image, obj still on the center.
    :param img:
    :param sx:
    :param sy:
    :param nw:
    :param nh:
    :return:
    """
    crop_shift_x = sx - img.shape[1]/2 + nh/2
    crop_shift_y = sy - img.shape[0]/2 + nw/2
    mapping_shift = np.array([[1, 0, crop_shift_x],
                            [0, 1, crop_shift_y],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    sh_img = cv2.warpAffine(img, project @ mapping_shift, (nw, nh), \
                            flags=2, borderMode=cv2.BORDER_REPLICATE)
    return sh_img


def img_rot_around_center(img, cx, cy, w, h, rot):
    """
    rotate the image(center as the pivot)
    :param img:
    :param cx:
    :param cy:
    :param w:
    :param h:
    :param rot:
    :return:
    """
    aa = cx
    bb = cy  # rot center
    cc = math.cos(rot)
    ss = math.sin(rot)
    """
    crop image with rot
    we first do the rotation before the original _crop_hwc
    [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
    [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
    [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    """

    mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss],
                            [ss, cc, bb - bb * cc - aa * ss],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    rot_img = cv2.warpAffine(img, project @ mapping_rot, (w, h),
                             flags=2,borderMode=cv2.BORDER_REPLICATE)
    return rot_img

def img_rot_scale_around_center(img, cx, cy, w, h, rot, scale):
    """
    rotate and scale the image(center as the pivot)
    :param img:
    :param cx:
    :param cy:
    :param w:
    :param h:
    :param rot:
    :param scale:
    :return:
    """
    a = scale  #isometry
    b = scale
    c = cx*(1-scale)
    d = cy*(1-scale)
    mapping = np.array([[a, 0, c],
                        [0, b, d],
                        [0, 0, 1]]).astype(np.float)
    aa = cx
    bb = cy  # rot center
    cc = math.cos(rot)
    ss = math.sin(rot)
    """
    crop image with rot
    we first do the rotation before the original _crop_hwc
    [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
    [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
    [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    """

    mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss],
                            [ss, cc, bb - bb * cc - aa * ss],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    rot_img = cv2.warpAffine(img, project @ mapping_rot@mapping, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    return rot_img



def img_shift_left_top_2_center(img):
    # shift the img left_top point to center
    center = [(img.shape[1]-1)/2, (img.shape[0]-1)//2]
    mapping_shift = np.array([[1, 0, center[0]],
                            [0, 1, center[1]],
                            [0, 0, 1]]).astype(np.float)
    #
    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    rot_img = cv2.warpAffine(img, project @ mapping_shift, (img.shape[0],img.shape[1]),
                             )
    return rot_img


def get_hamming_window(w, h, rot, sx, sy, out_size_w, out_size_h):
    """
    A hamming window map
    :param w: init ellipse w
    :param h: init ellipse h
    :param rot: rotation angle(rad)
    :param sx: center x coordinate of the window
    :param sy: center y-coordinate of the window
    :param out_size_w: output window size width
    :param out_size_h: output window size height
    :return:
    """
    alpha = 1
    w = math.floor(w * alpha)
    h = math.floor(h * alpha)
    ham_window = np.outer(np.hamming(h), np.hamming(w))
    sx -= w / 2
    sy -= h / 2
    aa = w / 2
    bb = h / 2  # rot center
    cc = math.cos(rot)
    ss = math.sin(rot)
    # """
    # we first do the rotation before the original _crop_hwc
    # [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
    # [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
    # [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    # """
    mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss + sx],
                            [ss, cc, bb - bb * cc - aa * ss + sy],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    new_ham_window = cv2.warpAffine(ham_window, project @ mapping_rot, (out_size_w, out_size_h),
                                    )
    # plt.close('all')
    # plt.imshow(new_ham_window)
    # plt.show()
    # plt.close('all')
    return new_ham_window



def get_mask_window(w, h, rot, sx, sy, out_size_w, out_size_h):
    """
    A hamming window map
    :param w: init ellipse w
    :param h: init ellipse h
    :param rot: rotation angle(rad)
    :param sx: center x coordinate of the window
    :param sy: center y-coordinate of the window
    :param out_size_w: output window size width
    :param out_size_h: output window size height
    :return:
    """
    alpha = 1
    w = math.floor(w * alpha)
    h = math.floor(h * alpha)
    window = np.ones([h,w]).astype('float32')
    sx -= w / 2
    sy -= h / 2
    aa = w / 2
    bb = h / 2  # rot center
    cc = math.cos(rot)
    ss = math.sin(rot)
    # """
    # we first do the rotation before the original _crop_hwc
    # [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
    # [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
    # [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    # """
    mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss + sx],
                            [ss, cc, bb - bb * cc - aa * ss + sy],
                            [0, 0, 1]]).astype(np.float)

    project = np.array([[1, 0, 0],
                        [0, 1, 0]]).astype(np.float)
    new_ham_window = cv2.warpAffine(window, project @ mapping_rot, (out_size_w, out_size_h),
                                    )
    # plt.close('all')
    # plt.imshow(new_ham_window)
    # plt.show()
    # plt.close('all')
    return new_ham_window
def homo_add_shift(H, shift):
    H[0][2] += shift[0]
    H[1][2] += shift[1]
    return H


def rot_scale_around_center_shift_tran(cx, cy, rot, scale, sx, sy):
    """
    this is designed for the  simi estimation, after we got the sim  estimation, if we need the transformed points, we need the homography matrix.
    this function is used to calculate the homography accroding to the rotation scale around center etc.
    :param cx:  center x
    :param cy:  center y
    :param rot:  rot angle, clockwise
    :param scale: scale ratio
    :param sx:  shift x
    :param sy: shift y
    :return:
    """
    # scale_H = rot_scale_around_center_shift_tran(cfg.TRACK.EXEMPLAR_SIZE // 2, cfg.TRACK.EXEMPLAR_SIZE // 2, 0,
    #                                              crop_points_w / cfg.TRACK.EXEMPLAR_SIZE, 0, 0)


    mapping_shift = np.array([[1, 0, sx],
                        [0, 1, sy],
                        [0, 0, 1]]).astype(np.float)
    tran = mapping_shift
    if abs(scale)>0 and scale!=1:
        a = scale
        b = scale
        c = cx*(1-scale)
        d = cy*(1-scale)
        mapping_scale = np.array([[a, 0, c],
                            [0, b, d],
                            [0, 0, 1]]).astype(np.float)
        tran = mapping_scale @ tran

    if abs(rot) > 0:
        aa = cx
        bb = cy  # rot center
        cc = math.cos(rot)
        ss = math.sin(rot)
        """
        we first do the rotation before the original _crop_hwc
        [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ cos(c), -sin(c), 0][ 1, 0, -a]
        [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sin(c),  cos(c), 0][ 0, 1, -b]
        [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
        """

        mapping_rot = np.array([[cc, -ss, aa - aa * cc + bb * ss ],
                                [ss, cc, bb - bb * cc - aa * ss ],
                                [0, 0, 1]]).astype(np.float)
        tran = mapping_rot @ tran
        # tran = mapping_rot @ mapping_scale @ mapping_shift

    return tran

def shift_tran(sx, sy):
    mapping_shift = np.array([[1, 0, sx],
                        [0, 1, sy],
                        [0, 0, 1]]).astype(np.float)
    return mapping_shift
def img_proj_trans(img, trans, w, h):
    rot_img = cv2.warpPerspective(img, trans, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    return rot_img

def find_homo_by_imgs_opencv_ransac(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype('uint8')
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype('uint8')
    # if len(im1.shape) == 3:
    #     im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #     im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # else:
    #     im1Gray = im1
    #     im2Gray = im2
    # Detect ORB features and compute descriptors.
    # plt.imshow(im1Gray)
    # plt.show()
    # im1Gray = np.expand_dims(im1Gray, 2)
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

    # Use homography
    # height, width, channels = im2.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return h


def decompose_affine(H):
    """
        [ A, B]   [ cos(c), -sin(c)][ 1, m][sx 0]
        [ C, D] = [ sin(c),  cos(c)][ 0, 1][0 sy]

    :param H:
    :return:
    """
    A = H[0][0]
    B = H[0][1]
    C = H[1][0]
    D = H[1][1]
    sx = np.sqrt(A**2 + C**2)
    theta = np.arctan2(C, A)
    msy = B*np.cos(theta) + D*np.sin(theta)
    if np.sin(theta) != 0:
        sy =  (msy * np.cos(theta) - B) / np.sin(theta)
    else:
        sy = (D - msy * np.sin(theta)) / np.cos(theta)
    m = msy / sy
    sh_x = H[0][2]
    sh_y = H[1][2]
    return sx, sy, theta, sh_x, sh_y, m

def compose_affine_homo_RKS(sx, sy, theta, sh_x, sh_y, m =0):
    """
    compose affine homography matrix accroding to rotation, K, shift matrix
        [ A, B]   [ cos(c), -sin(c)][ 1, m][sx 0]
        [ C, D] = [ sin(c),  cos(c)][ 0, 1][0 sy]
    :param H:
    :return:
    """
    p = np.cos(theta)
    q = np.sin(theta)
    A = sx * p
    B = sy * m * p  - sy * q
    C = sx * q
    D = sy * m * q + sy * p

    H = np.array([[A, B, sh_x],
                  [C, D, sh_y],
                  [0, 0, 1]]).astype(np.float32)#recover the square to rect
    return H
"""
GPU version (with batch)
"""

#combine_affine if center is origin, designed for image, because image is shrink,but groundtruth probably not.
def combine_affine_c0(nm_shift, scale, rot, scale_h, in_sz, out_sz):
    # tmp-> search  transformation
    #first, move the obj to img center, second, scale and rot. (sear_obj --rot--> --shift--> tmp_obj)
    """
    crop image with similarity
    we first do the rotation before the original _crop_hwc
    [ cos(c), -sin(c), a - a*cos(c) + b*sin(c)]   [ 1, 0, a][ sc*cos(c), -sc*sin(c), 0][ 1, 0, -a]
    [ sin(c),  cos(c), b - b*cos(c) - a*sin(c)] = [ 0, 1, b][ sc*sin(c),  sc*cos(c), 0][ 0, 1, -b]
    [      0,       0,                       1]   [ 0, 0, 1][      0,       0, 1][ 0, 0,  1]
    """
    cc = torch.cos(rot)
    ss = torch.sin(rot)
    sc = out_sz / in_sz * scale
    a = sc*cc
    b = -sc*ss
    c = sc*ss
    d = sc*cc
    if scale_h:
        affine_matrix = torch.stack([a, b, nm_shift[:, 0]/out_sz, \
                                     c, d, nm_shift[:, 1]/out_sz]) \
            .permute(1, 0).reshape([-1, 2, 3]).float()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]
    else:
        affine_matrix = torch.stack([a, b, nm_shift[:, 0], \
                                     c, d, nm_shift[:, 1]]) \
            .permute(1, 0).reshape([-1, 2, 3]).floatl()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]
    return affine_matrix


#combine_affine if center is origin, designed for image, because image is shrink,but groundtruth probably not.
def combine_affine_c0_v2(cx, cy, nm_shift, scale, rot, scale_h, in_sz, out_sz):
    # tmp-> search  transformation
    #first, move the obj to img center, second, scale and rot. (sear_obj --rot--> --shift--> tmp_obj)
    #nm_shift: the shift  origin as the reference point
    """
    crop image with similarity
    [ sc*cos(c), -sc*sin(c), -o(cx)-p(cy)+sx]    [ 1, 0, sx][ o, p, 0][ 1, 0, -cx]    [ 1, 0, sx][ sc*cos(c), -sc*sin(c), 0][ 1, 0, -cx]
    [ sc*sin(c),  sc*cos(c), -q(cx)-l(cy)+sy] <= [ 0, 1, sy][ q, l, 0][ 0, 1, -cy] <= [ 0, 1, sy][ sc*sin(c),  sc*cos(c), 0][ 0, 1, -cy]
    [      0,       0,                 1]        [ 0, 0, 1][ 0, 0, 1][ 0, 0,    1]    [ 0, 0, 1 ][      0,       0,       1][ 0, 0,  1 ]
    """
    cc = torch.cos(rot)
    ss = torch.sin(rot)
    sc = out_sz / in_sz * scale
    a = sc*cc
    b = -sc*ss
    c = sc*ss
    d = sc*cc
    cx_s = (cx - out_sz/2) * in_sz / out_sz # i don't know why we need to scale here
    cy_s = (cy - out_sz/2) * in_sz / out_sz
    sx = nm_shift[:, 0]
    sy = nm_shift[:, 1]
    # if scale_h:
    #     affine_matrix = torch.stack([a, b, (-a*(sx)-b*(sy)+cx)/out_sz, \
    #                                  c, d, (-c*(sx)-d*(sy)+cy)/out_sz]) \
    #         .permute(1, 0).reshape([-1, 2, 3]).float()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]
    # else:
    #     affine_matrix = torch.stack([a, b, (-a*(sx)-b*(sy)+cx), \
    #                                  c, d, (-c*(sx)-d*(sy)+cy)]) \
    #         .permute(1, 0).reshape([-1, 2, 3]).floatl()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]

    if scale_h:
        affine_matrix = torch.stack([a, b, (-a*(cx_s)-b*(cy_s)+sx)/out_sz, \
                                     c, d, (-c*(cx_s)-d*(cy_s)+sy)/out_sz]) \
            .permute(1, 0).reshape([-1, 2, 3]).float()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]
    else:
        affine_matrix = torch.stack([a, b, (-a*(cx_s)-b*(cy_s)+sx), \
                                     c, d, (-c*(cx_s)-d*(cy_s)+sy)]) \
            .permute(1, 0).reshape([-1, 2, 3]).float()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]

    return affine_matrix



#combine_affine if left-top point is origin, designed for image, because image is shrink,but groundtruth probably not.
def combine_affine_lt0(cx, cy, nm_shift, scale, rot, in_sz, o_sz):
    """
    cal search -> temp
    :param cx:  template center  (we estimate the image move, not the obj center move)
    :param cy:
    :param nm_shift:  search shift
    :param scale:
    :param rot:
    :param in_sz: 255
    :param o_sz: 127
    :return:
    """
    #first, move the obj center to origin , second, scale and rot. then move back to tmp center,
    """
    crop image with similarity
    [ sc*cos(c), -sc*sin(c), -o(sx+cx)-p(sy+cy)+cx]    [ 1, 0, cx][ o, p, 0][ 1, 0, -sx]    [ 1, 0, cx][ sc*cos(c), -sc*sin(c), 0][ 1, 0, -sx ]
    [ sc*sin(c),  sc*cos(c), -q(sx+cx)-l(sy+cy)+cy] <= [ 0, 1, cy][ q, l, 0][ 0, 1, -sy)] <= [ 0, 1, cy][ sc*sin(c),  sc*cos(c), 0][ 0, 1, -sy ]
    [      0,       0,                       1]        [ 0, 0, 1 ][ 0, 0, 1][ 0, 0,  1  ]    [ 0, 0, 1 ][      0,       0,        1][ 0, 0,      1]
    """
    #need to compute first shift then rot finally shift_back
    cc = torch.cos(rot)
    ss = torch.sin(rot)
    sc = scale
    o = sc*cc
    p = -sc*ss
    q = sc*ss
    l = sc*cc
    # delta_x_tmp =  cx - o_sz/2
    # delta_y_tmp = cy - o_sz/2
    # sx = -delta_x_tmp + nm_shift[:, 0] + in_sz/2
    # sy = -delta_y_tmp + nm_shift[:, 1] + in_sz/2
    sx = nm_shift[:, 0] + in_sz/2
    sy = nm_shift[:, 1] + in_sz/2

    affine_matrix = torch.stack([o, p, -o*(sx)-p*(sy)+cx, \
                                 q, l, -q*(sx)-l*(sy)+cy]) \
        .permute(1, 0).reshape([-1, 2, 3]).float()  # [6,batch_size] -> [batch_size, 6] -> [batch_size, 2, 3]
    return affine_matrix
