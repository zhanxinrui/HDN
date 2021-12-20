# coding: utf-8
import argparse
from homo_estimator.Deep_homography.Oneline_DLTv1.dataset import *
import numpy as np
"""
In order to get template and search images info as input of homo-estiamtor network.
"""
def get_template_info(template):
    """
    In order to preserve time, we separate the procedure of obtaining template&search info
    :param template: template image
    :return:  search info (search_tmp[1, H, W]: numpy.ndarray ,  print_tmp[3, H, W]: numpy.ndarray)
    """
    #fixme remove norm
    _mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
    _std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
    _patch_w = 127 #560
    _patch_h = 127 #315
    _rho = 0 #8 16
    _WIDTH = 127 # we still leave to space to adjust the patch　and original image. 640  560
    _HEIGHT = 127 # 360 315
    _x_mesh, _y_mesh = make_mesh(_patch_w, _patch_h)

    tmp_tensor = template[0]  # template 1*3*127*127, already croped
    tmp = tmp_tensor.cpu().permute(1, 2, 0).numpy()# HwC
    # load img2
    height, width = tmp.shape[:2]

    if height != _HEIGHT or width != _WIDTH:
        tmp = cv2.resize(tmp, (_WIDTH, _HEIGHT))

    print_tmp = tmp.copy()
    print_tmp = np.transpose(print_tmp, [2, 0, 1]) # C,H,W
    tmp = (tmp - _mean_I) / _std_I
    tmp = np.mean(tmp, axis=2, keepdims=True)
    tmp = np.transpose(tmp, [2, 0, 1])
    return tmp, print_tmp




def get_search_info(search):
    # it is the same as get_temp_info method currently
    """
    In order to preserve time, we separate the procedure of obtaining template&search info
    :param search: search image  [N, 3, H, W] torch.tensor
    :return:  search info (search_image[1, H, W]: numpy.ndarray ,  print_search[3, H, W]: numpy.ndarray)
    """
    #fixme remove norm
    _mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
    _std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
    _patch_w = 127 #560
    _patch_h = 127 #315
    _rho = 0 #16 8
    _WIDTH = 127 # 640 560
    _HEIGHT = 127 # 360 315
    search_tensor = search[0]  # template 1*3*127*127, already croped
    search = search_tensor.cpu().permute(1, 2, 0).numpy()# HwCf
    # load img2
    height, width = search.shape[:2]
    if height != _HEIGHT or width != _WIDTH:
        search = cv2.resize(search, (_WIDTH, _HEIGHT))
    print_search = search.copy()
    print_search = np.transpose(print_search, [2, 0, 1]) # C,H,W
    # search = search/255
    #fixme remove norm
    search = (search - _mean_I) / _std_I
    search = np.mean(search, axis=2, keepdims=True)
    search = np.transpose(search, [2, 0, 1])
    return search, print_search

def merge_tmp_search(tmp,search):
    """

    :param tmp: template_image[1, H, W]
    :param search: search image[1, H, W]
    :return: (org_img, input_tensor, patch_indices, four_points)
    """
    _patch_w = 127
    _patch_h = 127
    # merge
    org_img = np.concatenate([tmp, search], axis=0)
    WIDTH = org_img.shape[2]
    HEIGHT = org_img.shape[1]
    _x_mesh, _y_mesh = make_mesh(_patch_w, _patch_h) #[_patch_w, _patch_h]
    _rho = 0 #16

    x, y = 0, 0 #4,4
    input_tensor = org_img[:, y: y + _patch_h, x: x + _patch_w]
    y_t_flat = np.reshape(_y_mesh, [-1])
    x_t_flat = np.reshape(_x_mesh, [-1])
    patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)
    top_left_point = (x, y)
    bottom_left_point = (x, y + _patch_h)
    bottom_right_point = (_patch_w + x, _patch_h + y)
    top_right_point = (x + _patch_w, y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    four_points = np.reshape(four_points, (-1))
    return {'org_imgs': org_img,
            'input_tensors':input_tensor,
            'patch_indices': patch_indices,
            'four_points': four_points
            }

def get_img_info_from_dir(img_dir, img_pair_path):
    """

    :param img_dir: dataset dir("DeepHomography/Data/POT/")
    :param img_pair_path: ( "V02_2/img/0015.jpg V02_2/img/0018.jpg\n")
    :return: list ( (org_img, input_tensor, patch_indices, four_points, print_img_1, print_img_2) )
    """
    _mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
    _std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

    _patch_w = 560
    _patch_h = 315
    _rho = 16
    _WIDTH = 640
    _HEIGHT = 360
    _x_mesh, _y_mesh = make_mesh(_patch_w, _patch_h)
    img_pair = img_pair_path
    _img_path = img_dir
    pari_id = img_pair.split(' ')
    # video_name = img_pair.split('/')[0]
    print('img_path = ',_img_path + pari_id[0])
    # load img1
    if pari_id[0][-1] == 'M':
        img_1 = cv2.imread(_img_path + pari_id[0][:-2])
    else:
        img_1 = cv2.imread(_img_path + pari_id[0])

    print('img_path = ',_img_path +  pari_id[1][:-1])
    # load img2
    if pari_id[1][-2] == 'M':
        img_2 = cv2.imread(_img_path + pari_id[1][:-3])
    else:
        img_2 = cv2.imread(_img_path + pari_id[1][:-1])
    height, width = img_1.shape[:2]
    if height != _HEIGHT or width != _WIDTH:
        img_1 = cv2.resize(img_1, (_WIDTH, _HEIGHT))
    print_img_1 = img_1.copy()
    print_img_1 = np.transpose(print_img_1, [2, 0, 1])
    img_1 = (img_1 - _mean_I) / _std_I
    img_1 = np.mean(img_1, axis=2, keepdims=True)
    img_1 = np.transpose(img_1, [2, 0, 1])
    height, width = img_2.shape[:2]
    if height != _HEIGHT or width != _WIDTH:
        img_2 = cv2.resize(img_2, (_WIDTH, _HEIGHT))
    print_img_2 = img_2.copy()
    print_img_2 = np.transpose(print_img_2, [2, 0, 1])
    img_2 = (img_2 - _mean_I) / _std_I
    img_2 = np.mean(img_2, axis=2, keepdims=True)
    img_2 = np.transpose(img_2, [2, 0, 1])
    org_img = np.concatenate([img_1, img_2], axis=0)
    WIDTH = org_img.shape[2]
    HEIGHT = org_img.shape[1]
    x = np.random.randint(_rho, WIDTH - _rho - _patch_w)
    y = np.random.randint(_rho, HEIGHT - _rho - _patch_h)
    input_tensor = org_img[:, y: y + _patch_h, x: x + _patch_w]
    y_t_flat = np.reshape(_y_mesh, [-1])
    x_t_flat = np.reshape(_x_mesh, [-1])
    patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)
    top_left_point = (x, y)
    bottom_left_point = (x, y + _patch_h)
    bottom_right_point = (_patch_w + x, _patch_h + y)
    top_right_point = (x + _patch_w, y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    four_points = np.reshape(four_points, (-1))
    return (org_img, input_tensor, patch_indices, four_points, print_img_1, print_img_2)

