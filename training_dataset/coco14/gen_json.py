from pycocotools.coco import COCO
from os.path import join
import json
import math
dataDir = '.'
count = 0
for dataType in ['val2014', 'train2014']:
    dataset = dict()
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)
    print('n_imgs', n_imgs)
    for n, img_id in enumerate(coco.imgs):
        # print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])

        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()

        for trackid, ann in enumerate(anns):
            rect = ann['bbox']
            c = ann['category_id']

            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                count += 1
                print(count, rect)
                continue
            w = rect[2]
            h = rect[3]
            theta = 0
                # aligned_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            # aligned_bbox = [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]

            dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': [bbox,\
                                                                        [float(w), float(h), theta, w, h, center_x, center_y]]}


    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)
    print('done!')

