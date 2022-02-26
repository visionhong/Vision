from __future__ import print_function, division
import sys
import os
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image


# root_dir : E:\\Computer Vision\\data\\COCO
class CocoDataset(Dataset):

    def __init__(self, root_dir, set_name='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations',
                                      'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()  # 이미지 ID를 반환해주는 함수

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        # [..., {"supercategory": "person","id": 1,"name": "person"}, ...]
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']  # 0 : 1
            self.coco_labels_inverse[c['id']] = len(self.classes)  # 1 : 0
            self.classes[c['name']] = len(self.classes)  # person : 0

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx).astype('uint8')
        annot = self.load_annotations(idx).astype('int')
        # sample = {'img':img, 'annot':annot}
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample

        bbox = annot[:, :4]
        labels = annot[:, 4]
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
        return {'image': img, 'bboxes': bbox, 'category_id': labels}



    def load_image(self, image_index):
        # image_info
        # {"license": 3,"file_name": "000000391895.jpg","coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
        # "height": 360,"width": 640,"date_captured": "2013-11-14 11:18:45",
        # "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg","id": 391895}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]  # [0]의 의미?
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0,5))  # print해보면 빈 리스트  dim : (0,5)

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:  # bbox : [x_min, y_min, width, height]
                continue

            annotation = np.zeros((1,5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])  # label index
            annotations = np.append(annotations, annotation, axis=0)  # (0,5) -> (1,5) -> (2,5) ...

        # transform from [x,y,w,h] to [x1,y1,x2,y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


if __name__ == '__main__':
    from augmentation import get_augmentation
    import matplotlib.pyplot as plt
    dataset = CocoDataset(root_dir='E:\\Computer Vision\\data\\COCO',
                          set_name='train2017',
                          transform=get_augmentation(phase='train'))

    sample = dataset[0]
    print('sample: ', sample)


    img = np.array(sample['image'].permute(1, 2, 0))
    bboxes = sample['bboxes']

    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,0), 2)
    plt.imshow(img)
    plt.show()
















