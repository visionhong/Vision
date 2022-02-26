import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT = 'E:\\Computer Vision\\data\\VOC'


# xml파일 parsing
class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        # {'aeroplane': 0, 'bicycle': 1, ...}
        self.class_to_ind = class_to_ind or {v: idx for idx, v in enumerate(VOC_CLASSES)}
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1  # T or F
            if not self.keep_difficult and difficult:  # 둘이 다르면 즉, difficult가 True 이면 다음 물체 (self.keep_difficult = False 기준)
                continue

            name = obj.find('name').text.lower().strip()  # 소문자 + 양 끝 공백제거
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# root : E:\\Computer Vision\\data\\VOC
class VOCDetection(data.Dataset):
    def __init__(self, root, image_sets=[('2007','trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(), dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s','Annotations','%s.xml')
        self._imgpath = osp.join('%s','JPEGImages','%s.jpg')
        self.ids = list()

        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC'+year)
            for line in open(osp.join(rootpath,'ImageSets','Main',name+'.txt')):
                self.ids.append((rootpath,line.strip()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]  # 튜플반환 (rootpath, img id)
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32) / 255.
        # img = img.astype('uint8')
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  # [:,5] 리스트
        target = np.array(target)  # (:, 5)
        # sample = {'img': img, 'annot': target}
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # return sample

        bbox = target[:, :4]
        labels = target[:, 4]
        print(img)
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
        return {'image': img, 'bboxes': bbox, 'category_id': labels}

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(VOC_CLASSES)

    def label_to_name(self, label):
        return VOC_CLASSES[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt

