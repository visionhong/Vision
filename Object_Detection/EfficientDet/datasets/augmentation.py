import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
import torch
import numpy as np
import cv2


def get_augmentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            A.LongestMaxSize(max_size=width, always_apply=True),  # 초기 이미지의 비율을 유지하면서 한쪽(w,h)이 max_size와 같도록 이미지 크기 조정
            A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=[0,0,0]),  # 입력 이미지 size가 min_height, min_width값이 될때 까지 0으로 채움
            A.RandomResizedCrop(height=height, width=width, p=1),  # 랜덤 crop하고 height, width로 다시 키움
            A.Flip(),  # horizontal, vertical flip
            A.Transpose(),  # 우측으로 90도 돌린것과 같음(width, height 바뀜)
            A.OneOf([
                A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
                A.NoOp(),  # 아무일도 하지 않음
            ]),
            A.CLAHE(p=0.8),  # 이미지가 뭔가 진해지고 선명해짐 / Doc: Apply Contrast Limited Adaptive Histogram Equalization
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    if phase == 'test' or phase == 'valid':
        list_transforms.extend([
            A.Resize(height=height, width=width)
        ])
    list_transforms.extend([
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])
    if phase == 'test':
        return A.Compose(list_transforms)
    return A.Compose(list_transforms, bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area,
                                                                   min_visibility=min_visibility, label_fields=['category_id']))


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots) # 해당 배치에서 가장많은 물체를 가진 이미지의 물체 수
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1  # -1로 주는 이유? / 5 : bbox + label

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot  # 해당 이미지의 박스 좌표값을 대입하고 남은자리는 -1유지
                annot_padded[idx, :len(annot), 4] = lab  # 라벨값도 대입
    # image만 따로 (batch, channel, w, h)로 stack하고 annot_padded를 tensor (batch, max_num_annot, 5)로 반환
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return (imgs, torch.FloatTensor(annot_padded))


class Resizer(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)

        else:
            scale = common_size / width
            resized_height = int(height*scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Agmenter(object):
    ''' Convert ndarrays in sample to Tensors'''
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:  # 50% 확률로
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]   # 좌우반전

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()  # 2번 copy 하는 이유?

            # 이미지를 좌우반전 했으므로 박스도 좌우반전
            annots[:, 0] = cols - x2
            annots[:, 1] = cols - x_tmp

            sample = {'img':image, 'annot': annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['image'], sample['annots']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}