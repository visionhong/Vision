import cv2
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:
    epochs = 10
    batch_size = 1
    num_workers = 4
    image_size = 608
    num_classes = 7
    classes = ['eye_opened', 'eye_closed', 'mouth_opened', 'mouth_closed', 'face', 'phone', 'cigar']
    learning_rate = 1e-4
    weight_decay = 1e-6
    seed = 42
    anchors = [
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
        [(0.06, 0.12), (0.13, 0.09), (0.12, 0.24)],
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
    ]
    S = [image_size // 8, image_size // 16, image_size // 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    conf_threshold = 0.75
    MAP_IOU_THRESH = 0.5
    NMS_IOU_THRESH = 0.5
    n_fold = 5

    output_dir = './output'
    path = 'C:\\Users\hong\\Desktop\\sample_driving'
    class_dict = {'eye_opened': 0, 'eye_closed': 1, 'mouth_opened': 2, 'mouth_closed': 3, 'face': 4, 'phone': 5, 'cigar': 6}
    amp = True


def image_transform(data : str):
    scale = 1.1
    if data == 'train':
        return A.Compose(
            [

                A.LongestMaxSize(max_size=int(CFG.image_size)),
                # 초기 이미지의 비율을 유지하면서 한쪽(w,h)이 max_size와 같도록 이미지 크기 조정 (가로 세로중 한쪽이 416*1.1 이 되도록 resize)
                A.PadIfNeeded(  # 입력 이미지 size가 min_height, min_width값이 될때 까지 0으로 채움
                    min_height=int(CFG.image_size * scale),
                    min_width=int(CFG.image_size * scale),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomCrop(width=CFG.image_size, height=CFG.image_size),
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            rotate_limit=10, p=1, border_mode=cv2.BORDER_CONSTANT
                        ),
                        A.IAAAffine(shear=10, p=1, mode="constant"),  # rotate와 비슷한 느낌
                    ],
                    p=1.0,
                ),

                # A.Resize(CFG.image_size, CFG.image_size),
                A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=1),
                #A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1),
                A.Blur(p=0.3),
                # A.CLAHE(p=1),  # 쓰면 안될듯
                # A.Posterize(p=1),
                # A.Cutout(p=1),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255, ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=[])
        )

    elif data == 'valid':
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255, ),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=[])
        )
    # TTA
    elif data == 'test':
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size),
                ToTensorV2(),
            ]
        )

