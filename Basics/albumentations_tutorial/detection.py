import cv2
import albumentations as A
import numpy as np
from albumentations_tutorial.utils import plot_examples
from PIL import Image

image = cv2.imread('images/cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[13, 170, 224, 410]]  # corner

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=1080),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),  # BORDER_CONSTANT : 회전했을때 남은부분 검은색으로
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ], bbox_params= A.BboxParams(format='pascal_voc', min_area=2048,
                                 min_visibility=0.3, label_fields=[])
)
'''
format
pascal_voc = x_min, y_min, x_max, y_max
albumentations = x_min, y_min, x_max, y_max + Normalize
coco = x_min, y_min, width, height
yolo = x_center, y_center, width, height

min_area
augmentation 후의 bbox의 area가 min_area의 값보다 작은 경우 해당 박스는 사용하지 않음.

min_visibility
0~1 사이 값을 가지며 원본 bbox와 augmentation bbox의 면적 비율이 해당 값보다 작은 경우 사용되지 않음.

label_fields
class_labels = ['cat', 'dog', 'parrot']
class_categories = ['animal', 'animal', 'item']

label_fields=['class_labels', 'class_categories'] 와 같이 사용 가능
후에 'image' or 'bboxes'처럼 'class_labels','class_categories'로 접근가능
'''

# 원본 추가
images_list = [image]
saved_bboxes = [bboxes[0]]


for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_img = augmentations['image']

    if len(augmentations['bboxes']) == 0:  # 위의 조건들에 의해 bbox가 살아남지 못했으면 그땐 저장하지 않음
        continue
    print(augmentations['bboxes'])
    images_list.append(augmented_img)
    saved_bboxes.append(augmentations['bboxes'][0])  # utils 함수에서 튜플로 받아야 해서 [()] -> () 로 풀어줌

plot_examples(images_list, saved_bboxes)