import cv2
import numpy as np
from augmentation import get_augmentation
from voc0712 import VOCDetection
import matplotlib.pyplot as plt

# 논문과 bifpn depth가 1씩 작음
EFFICIENTDET = {
    'efficientdet-d0': {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3},
    'efficientdet-d1': {'input_size': 640,
                        'backbone': 'B1',
                        'W_bifpn': 88,
                        'D_bifpn': 3,
                        'D_class': 3},
    'efficientdet-d2': {'input_size': 768,
                        'backbone': 'B2',
                        'W_bifpn': 112,
                        'D_bifpn': 4,
                        'D_class': 3},
}


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    return img

def visualize(annotations, category_id_to_name):
    img = annotations['image'].clone()
    img = np.array(img.permute(1,2,0))

    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()
    return img


dataset_root = 'E:\\Computer Vision\\data\\VOC'
network = 'efficientdet-d0'
dataset = VOCDetection(root=dataset_root,
                       transform=get_augmentation(phase='train', width=EFFICIENTDET[network]['input_size'], height=EFFICIENTDET[network]['input_size']))

def visual_data(data, name):
    img = data['image']
    bboxes = data['bboxes']
    annotations = {'image': data['image'], 'bboxes': data['bboxes'], 'category_id': range(
        len(data['bboxes']))}
    category_id_to_name = {v: v for v in range(len(data['bboxes']))}

    img = visualize(annotations, category_id_to_name)
    cv2.imwrite(name, img)

for i in range(10, 15):
    visual_data(dataset[i], "name"+str(i)+".png")

































