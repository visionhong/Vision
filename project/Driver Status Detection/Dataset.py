import os
import pandas as pd
import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from util import (
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)
from config import CFG
import pdb

class DriverDataset(Dataset):
    def __init__(self, path, df, classes, S=[76, 38, 19], anchors=CFG.anchors, transform=None):
        self.path = path
        self.df = df
        self.classes = classes
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])  # (9,2)
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3  # 3
        self.S = S
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, 'file_name']
        image = np.array(Image.open(os.path.join(self.path, image_id)))
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)

        H, W, _ = image.shape

        bboxes = []
        for object in self.df.loc[idx, 'objects']:
            bbox = object['position']  # bounding box
            bbox.append(self.classes[object['class']])  # label
            bboxes.append(bbox)


        bboxes = np.array(bboxes)
        bboxes[:, 0:-1:2] /= W
        bboxes[:, 1::2] /= H
        bboxes = bboxes.tolist()

        if self.transform:  # use albumentations
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # dim [(3,76,76,6),(3,38,38,6)(3,19,19,6)]  6 : (object_prob, x, y, w, h, class)
        targets = [torch.zeros((3, S, S, 6)) for S in self.S]

        for box in bboxes:  # 각 스케일 셀 별 하나의 anchor box scale에 target값을 설정해주는 로직

            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)  # 한개의 박스와 9개의 anchor간의 w,h iou tensor(9,)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # 내림차순 정렬의 인덱스 값
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # [False, False, False]
            for anchor_idx in anchor_indices:  # true bbox와 iou가 큰 앵커 부터
                scale_idx = anchor_idx // self.num_anchors_per_scale  # anchor_idx가 8이면 scale_idx가 2가되고 20x20 grid cell 의미 (0, 1, 2)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # 각 그리드스케일 에서 사용할 3개의 anchor 스케일 (0, 1, 2)
                S = self.S[scale_idx]  # anchor_idx가 8이면 20

                # 만약 x,y가 0.53 이라면 물체가 중앙쪽에 있을 것이다.
                # 다시말해 S가 20x20이면 10x10 셀 안에 물체의 중심이 있다는 의미 (20 / 0.53 = 10.6)
                # bbox가 0~1로 Normalize 되어있기 때문에 가능
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][
                    anchor_on_scale, i, j, 0]  # [anchor_idx, 행(y), 열(x), object probability]

                if not anchor_taken and not has_anchor[scale_idx]:  # 둘다 False(혹은 0)이어야 추가. 즉 한박스당 3개의 스케일에 한번씩만 아래 작업수행
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # object probability= 1
                    # pdb.set_trace()
                    x_cell, y_cell = S * x - j, S * y - i  # 중심점이 있는 셀에서의 위치 0~1  (위에서 i,j구할때 int를 씌우면서 사라진 소수점 값(0.6)이라고 생각하면 됨)
                    width_cell, height_cell = (
                    width * S, height * S)  # 해당 스케일(80x80 or 40x40 or 20x20)에서의 비율로 나타냄 (당연히 1보다 클 수있음)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)  # class_label이 float으로 되어있음
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config import image_transform, CFG
    from util import cells_to_bboxes
    from torch.utils.data import DataLoader


    image_path = os.path.join(CFG.path, 'sample_images')
    with open(os.path.join(CFG.path, 'sample_labels.json')) as json_file:
        json_data = json.load(json_file)

    df = pd.DataFrame(json_data['annotation'])
    classes = {'eye_opened': 0, 'eye_closed': 1, 'mouth_opened': 2, 'mouth_closed': 3, 'face': 4, 'phone': 5, 'cigar': 6}
    dataset = DriverDataset(image_path, df, classes, transform=image_transform(data='train'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    original_list = os.listdir('C:\\Users\hong\\Desktop\\sample_driving\\sample_images')
    S = CFG.S

    scaled_anchors = torch.tensor(CFG.anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # 3x3x2 * 3x3x2

    for idx, (image, targets) in enumerate(dataloader):
        boxes = []

        for i in range(3):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(targets[i], is_preds=False, S=S[i], anchors=anchor)[0] # batch 제외 (num_anchors * S * S, 6)

        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format='midpoint', method='greedy')

        inp = image[0].permute(1,2,0).to('cpu').numpy()
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        plot_image(inp, boxes)






    # for i in range(len(dataset)):
    #     img_list = []
    #     original = np.array(Image.open(os.path.join(path, original_list[i])))
    #     if len(original.shape) == 2:
    #         original = original[:, :, np.newaxis]
    #         original = np.concatenate((original, original, original), axis=-1)
    #     img_list.append(original)
    #
    #     image, bboxes, labels = dataset[i]
    #     bboxes = xywh_to_xyxy(bboxes)
    #     bboxes = (bboxes * CFG.image_size).astype(int)
    #
    #     image = bbox_visualization(image, bboxes, labels)
    #     img_list.append(image)
    #
    #     fig = plt.figure(figsize=(10, 15))
    #     columns=2
    #     row=1
    #     for i in range(2):
    #         fig.add_subplot(row, columns, i+1)
    #         plt.imshow(img_list[i])
    #
    #     plt.show()





