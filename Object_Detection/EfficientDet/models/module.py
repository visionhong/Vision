import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(
                np.array([0,0,0,0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1,0.1,0.2,0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        '''
        :param boxes: anchors
        :param deltas: regression
        :return: predicted boxes
        '''
        # 중심점좌표로 바꿔서 계산한 후에 다시 꼭지점 좌표로 변환
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] *self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] *self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] *self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] *self.std[3] + self.mean[3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self, width=None, heigh=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        '''
        :param boxes: (x1, y1, w, h)
        :param img: tensor (:, :, :, :)
        :return: cliped boxes
        '''
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)  # -값을 가지지 못하도록
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 0], min=width)  # 우하단 최대값을 넘어가지 못하도록
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 1], min=height)

        return boxes


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.reg = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.reg(x)
        return out.contiguous().view(out.shape[0], -1, 4)  # (batch, num_anchors, bbox)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.cls(x)
        # out is B x C x W x H, with C = n_claaases + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)  # -1 : width x height x self.num_anchors


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]  # 8, 16, 32, 64, 128
        if sizes is None:
            self.sizes = [2**(x+2) for x in self.pyramid_levels]  # 32, 64, 128, 256, 512
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # 1, 1.26, 1.587

    def forward(self, image):
        image_shape = image.shape[2:]  # height, width
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2**x-1) // (2**x) for x in self.pyramid_levels]  # 왜 굳이 이렇게 썼을까..
        # image_shapes = [(image_shape / 2**x ) for x in self.pyramid  # 이걸 쓰면 딱 떨어지는데
        # image_shapes가 512x512 라면 3~7 에서의 이미지 사이즈는 [64, 32, 16, 8, 4]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)  # shape만 있고 아직 비어있음

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)








def generate_anchors(base_size=16, ratios=None, scales=None):
    '''
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    '''

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)  # 9

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T  # (9,2)

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]  # base size의 area

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors
'''
anchors 
array([[-11.3137085 ,  -5.65685425,  11.3137085 ,   5.65685425],
       [-14.25437949,  -7.12718975,  14.25437949,   7.12718975],
       [-17.95939277,  -8.97969639,  17.95939277,   8.97969639],
       [ -8.        ,  -8.        ,   8.        ,   8.        ],
       [-10.0793684 , -10.0793684 ,  10.0793684 ,  10.0793684 ],
       [-12.69920842, -12.69920842,  12.69920842,  12.69920842],
       [ -5.65685425, -11.3137085 ,   5.65685425,  11.3137085 ],
       [ -7.12718975, -14.25437949,   7.12718975,  14.25437949],
       [ -8.97969639, -17.95939277,   8.97969639,  17.95939277]])
'''



def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride


























