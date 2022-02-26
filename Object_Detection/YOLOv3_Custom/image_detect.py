import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import config
import torch
from backbone.darknet53 import darknet53_model
#
# model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
# model2 = darknet53_model(1000).to(config.DEVICE)
#
# state_dict = model.state_dict()
# param_names = list(state_dict.keys())
#
# check_point = torch.load('darknet53_pretrained.pth.tar', map_location=config.DEVICE)
# pretrained_state_dict = check_point['state_dict']
# pretrained_param_names = list(check_point['state_dict'].keys())
#
# print(param_names[308:])
# # print(pretrained_param_names[:312])
# print(state_dict['layers.10.layers.3.1.bn.weight'])
# for i, param in enumerate(param_names[:312]):
#     state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
# model.load_state_dict(state_dict)
# print(state_dict['layers.10.layers.3.1.bn.weight'])
#
# # # model2.load_state_dict(check_point['state_dict'], strict=False)
# # print(model2.state_dict()['conv1.layers.0.weight'])

import cv2
import torch
import argparse
import config
from model import YOLOv3
from util import cells_to_bboxes, non_max_suppression, show_image
import pdb
import time


if __name__ == '__main__':

    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(11)]

    torch.backends.cudnn.benchmark = True
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load('checkpoint.pth.tar', map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # (3, 3, 2)
    scaled_anchors = scaled_anchors.to(config.DEVICE)

    frame = cv2.imread('data/4.jpg')
    frame = cv2.resize(frame, (416, 416))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2,0,1)).to(config.DEVICE)  # 차원변경 + tensor + cuda
    img = img / 255.0  # scaling
    if img.ndimension() == 3:  # channels, width, height
        img = img.unsqueeze(0)  # 1 batch

    output = model(img)
    boxes = []
    for i in range(output[0].shape[1]):  # y[0].shape : (batch, 3, 13, 13, 6)
        anchor = scaled_anchors[i]  # tensor(3, 2)
        print(anchor.shape)
        print(output[i].shape)
        boxes += cells_to_bboxes(
            output[i], is_preds=True, S=output[i].shape[2], anchors=anchor
        )[0]  # batch 제외 (num_anchors * S * S, 6)

    boxes = non_max_suppression(boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD, box_format='midpoint')
    print(len(boxes))
    # boxes : [[class_pred, prob_score, x1, y1, x2, y2], ...]

    image = show_image(frame, boxes,colors)

    cv2.imshow('fruit_detect', image)
    cv2.waitKey(0)