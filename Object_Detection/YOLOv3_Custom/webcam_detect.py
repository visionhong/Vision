import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import config
from model import YOLOv3
from util import cells_to_bboxes, non_max_suppression, show_image, my_non_max_suppression
import pdb
import time

cap = cv2.VideoCapture(1)

torch.backends.cudnn.benchmark = True
model = YOLOv3(num_classes=config.NUM_CLASSES, backbone='darknet53').to(config.DEVICE)
checkpoint = torch.load('checkpoint.pth.tar', map_location=config.DEVICE)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

# colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(11)]
colors = [[0, 0, 255], [0, 138, 255],[0, 153, 207], [0, 74, 36], [135, 254, 186], [46, 252, 255], [164, 0, 115],
         [164, 175, 46], [164, 175, 255], [2, 1, 70], [56, 232, 187]]

S = [13, 26, 52]
scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # (3, 3, 2)
scaled_anchors = scaled_anchors.to(config.DEVICE)
pTime = 0
while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image pre-processing before predict
    img = config.inference_transforms(image=img)['image']
    img = img.to(config.DEVICE)
    if img.ndimension() == 3:  # channels, width, height
        img = img.unsqueeze(0)  # 1 batch

    with torch.no_grad():
        output = model(img)

    boxes = []
    for i in range(3):
        anchor = scaled_anchors[i]  # tensor(3, 2)
        # print(anchor.shape)
        # print(output[i].shape)
        boxes += cells_to_bboxes(
            output[i], is_preds=True, S=output[i].shape[2], anchors=anchor
        )[0]  # batch 제외 (num_anchors * S * S, 6)

    #boxes = non_max_suppression(boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD, box_format='midpoint')
    boxes = my_non_max_suppression(boxes, iou_threshold=0.3, threshold=config.CONF_THRESHOLD, score_threshold=0.3, box_format='midpoint', method='greedy')
    boxes = [box for box in boxes if box[1] > 0.3]  # nms에서 0.0인 confidence가 안사라지는 박스들이 있음
    # print(len(boxes))
    # boxes : [[class_pred, prob_score, x1, y1, x2, y2], ...]

    image = show_image(frame, boxes, colors)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2,lineType=cv2.LINE_AA)

    cv2.imshow('fruit_detect', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 자원 반납
cv2.destroyAllWindows()