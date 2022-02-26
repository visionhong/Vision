from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    '''
    loads class labels at 'path'
    '''
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


# 초기 weight 값 설정
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 다시보기
def rescale_boxes(boxes, current_dim, original_shape):
    '''
    Rescales bounding boxes to the original shape
    즉 현재 박스의 크기를 원본 이미지의 스케일에 맞게 변경
    '''
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))  # h가 더 크면 pad_x
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))  # w가 더 크면 pad_y
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

# a =rescale_boxes(np.array([[0,0,10,10]]), current_dim= 416, original_shape=(100,100))
# print(a)


def xywh2xyxy(x):
    ''' x :tensor (batch, num_boxes, values(x,y,w,h,conf) ? )'''
    y = x.new(x.shape)  # 이걸쓰면 값을 지정해준 values차원 말고는 값이 랜덤하게되는데 굳이 왜 쓰는거지?
    y[...,0] = x[..., 0] - x[..., 2] / 2
    y[...,1] = x[..., 1] - x[..., 3] / 2
    y[...,2] = x[..., 0] + x[..., 2] / 2
    y[...,3] = x[..., 1] + x[..., 3] / 2
    return y  # tensor

# a = torch.tensor([[1,2,3,5],[1,2,3,5]])
# print(a.new(a.shape))


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y  # numpy


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)  # conf가 tensor라면 -로 내림차순 index값을 뽑을 수 있고 그냥 리스트면 에러 / 리스트면 [::-1] 사용
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # 전부 내림차순

    #  Find unique classes
    unique_classes = np.unique(pred_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []  # 클래스별 최종 값으로 하나씩 채워짐
    for c in tqdm.tqdm(unique_classes, desc='Computing AP'):
        i = pred_cls == c  # 예측 클래스중에 c 클래스 인것만 가져와서 인덱스로 사용
        n_gt = (target_cls == c).sum()  # 현재 클래스의 gt 개수
        n_p = i.sum()  # 현재 클래스의 예측 개수

        if n_p == 0 and n_gt == 0:
            continue  # 다음 클래스로
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1- tp[i]).cumsum()  # tp가 0아니면 1이므로 1에서 빼면 FP
            tpc = (tp[i]).cumsum()  # TP


            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2* p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        한 클래스에 대한 recall과 precision
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))  # 면적을 구하기 위해 수정
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    ''' Compute true positive, predicted scores and predicted labels per sample'''
    batch_metrics = []
    for sample_i in range(len(outputs)): # sample_i는 각 이미지

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # 현재 이미지
        pred_boxes = output[:, :4]  # (num_obj(살아남은 박스 수), values)
        pred_scores = output[:, 4]  # (num_obj, class_max_confs)
        pred_labels = output[:, -1]  # (num_obj, labels)

        true_positives = np.zeros(pred_boxes.shape[0])  # 예측한 박스 개수

        # targets[:,0] 은 이미지의 인덱스값
        annotations = targets[targets[:,0] == sample_i][:, 1:]  # 해당 이미지의 정답 labels, bboxes (0~1사이 값)
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #  해당 이미지의 모든 target bbox와 비교하면서 가장 iou가 큰것의 값과 인덱스를 가져옴
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)  # (1, pred_bbox)  # (:, target_bbox)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  # 선택된 target이미지는 다음부터 선택할 수 없음
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics

# 같은 중심점에 대해 anchor와 target의 크기가 얼마나 비슷한지 비교하는(IOU를 계산하는) 함수
def bbox_wh_iou(wh1, wh2):
    '''
    :param wh1: anchor box w,h (2,)
    :param wh2: batch이미지의 모든 정답 box의 w,h (num_boxes, 2)
    :return: IOU (num_boxes,)
    '''
    wh2 = wh2.t()  # (2, num_boxes)
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    ''' Returns the IOU of two bounding boxes'''
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1+1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):

    '''
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    '''

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    #  (batch, 3, 13, 13, 85)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # (batch, 10647, 4)
    output = [None for _ in range(len(prediction))]  # 이미지 수 만큼 None list를 만듦
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]  # object confidence와 비교
        #pdb.set_trace()
        # 만약 현재 이미지에서 conf_thres에 의해 모든 박스가 사라졌다면 다음 이미지로
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]  # score기준으로 pred값 내림차순
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # keepdim=True (num_pred,) -> (num_pred, 1)

        # second dim -> (x1,y1,x2,y2,object_conf,max class_conf,max class index)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        #Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # 이제부터 top score를 가진 detection과 나머지를 계속 비교
            large_overlap = bbox_iou(detections[0,:4].unsqueeze(0), detections[:,:4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]  # 라벨이 서로 같은지
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match  # 같은 클래스에 대해서 iou가 기준이상이면 삭제대상 (즉 True가 삭제대상)
            weights = detections[invalid, 4:5]  # object confidence, keep dim
            # Merge overlapping bboxes by order of confidence

            # 해당 클래스의 best box와 삭제대상인 box들 끼리 병합해서 box좌표를 조금 더 좋게 수정하는 구문?
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()

            keep_boxes += [detections[0]]  # 살아남을 박스들 하나씩 추가
            detections = detections[~invalid] # 같은클래스면서 iou가 임계값을 넘은 값들은 제외

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)


    return output
a = torch.randn((2,10647, 85))
non_max_suppression(a, conf_thres=0.5, nms_thres=0.4)


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    '''
    if grid : 13
    :param pred_boxes: (batch, 3, 13, 13, 4) or (batch, 3, 26, 26, 4) or (batch, 3, 52, 52, 4)
    :param pred_cls: (batch, 3, 13, 13, 80) or (batch, 3, 26, 26, 80) or (batch, 3, 52, 52, 80)
    :param target: (num_boxes, 6)
    :param anchors: scaled (3, 2)
    :param ignore_thres: 0.5
    :return:
    '''

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch
    nA = pred_boxes.size(1)  # num_anchor 3
    nC = pred_cls.size(-1)  # 80
    nG = pred_boxes.size(2)  # grid_size 13 or 26 or 52

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)  # (batch, 3, 13, 13) or (batch, 3, 26, 26) or (batch, 3, 52, 52)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  # (batch, 3, 13, 13, 80) or (batch, 3, 26, 26, 80) or (batch, 3, 52, 52, 80)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG  # 현재 grid cell(13,26,52)에 대한 스케일로 bbox값을 바꿈 (num_boxes, 4)
    gxy = target_boxes[:, :2]  # (num_boxes, 2)
    gwh = target_boxes[:, 2:]  # (num_boxes, 2)
    # Get anchors with best iou

    # 3가지 ratio의 anchor박스와 현재 batch 이미지의 모든 정답 box간의 IOU계산
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # (3, num_boxes)


    # 3가지 anchor box중 iou가 큰것들 가져옴
    best_ious, best_n = ious.max(0)  # (num_boxes,)
    # Separate target values
    b, target_labels = target[:, :2].long().t()  # b: 각 이미지 index, target_labels : 해당 box label  (num_boxes,)
    gx, gy = gxy.t()  # 현재 grid size 스케일의 정답 x,y 좌표값 (num_boxes,)
    gw, gh = gwh.t()  # 현재 grid size 스케일의 정답 w,h size (num_boxes,)
    gi, gj = gxy.long().t()  # gx, gy 소수점 제거

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1  # 정답이 있는 위치(cell)은 obj_mask를 1로 바꿔준다.
    noobj_mask[b, best_n, gj, gi] = 0  # 정답이 있는 위치(cell)은 noobj_mask를 0으로 바꿔준다.

    # Set noobj mask to zero where iou exceeds ignore threshold
    # 그리고 best_iou가 아니더라도 iou 가 0.5이상인 곳의 noobj_mask를 0으로 바꿔준다.
    for i, anchor_ious in enumerate(ious.t()):  # ious.t() : (num_boxes, 3)
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    # 정답 박스의 x,y좌표값을 grid 중점에서 각 셀 중점 (0~1 사이값)으로 바꿔서 tx에 저장
    tx[b, best_n, gj, gi] = gx - gx.floor()  #  ex) gx = 7.35 이면 gx.floor()는 7 , 그러므로 정답의 중심이 위치한 셀의 x값은 0.35
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1  # 정답 labels가 있는곳을 1로 바꿔줌
    # Compute label correctness and iou at best anchor
    # 정답이 있는 위치의 예측 class와 정답 class가 같다면 그위치에 class_mask을 1로 바꿔준다.
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # 정답이 있는 위치의 예측 box와 정답 box의 iou를 구해서  iou_scores에 저장한다.
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)  # 중심점 좌표라는 의미

    tconf = obj_mask.float()  # True or False -> 1 or 0
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf