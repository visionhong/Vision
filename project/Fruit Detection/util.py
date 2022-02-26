import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import cv2

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

import pdb

def mixup_data(x, y):
    y_a = []
    y_b = []
    lam = np.random.rand()  # 0~1
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(config.DEVICE)  # randperm: batch_size 개수만큼 unique random index를 가진 1차원 tensor
    mixed_x = lam * x + (1-lam) * x[index]
    for i in range(3):
        y_a.append(y[i])
        y_b.append(y[i][index])
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, scaled_anchor):
    return lam * criterion(pred, y_a, scaled_anchor) + (1-lam) * criterion(pred, y_b, scaled_anchor)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        # pdb.set_trace()
        for i in range(3):
            mean[i] += (inputs[:,:,:,i]/255.0).mean()
            std[i] += (inputs[:,:,:,i]/255.0).std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[...,0], boxes2[..., 0]) * torch.min(boxes1[...,1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)

    return intersection / union


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    '''
    :param predictions: tensor of size (N, 3, S, S, num_classes+5)
    :param anchors: the anchors used for the predictions  (0~1) x S
    :param S: the number of cells the image is deivided in on the width and height
    :param is_preds: whether the input is predictions or the true bounding boxes
    :return: converted_bboxes: the converted boxes of sizes list(N, num_anchorsxSxS, 1+5) with class index,
                                object score, bounding box coordinates
    '''
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)  # 3
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)  # 1,3,1,1,2  broad casting
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])  # sigmoid(tx), sigmoid(tw)
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors  # e(tw) * pw
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)  # 아래서 concat하기 위해 unsqueeze 사용
        # best_class: (n, 3, 13, 13) -> (n, 3, 13, 13, 1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]


    cell_indices = (torch.arange(S)  # (13,)
                    .repeat(BATCH_SIZE,3,S,1)  # (batch, 3, S, S)
                    .unsqueeze(-1)  #
                    .to(predictions.device))  # (Batch, 3, S, S, 1)  -> x 방향으로 0,1,2,3,4,5,6,7~

    # 중심을 가진 셀에 대해 0~1 값이었던 x,y를 현재 셀 스케일 전체에 대한 Normalize 좌표로 변환 (0~1)

    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0,1,3,2,4))  # y방향으로 바꿔주고 더함
    w_h = 1 / S * box_predictions[..., 2:4]

    converted_bboxes = torch.cat((best_class,scores,x,y,w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()



def non_max_suppression(bboxes, iou_threshold, threshold, box_format='midpoint'):
    '''

    :param bboxes: list of lists containing all bboxes with each bboxes
    specified as [class_pred, prob_score, x1, y1, x2, y2]
    :param iou_threshold: threshold where predicted bboxes is coorect
    :param threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format: "midpoint" or "corners" used to specify bboxes
    :return:
        list: bboxes agter performing NMS given a specific IoU threshold
    '''
    assert type(bboxes) == list
    # input bboxes 값 확인하기
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(
                    torch.tensor(box[2:]),
                    torch.tensor(chosen_box[2:]),
                    box_format=box_format) < iou_threshold]  # chosen_box와 다른 클래스이거나 iou가 threshold값 보다 낮으면 생존

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def my_non_max_suppression(bboxes, iou_threshold, threshold, score_threshold=0.001, sigma=0.5, box_format='corners', method='linear'):
    # boxes : [[class_pred, prob_score, x1, y1, x2, y2], ...]
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = np.array(bboxes)
    bboxes_after_nms = []

    while bboxes.size > 0:
        max_idx = np.argmax(bboxes[:, 1], axis=0)
        bboxes[[0, max_idx], :] = bboxes[[max_idx, 0], :]  # 첫번째 row와 score가 가장 큰 row의 위치 전환
        bboxes_after_nms.append(bboxes[0, :].tolist())

        iou = intersection_over_union(torch.from_numpy(bboxes[1:, 2:]),
                                      torch.from_numpy(bboxes[0, 2:]),
                                      box_format=box_format)
        iou = np.array(iou.squeeze())  # tensor(n ,1)  -> numpy(n)
        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_threshold] -= iou[iou > iou_threshold]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_threshold] = 0


        bboxes[1:, 1] *= weight  # numpy * numpy (ok)
        bboxes = np.array([box.tolist() for box in bboxes[1:, :] if (bboxes[0, 0] != box[0]) or (box[1] >= score_threshold)])

        # retained_idx = np.where(box for box in bboxes if box[0,0] != bboxes[0, 0] or bboxes[1:, 1] >= score_threshold )[0]  # 다 살아남으면 0,1,2,3,4~~~
        # bboxes = bboxes[retained_idx+1, :]  # 현재 iter에서의 top score(index 0)를 제외한 살아남은 박스만 다음 반복에 포함

    return bboxes_after_nms






def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=11
):
    """
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)  # ap를 구할 클래스가 c인것만 따로 저장

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        # 각 example(batch만큼의 이미지들)이 가지고 있는 현재 클래스의 true box 개수
        amount_bboxes = Counter(gt[0] for gt in ground_truths)  # {0:3, 1:5}  인덱스:개수

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)  # amount_bboxes = {0:torch.tensor[0,0,0], 0:torch.tensor[0,0,0,0,0]}

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))   # True Positive (정답을 정답이라고 예측한 것)
        FP = torch.zeros((len(detections)))   # False Positive (배경을 정답이라고 예측한 것)
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):  # 여러 sample image 중에 한개의 detection box씩 뽑음
            # 우선 detection box의 이미지의 ground_truth를 가져옴
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)  # detection 한 이미지의 ground_truth 개수
            best_iou = 0

            # 예측 detection 박스와 그 이미지의 ground truth들과 다 비교하면서
            # iou가 가장 높은값이 그 물체를 예측하려 한 것임을 알 수 있음
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # 가장높은 iou값이 0.5보다 작다면 FP 크다면 TP
            if best_iou > iou_threshold:
                # ground truth는 한개의 detection값만 가질 수 있음
                # 즉 같은 한 물제에 두개의 예측 박스가 나오면 뒤에서 예측한 박스는 FP가 됨
                # o_p가 높은것부터 순서대로 진행이 되는데 o_p가 높다는 것은 그만큼 TP일 확률이 높기 때문에 납득 가능
                # amount_bboxes = {0:torch.tensor[0,0,0], 0:torch.tensor[0,0,0,0,0]}
                if amount_bboxes[detection[0]][best_gt_idx] == 0:  # 해당 gt를 처음 예측한 detection은 TP
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1  # iou가 높더라도 o_p가 더 높았던 이전 예측이 이미 예측한 gt이기 때문에 FP로 설정

            else:  # iou가 작다면 그냥 잘못된 예측 (background를 예측)
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)  # precision과 recall을 구하기 위해 누적합을 계산
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # TP / TP + FN   , epsilon
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls))  # recall은 0부터 시작
        precisions = torch.cat((torch.tensor([1]), precisions))
        ap = torch.trapz(precisions, recalls) # trapz : 아래 면적 계산
        average_precisions.append(ap)
        print(f"{config.CLASSES[c]} AP: {ap:.2f}")

    return sum(average_precisions) / len(average_precisions)


def xywh_to_xyxy(bboxes):
    bboxes = torch.tensor(bboxes)
    bboxes_xyxy = bboxes.clone().detach()

    bboxes_xyxy[:, 0:2] = bboxes[:, 0:2] - bboxes[:, 2:4] / 2
    bboxes_xyxy[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4] / 2
    bboxes_xyxy[:, :-1] = torch.clamp(bboxes_xyxy[:, :-1], 0, 1)

    return bboxes_xyxy.tolist()

def xyxy_to_xywh(bboxes):
    bboxes = torch.tensor(bboxes)
    bboxes_xywh = bboxes.clone().detach()
    bboxes_xywh[:, 0:2] = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2
    bboxes_xywh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
    bboxes_xywh[:, :-1] = torch.clamp(bboxes_xywh[:, :-1], 0, 1)

    return bboxes_xywh.tolist()


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    # 코드 줄여보기
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# bouning box regression에 사용될 giou loss 함수
def generalized_intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    '''
    Args:
        boxes_preds: tensor (N, 3, 13, 13, 4)
        boxes_labels: tensor (N, 3, 13, 13, 4)
        box_format: midpoint
    Returns: GIoU loss
    '''
    boxes1 = torch.clone(boxes_preds)
    boxes2 = torch.clone(boxes_labels)

    if box_format == 'midpoint':
        boxes1[..., :2] = boxes_preds[..., :2] - (boxes_preds[..., 2:] / 2)
        boxes1[..., 2:] = boxes_preds[..., :2] + (boxes_preds[..., 2:] / 2)
        boxes2[..., :2] = boxes_labels[..., :2] - (boxes_labels[..., 2:] / 2)
        boxes2[..., 2:] = boxes_labels[..., :2] + (boxes_labels[..., 2:] / 2)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    outer_min_xy = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_max_xy = torch.max(boxes1[..., 2:], boxes2[..., 2:])

    inter = torch.clamp((inter_max_xy - inter_min_xy), 0)  # -값 0으로 대체 -> 두 박스가 아예 겹쳐져 있지 않은 상황
    inter_area = inter[..., 0] * inter[..., 1]
    outer = torch.clamp((outer_max_xy - outer_min_xy), 0)
    outer_area = outer[..., 0] * outer[..., 1]

    union = area1 + area2 - inter_area
    giou = (inter_area/union) - (outer_area - union) / outer_area  # GIoU = IoU - (outer - union / outer)  빈 공간이 많을수록 giou는 작아짐
    giou = torch.clamp(giou, -1.0, 1.0)  # -1~1 사이의 값을 가짐

    loss = 1. - giou
    return loss.mean()








def plot_image(image, boxes):
    cmap = plt.get_cmap('tab20b')
    class_labels = config.CLASSES
    colors = [cmap(i) for i in np.linspace(0,1,len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),  # 좌상단 Normalize 좌표를 원본 이미지 스케일로 변환
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()

def show_image(image, boxes, colors):
    class_labels = config.CLASSES
    image = cv2.resize(image, (416,416))
    for box in boxes:
        box[2:] = list(map(lambda x: int(x *416), box[2:]))

        class_pred = int(box[0])
        prob_score = str(round(box[1], 2))
        x1 = box[2] - (box[4] // 2)
        y1 = box[3] - (box[5] // 2)
        x2 = box[2] + (box[4] // 2)
        y2 = box[3] + (box[5] // 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=colors[class_pred], thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, f'{class_labels[class_pred]}{prob_score}', (x1,y1-5), 0, 0.5, color=colors[class_pred], thickness=1,
                    lineType=cv2.LINE_AA)


    image = cv2.resize(image, (480, 640))
    return image


def get_evaluation_bboxes(
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format='midpoint',
        device='cuda',
):
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]  # 각 리스트마다 3개의 스케일에 대한 예측값이 들어감
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S  # 앞에*을 붙여서 리스트를 벗고 들어감 -> tenor변환 하면서 차원하나 축소효과
            # pdb.set_trace()
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )  # 해당 스케일의 예측값을 t에서 b로 변환하고 (batch, SxSx3, 6)으로 차원변경
            for idx, (box) in enumerate(boxes_scale_i):  # 배치만큼
                bboxes[idx] += box  # ex) 0번째 리스트에 0번째 이미지에 대한 예측 정보를 담음

        # 정답에 대한 변환값은 아무 스케일 1개에서 가져오면 됨 (cells_to_bboxes 함수에서 box 좌표를 0~1사이 값으로 scailing하기 떄문)
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)  # 52 x 52 스케일

        for idx in range(batch_size):  # 각 배치리스트 값마다 NMS를 수행
            # nms_boxes = non_max_suppression(
            #     bboxes[idx],
            #     iou_threshold=iou_threshold,
            #     threshold=threshold,
            #     box_format=box_format,
            # )  # return list
            nms_boxes = my_non_max_suppression(bboxes[idx], iou_threshold=0.3, threshold=config.CONF_THRESHOLD,
                                           score_threshold=0.3, box_format='midpoint', method='linear')

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)  # 각 nms_box list에 0번째에 batch index 추가해서 append

            for box in true_bboxes[idx]:
                if box[1] > threshold:  # score가 threshold보다 큰것. 즉 정답의 정보만 담는다는 의미
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    model.train()

    return all_pred_boxes, all_true_boxes  # [[batch_idx, class_idx, o_p, x, y, w, h],[]...]



def check_class_accuracy(model, loss_fn, loader, scaled_anchors, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    loop = tqdm(loader, leave=True)
    losses = []
    for idx, (x, y) in enumerate(loop):
        if idx == 100:
            break
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.no_grad():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(validation_loss=mean_loss)



        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold  # 0.6
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()
    return mean_loss





def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")

    for param_group in optimizer.param_groups:
        learning_rate = param_group["lr"]  # 현재 optimizer의 learning rate를 저장

    checkpoint = {
        'state_dict': model.state_dict(),
        "optimizer": optimizer.state_dict(),
        'learning_rate': learning_rate
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint['learning_rate']

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders():
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        root=config.TRAIN_DIR,
        anchors=config.ANCHORS,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        transform=config.train_transforms
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,  # cpu를 이용한 데이터 로드 멀티 프로세싱 / 값이 클수록 gpu로 데이터를 빨리 던져줄 주 있지만 너무 크다면 데이터 로딩 외의 작업이 영향을 받을 수 있기때문에 적당한 값 필요
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,  # 배치로 나누고 마지막에 남는 데이터도 다 사용
    )

    test_dataset = YOLODataset(
        root=config.VAL_DIR,
        anchors=config.ANCHORS,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        transform=config.test_transforms,
        mosaic=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    return train_loader, test_loader


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




