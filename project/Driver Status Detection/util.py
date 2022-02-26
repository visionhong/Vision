import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import cv2
import time
from tqdm import tqdm
import random
from collections import Counter
from config import CFG, image_transform
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json


def to_cpu(tensor):
    return tensor.detach().cpu()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 대신 연산속도를 감소시킬 우려가 있음
    torch.backends.cudnn.benchmark = False


def init_logger(log_file=os.path.join(CFG.output_dir, 'train.log')):
    from logging import getLogger, INFO, FileHandler, StreamHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)  # Debug, Info, warning, error ,critical 레벨이 INFO 이상인 것만 로깅
    handler1 = StreamHandler()  # 일반 print문 같이 콘솔창에도 출력을 해주는 핸들러
    handler1.setFormatter(Formatter("%(message)s"))  # formatter는 메시지만 출력하도록
    handler2 = FileHandler(filename=log_file)  # train.log 라는 파일안에 로그를 남기는 핸들러
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)  # 메인모듈이 아닌 변수로 만든 로거에 핸들러를 추가함으로써 다른 로거가 있더라도 서로 따로 움직임
    logger.addHandler(handler2)
    return logger


def xywh_to_xyxy(bboxes):
    # bboxes = np.array(bboxes)
    bboxes_xyxy = torch.zeros_like(bboxes)

    bboxes_xyxy[:, 0:2] = bboxes[:, 0:2] - bboxes[:, 2:4] / 2
    bboxes_xyxy[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4] / 2
    return bboxes_xyxy

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
                    .unsqueeze(-1)  # (Batch, 3, S, S, 1)
                    .to(predictions.device))  # x 방향으로 0,1,2,3,4,5,6,7~

    # 중심을 가진 셀에 대해 0~1 값이었던 x,y를 현재 셀 스케일 전체에 대한 Normalize 좌표로 변환 (0~1)

    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0,1,3,2,4))  # y방향으로 바꿔주고 더함
    w_h = 1 / S * box_predictions[..., 2:4]

    converted_bboxes = torch.cat((best_class,scores,x,y,w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def non_max_suppression(bboxes, iou_threshold, threshold, score_threshold=0.001, sigma=0.5, box_format='midpoint', method='greedy'):
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]  # 먼저 confidence가 낮은 박스들을 제거
    bboxes = np.array(bboxes)
    bboxes_after_nms = []

    while bboxes.size > 0:
        max_idx = np.argmax(bboxes[:, 1], axis=0)  # confidence가 가장 높은 박스의 인덱스
        bboxes[[0, max_idx], :] = bboxes[[max_idx, 0], :]  # 위에서 구한 인덱스의 박스와 첫번째 박스의 위치를 바꿈
        bboxes_after_nms.append(bboxes[0, :].tolist())

        iou = intersection_over_union(torch.from_numpy(bboxes[1:, 2:]),
                                      torch.from_numpy(bboxes[0, 2:]),
                                      box_format=box_format)

        iou = np.array(iou.squeeze())  # tensor(n, 1) -> numpy(n)
        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_threshold] -= iou[iou > iou_threshold]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_threshold] = 0

        bboxes[1:, 1] *= weight  # 선택된 box를 제외한 나머지 박스에 마스킹
        # 선택된 박스의 클래스와 다르거나 선택된 박스와의 iou가 일정 이하면 살아남음
        bboxes = np.array([box.tolist() for box in bboxes[1:, :] if (bboxes[0, 0] != box[0]) or (box[1] >= score_threshold)])

    return bboxes_after_nms



def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, DIoU=False, CIoU=False):
    if box_format == 'midpoint':

        if len(boxes_labels.shape) == 1:
            boxes_labels = boxes_labels.unsqueeze(0)
        if len(boxes_preds.shape) == 1:
            boxes_preds = boxes_preds.unsqueeze(0)

        boxes_labels = xywh_to_xyxy(boxes_labels)
        boxes_preds = xywh_to_xyxy(boxes_preds)


    pred_x1 = boxes_preds[..., 0:1]
    pred_y1 = boxes_preds[..., 1:2]
    pred_x2 = boxes_preds[..., 2:3]
    pred_y2 = boxes_preds[..., 3:4]

    true_x1 = boxes_labels[..., 0:1]
    true_y1 = boxes_labels[..., 1:2]
    true_x2 = boxes_labels[..., 2:3]
    true_y2 = boxes_labels[..., 3:4]

    inner_x1 = torch.max(pred_x1, true_x1)
    inner_y1 = torch.max(pred_y1, true_y1)
    inner_x2 = torch.min(pred_x2, true_x2)
    inner_y2 = torch.min(pred_y2, true_y2)

    intersection = (inner_x2 - inner_x1).clamp(0) * (inner_y2 - inner_y1).clamp(0)  # 최소값은 0으로 제한
    box1_area = abs(pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    box2_area = abs(true_x2 - true_x1) * (true_y2 - true_y1)

    union = box1_area + box2_area - intersection
    iou = intersection / (union + 1e-16)

    outer_x1 = torch.min(pred_x1, true_x1)
    outer_y1 = torch.min(pred_y1, true_y1)
    outer_x2 = torch.max(pred_x2, true_x2)
    outer_y2 = torch.max(pred_y2, true_y2)
    outer_area = abs(outer_x2 - outer_x1) * (outer_y2 - outer_y1)

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU  두 박스가 겹쳐있지 않더라도 학습이 가능해짐
            giou = iou - abs(outer_area - union) / abs(outer_area)
            return torch.clamp(giou, min=-1.0, max=1.0)

        if DIoU or CIoU:  # Distance or Complete IoU
            c2 = (outer_x2 - outer_x1)**2 + (outer_y2 - outer_y1)**2 + 1e-16  # 피타고라스
            center_x1 = (pred_x1 + pred_x2) / 2
            center_y1 = (pred_y1 + pred_y2) / 2
            center_x2 = (true_x1 + true_x2) / 2
            center_y2 = (true_y1 + true_y2) / 2
            rho2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2  # 유클리디안 거리
            if DIoU:  # 두박스의 중심을 통해 거리를 최소화 하는 방향을 제공 -> GIoU 보다 빠르게 수렴
                diou = iou - rho2 / c2
                return torch.clamp(diou, min=-1.0, max=1.0)
            if CIoU:  # DIoU에서 aspect ratio까지 고려한 방법
                # v : aspect ration의 일치성을 측정
                pred_w = pred_x2 - pred_x1
                pred_h = pred_y2 - pred_y2
                true_w = true_x2 - true_x1
                true_h = true_y2 - true_y1

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(true_w / true_h) - torch.atan(pred_w / pred_h), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                ciou = iou - (rho2 / c2 + v * alpha)
                return torch.clamp(ciou, min=-1.0, max=1.0)

    return iou


def plot_image(image, boxes):
    cmap = plt.get_cmap('tab20b')
    class_labels = CFG.classes
    colors = [cmap(i) for i in np.linspace(0,1,len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape
    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2  # x,y만 좌상단 좌표로
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
            s=class_labels[int(class_pred)],  # text
            color="white",
            verticalalignment="bottom",
            bbox={"color": colors[int(class_pred)], "pad": 0},  # text를 감싸는 박스 생성
        )

    plt.show()


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
    checkpoint = torch.load(checkpoint_file, map_location=CFG.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint['learning_rate']

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def get_loaders(df, fold):
    from Dataset import DriverDataset
    train_idx = df[df['fold'] != fold].index
    val_idx = df[df['fold'] == fold].index

    train_df = df.loc[train_idx].reset_index(drop=True)
    valid_df = df.loc[val_idx].reset_index(drop=True)


    train_dataset = DriverDataset(
        path=os.path.join(CFG.path, 'sample_images'),
        df=train_df,
        classes=CFG.class_dict,
        S=CFG.S,
        transform=image_transform(data='train')
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,  # cpu를 이용한 데이터 로드 멀티 프로세싱 / 값이 클수록 gpu로 데이터를 빨리 던져줄 주 있지만 너무 크다면 데이터 로딩 외의 작업이 영향을 받을 수 있기때문에 적당한 값 필요
        pin_memory=True,
        shuffle=True,
        drop_last=False,  # 배치로 나누고 마지막에 남는 데이터도 다 사용
    )

    valid_dataset = DriverDataset(
        path=os.path.join(CFG.path, 'sample_images'),
        df=valid_df,
        classes=CFG.class_dict,
        S=CFG.S,
        transform=image_transform(data='valid')
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        # cpu를 이용한 데이터 로드 멀티 프로세싱 / 값이 클수록 gpu로 데이터를 빨리 던져줄 주 있지만 너무 크다면 데이터 로딩 외의 작업이 영향을 받을 수 있기때문에 적당한 값 필요
        pin_memory=True,
        shuffle=False,
        drop_last=False,  # 배치로 나누고 마지막에 남는 데이터도 다 사용
    )

    return train_loader, valid_loader



def check_class_accuracy(loader, model, loss_fn, scaled_anchors, threshold, device):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    loop = tqdm(loader, leave=True)
    losses = []
    for idx, (x, y) in enumerate(loop):
        if idx == 100:
            break
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device)
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
            y[i] = y[i].to(device)
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
        for i in range(3):  # 3개의 스케일 prediction을 모두 모은 리스트
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S  # 앞에*을 붙여서 리스트를 벗고 들어감 -> tenor변환 하면서 차원하나 축소효과
            # pdb.set_trace()
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )  # 해당 스케일의 예측값을 t에서 b로 변환하고 (batch, SxSx3, 6)으로 차원변경
            for idx, (box) in enumerate(boxes_scale_i):  # 배치만큼
                bboxes[idx] += box  # ex) 0번째 리스트에 0번째 이미지에 대한 예측 정보를 담음

        # 정답에 대한 변환값은 아무 스케일 1개에서 가져오면 됨 (cells_to_bboxes 함수에서 box 좌표를 0~1사이 값으로 scailing하기 떄문)
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):  # 각 배치리스트 값마다 NMS를 수행
            # nms_boxes = non_max_suppression(
            #     bboxes[idx],
            #     iou_threshold=iou_threshold,
            #     threshold=threshold,
            #     box_format=box_format,
            # )  # return list
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=CFG.conf_threshold,
                                           score_threshold=0.001, box_format='midpoint', method='greedy')
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)  # 각 nms_box list에 0번째에 batch index 추가해서 append

            for box in true_bboxes[idx]:
                if box[1] > threshold:  # score가 threshold보다 큰것. 즉 정답의 정보만 담는다는 의미
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    model.train()

    return all_pred_boxes, all_true_boxes


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
        TP = torch.zeros((len(detections)))   # True Positive (물체를 정답이라고 예측한 것)
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
        print(f"{CFG.classes[c]} AP: {ap:.2f}")

    return sum(average_precisions) / len(average_precisions)








def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True