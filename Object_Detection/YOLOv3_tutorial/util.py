from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)  # 416 // 13 = 32
    grid_size = inp_dim // stride  # 13
    bbox_attrs = 5 + num_classes  # 85
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # (batch, 255, 169)
    prediction = prediction.transpose(1,2).contiguous()  # (batch, 169, 255)
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)  # (batch, 507, 85)  13*13 feature map에서는 507개의 bbox가 나옴
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the center_X(tx), center_Y(ty), and object confidence
    # 모든 예측 박스의 tx, ty, cbjectness를 sigmoid해준다.
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid) # a: 0,1,2,3,4 ...  b: 000.... , 111..., 222...

    # 각그리드셀의 좌상단 좌표
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()

    # 13x13 그리드셀의 3개의 anchorbox에대한 x,y좌표
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)  # (1, 507, 2)
    prediction[:,:,:2] += x_y_offset  # bx, by를 구함

    anchors = torch.FloatTensor(anchors)  # list -> tensor
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)  # (1, 507, 2)  Pw, Ph를 구함
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors  # bw,bh를 구함

    prediction[:,:,5: 5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    prediction[:,:,:4] *= stride  # 416x416으로 키움

    return prediction  # (batch, 507, 85) or (batch, 26*26*3, 85) or ()

def unique(tensor): # (survived. 1) 1은 class confidence중 max의 인덱스
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)  # class index값들의 unique값만 가져옴 (1차원 리스트로)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape) # unique값 개수만큼의 텐서
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    '''
    :return: The IOU of two bounding boxes
    '''

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the coordinates of the intersectuin rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    # 두 박스가 전혀 겹쳐있지 않으면 -값이 나오므로 0으로 만들어 IOU를 0으로 하기위해 torch.clamp 사용
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1+1, min = 0) * torch.clamp(inter_rect_y2 - inter_rect_y1+1, min = 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou







# objectness가 confidence보가 작은것 + nms를 수행해서 미달되는 박스의 정보는 모두 0으로 만드는 함수
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) # 0혹은 1로 채워진 mask(unsqueeze로 차원을 prediction과 같이 맞춤)
    prediction = prediction * conf_mask # objectness가 confidence보다 작으면 전부 0

    box_corner = prediction.new(prediction.shape)  # (batch, 10647, 85) 크기 하나 만듦
    # cx,cy,w,h -> x1,y1,x2,y2
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2] / 2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3] / 2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2] / 2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3] / 2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # confidence thresholding
        # NMS

        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)  # (10647, 1)
        max_conf_score = max_conf_score.float().unsqueeze(1)  # (10647, 1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)  # ((10647,5), (10647,1), (10647,1))
        image_pred = torch.cat(seq, 1)  # (10647, 7)

        non_zero_ind = (torch.nonzero(image_pred[:,4]))  # objectness가 confidence보다 높았던 값  # (non_zero_ind, )
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)  # (survived, 7)
        except:
            continue
        # 현재 이미지중에 살아남은 bbox가 없으면 다음 이미지
        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # unique인덱스값 리턴 tensor([0,50,43,...])

        # 클래스별로 NMS 실행
        for cls in img_classes:

            # get the detection
            cls_mask = image_pred_ * (image_pred_[:,-1] == cls).float().unsqueeze(1) # 현재 클래스와 masking된 값 (mask,7)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()  # max값이 0일 수도 있으므로 0이 아닌 것들의 인덱스
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)  # image_pred_에서 현재 class의 값만 가져옴

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            # class의 max condence값 (:,5)가 아니라 objectness를 기준으로 내림차순 sort함
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1] # [1] 이니까 sort 인덱스가져옴
            image_pred_class = image_pred_class[conf_sort_index] # sort
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    # max와 나머지끼리 비교 / 차원 맞추기 위해 max에 0번째인덱스에 차원추가
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:  # 마지막에 i+1 에서 값이 비어있는 tensor를 가져오는것에 대한 예외
                    break
                except IndexError:  # 아래에서 삭제되는 값들때문에 index가 줄어들기 때문에 발생
                    break

                # Zero  out all the detecions that have IOU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask  # iou가 nms_conf보다 작은애들만 살리고 나머지는 0

                # Remove
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze() # 2차원 -> 1차원
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind) # (survived, 1) 가 ind로 채워짐
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1) # (D x 8)
                '''
                 1. batch에서의 이미지 인덱스
                 2~5. 4개의 꼭지점 좌표
                 6. objectness 점수
                 7. maximum confidence를 가진 class의 점수
                 8. 그 class의 index
                '''
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0

# 이미지 비율을 유지하고 남은 영역을 (128,128,128)로 채운 상태로 이미지를 resize
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

# bgr -> RGB + transpose + to tensor + scaling
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img




def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]  # 맨 뒤에 공백('')이 하나 있어서 제외
    return names











