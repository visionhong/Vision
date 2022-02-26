from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_modules(module_defs):
    ''' Constructs module list of layer blocks from module configuration in module_defs'''
    hyperparams = module_defs.pop(0)  # net type정보들
    output_filters = [int(hyperparams['channels'])]  # 3
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):  # 딕셔너리(layer묶음) 하나씩 가져오기
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2  # int(module_def['pad'])
            modules.add_module(
                f'conv_{module_i}',  # 표시용
                nn.Conv2d(in_channels=output_filters[-1],
                          out_channels=filters,
                          kernel_size=kernel_size,
                          stride=int(module_def['stride']),
                          padding=pad,
                          bias=not bn  # bn이 0이면 True, 1이상이면 True
                ),
            )
            if bn:
                # eps : BN 식에서 안정성을 위해 분모에 추가된 값
                modules.add_module(f'batch_norm_{module_i}',
                                   nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_{module_i}',
                                   nn.LeakyReLU(0.1))

        # maxpool은 tiny model에 있음
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            # kernel_size 2에 stride1이면 padding할때 사이즈가 1 줄어들기때문에 우하단에다가 zero-padding 1씩 먼저 줌
            if kernel_size == 2 and stride == 1:
                modules.add_module(f'_debug_padding_{module_i}',
                                   nn.ZeroPad2d((0,1,0,1)))
            # stride가 1이면 사이즈가 그대로 stride가 2면 사이즈 절반
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size -1) // 2))
            modules.add_module(f'maxpool_{module_i}',maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module(f'upsample_{module_i}', upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])  # 1개 혹은 2개의 각 지점의 filter수 합(inputchannel 3 제외)
            modules.add_module(f'route_{module_i}', EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module(f'shortcut_{module_i}', EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            #pdb.set_trace()
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(module_def['classes'])
            img_size = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            #pdb.set_trace()
            modules.add_module(f'yolo_{module_i}', yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Upsample(nn.Module):
    ''' nn.Upsample is deprecated'''

    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)  # nearest 보간법
        return x


class EmptyLayer(nn.Module):
    ''' Placeholder for 'route' and 'shortcut' layers'''

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    ''' Detection layer'''

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)  # 3
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        '''
        :param grid_size:현재 feature map의 grid_size  13 or 26 or 52
        '''
        self.grid_size = grid_size
        g = self.grid_size  # 13 or 26 or 52
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 32 or 16 or 8
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # 0행 0,1,2,3~ 1행 0,1,2,3~
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  # 0행 0,0,0,0~ 1행 1,1,1,1~
        #pdb.set_trace()

        # 현재 grid_cell 스케일에 맞도록 anchor 박스 크기 조정
        # 만약 grid_size가 13이라면 (116, 90), (156, 198), (373, 326) 에서
        # ([ 3.6250,  2.8125],[ 4.8750,  6.1875],[11.6562, 10.1875]) tensor로 scaling
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  # (1,3,1,1)
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))  # (1,3,1,1)

    def forward(self, x, targets=None, img_dim=None):
        '''
        :param x: (batch, 255, 13, 13)  or (batch, 255, 26, 26)  or (batch, 255, 52, 52)
        :param targets: (현재 batch의 박스 수, 6(index, label, x, y, w, h)) # 여기서 x,y,w,h,는 0~1사이값이고 xy도 아직 전체 이미지에 대한 좌표임
        :param img_dim: 416
        :return: (batch, 3*13*13, 85) or (batch, 3*26*26, 85) or (batch, 3*52*52, 85)
        '''

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.img_dim = img_dim  # 416
        num_samples = x.size(0)  # batch
        grid_size = x.size(2)  # 13 or 26 or 52

        # (batch, 3, 13, 13, 85) or (batch, 3, 26, 26, 85) or (batch, 3, 52, 52, 85
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0,1,3,4,2)
                .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x: (batch, 3, 13, 13) or (batch, 3, 26, 26) or (batch, 3, 52, 52)
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # width
        h = prediction[..., 3]  # height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # cls pred.


        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)   # (batch, 3, grid_x, grid_y, 4)
        pred_boxes[..., 0] = x.data + self.grid_x  # self.grid_x : sigmoid 취한 예측 x값 + Cx(각 grid cell 좌상단 x좌표)
        pred_boxes[..., 1] = y.data + self.grid_y  # self.grid_x : sigmoid 취한 예측 y값 + Cy(각 grid cell 좌상단 y좌표)
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # exp취한 예측 w값(batch, 3, 13, 13) x 스케일링된 anchor box w size (1,3,1,1)
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h  # exp취한 예측 h값(batch, 3, 13, 13) x 스케일링된 anchor box h size (1,3,1,1)

        # output: ([batch, 507, 85]) or ([batch, 2028, 85]) or ([batch, 8112, 85])
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # featuremap 기준값을 416사이즈에 대한 값으로 바꿔줌
                pred_conf.view(num_samples, -1, 1),  # sigmoid 취한 object confidence
                pred_cls.view(num_samples, -1, self.num_classes)  # sigmoid 취한 classes 예측값 80개
            ),
            -1,
        )
        if targets is None:
            return output, 0
        else:
            # loss를 구하려면 정답이 있어야 하므로 정답 정보들을 build_targets 에서 가져옴

            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,  # (batch, 3, 13, 13, 4)
                pred_cls=pred_cls,  # (batch, 3, 13, 13, 80)
                target=targets,  # (num_boxes, 6)
                anchors=self.scaled_anchors,  # (3, 2)
                ignore_thres=self.ignore_thres,  # 0.5
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf, loss)
            # localization loss
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # sigmoid를 취하기 전의 날것의 x(즉 pred tx)와 target tx와의 loss를 계산할것!
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # confidence loss
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # pred_conf: 0~1 / tconf: 1
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  # pred_conf: 0~1 / tconf: 0
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # classification loss
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            # total
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()  # classification 정확도
            conf_obj = pred_conf[obj_mask].mean()  # 실제 정답이 있는곳의 예측평균 confidence (높아야 좋음)
            conf_noobj = pred_conf[noobj_mask].mean()  # 실제 정답이 없는곳+없을 확률이높은 곳의 예측평균 confidence  (낮아야 좋음)
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()  # 예측 box와 정답 box간의 iou가 0.5가 넘어간것은 1로
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask  # iou가 0.5넘어간것중에 클래스까지 맞춘것만 1
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    ''' YOLOv3 object detection model'''

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        #pdb.set_trace()
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                '''
                route의 layers가 하나라면 그 위치로 이동(예측을하는 feature_map에서 다시 위로 복귀할때 사용), 
                두개라면 두개의 위치에 있는 레이어를 concatenate한다(up).
                '''
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]  # skip connection (width, height, channel 모두 같아야 연산가능)
            elif module_def['type'] == 'yolo':
                #
                pdb.set_trace()
                x, layer_loss = module[0](x, targets, img_dim)  # YOLOLayer 클래스 forward 로 이동
                # x : (1, 507, 85) or (1, 2028, 85) or (1, 8112, 85)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)

        # len(layer_outputs) : 107

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)


    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
            #pdb.set_trace()

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

if __name__ == '__main__':
    from torchsummary import summary
    a = Darknet('config/yolov3.cfg')
    data = torch.randn((2,3,416,416))
    a(data)
    # summary(a, input_size=(3,416,416))

