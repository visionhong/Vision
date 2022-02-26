from __future__ import division

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import *

CUDA = torch.cuda.is_available()

def get_test_input():
    img = cv2.imread('cat4.jpg')
    img = cv2.resize(img, (416,416))  # Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:] / 255.0  # Add a channel at 0 (for batch) | Normalize
    img_ = torch.from_numpy(img_).float()  # Convert to float tensor
    img_ = Variable(img_)  # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    '''
    :param cfgfile: Configurarion 파일을 인자로 받음
    :return: Block list(딕셔너리들의 리스트)를 리턴. 각 block들은 neural network를 어떻게 빌드하는지에 대한것
    추가로 모든 값들은 string으로 되어있음
    '''

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# parse_cfg 함수에서 리턴받은 blocks 리스트를 인자로 받는다.
def create_modules(blocks):
    net_info  = blocks[0]  # 네트워크의 하이퍼 파라미터들
    module_list = nn.ModuleList()
    prev_filters = 3  # 처음 RGB가 filter의 개수
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # logic
        # 블록의 타입을 확인
        # 블록에 대해 새로운 module을 만든다
        # module list에 추가한다

        # 아래는 conv와 batch_norm, activation
        if (x['type'] == 'convolutional'):
            # Get the info about the layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # convolutional layer 추가
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Batch Norm layer 추가
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # activation 체크
            # YOLO는 Linear 혹은 Leaky ReLU를 사용
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        # 아래는 upsample 일때
        elif (x['type'] == 'upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')  # nearest는 원래 픽셀값 그대로 늘어난 픽셀값에 부여
            module.add_module('upsample_{}'.format(index), upsample)

        # route layer 일때
        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            # Strat of a route
            start = int(x['layers'][0])
            # end, if there exists one.
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

            # shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            # [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]  # [(10, 13), (16, 30), (33, 23)

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


# Testing the code
# blocks = parse_cfg('cfg/yolov3.cfg')
# a,b = create_modules(blocks)
# print(b)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)


    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for thr route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                #pdb.set_trace()

                # Get the input dimensions
                inp_dim = int(self.net_info['height'])

                # Get the number of classes
                num_classes = int(module['classes'])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)  # (batch, 10647, 85)

            outputs[i] = x

        return detections


#
# model = Darknet("cfg/yolov3.cfg")
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print(pred.shape)  # 1, 10647, 85

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            # weights 파일을 iterate하면서 conv에 대해서만 weights를 네트워크의 module에 load를 한다.

            module_type = self.blocks[i+1]['type']

            # If module_type is convolutional load weights
            # Otherwise ignore

            if module_type == 'convolutional':
                model = self.module_list[i]
                '''
                Batch norm layer가 convolutional 블록에 나타날 때는 bias가 없다. 
                하지만, batch norm layer가 없다면 bias "weights"가 파일에서 읽혀와야 된다.
                '''
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0] # convolution layer

                if (batch_normalize):
                    bn = model[1] # batchnorm layer   /  model[2] : leacky relu


                    # Get the numver of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else: # batchnorm이 없다면 conv의 bias만 load
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # convolutional layer의 weight를 load
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

if __name__ == '__main__':
    model = Darknet("cfg/yolov3.cfg")
    model(torch.randn(1,3,416,416), torch.cuda)
    #model.load_weights("yolov3.weights")

















