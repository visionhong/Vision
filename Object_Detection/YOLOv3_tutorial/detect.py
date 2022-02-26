from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

'''
--images 는 이 argument에 해당하는 값을 넣기 원할 때 먼저 나와야 하는 flag이다. Ex) python detect.py --images dog.png
dest 는 argument에 접근할 수 있는 이름이다. args.images 로 해당 argument의 값에 접근할 수 있다.
help 는 단순히 해당 argument가 무엇을 의미하는지 알려주는 역할을 한다.
default 는 아무 argument를 주지 않았을 때 default의 값을 전달한다는 것을 의미한다.
type 은 전달 받을 argument의 자료형이다.
'''


def arg_parse():
    '''
    Parse arguements to the detect module
    '''

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    # 이미지 경로
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)

    # detection 결과 저장 경로
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)

    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)

    # objectness에 비교할 confidence
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()

args = arg_parse()
images = args.images # 이미지 경로
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes('data/coco.names') # class가 담긴 list



# Set up the neural network
print('Loading network.....')
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print('Network successfully loaded')

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])
assert inp_dim % 32 == 0
assert inp_dim > 32


# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'),images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print('No file or directory with the name {}'.format(images))
    exit()


if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    im_dim_list = im_dim_list.cuda()


leftover = 0
if (len(im_dim_list) % batch_size):
   leftover = 1

if batch_size != 1:
   num_batches = len(imlist) // batch_size + leftover
   im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                       len(im_batches))]))  for i in range(num_batches)]


write = 0

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print ("No detections were made")
    exit()

'''
Prediction인 output tensor는 네트워크의 입력 크기에 대응되지만 원본 이미지 크기에는 대응되지 않는다. 
그래서, bounding box들을 그리기 전에 다시 output을 transform(변환)해야 한다.
Bounding box의 꼭지점 좌표들을 원본 이미지에 맞게 변환을 한다.
'''
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

output[:, 1:5] /= scaling_factor

# 이미지 좌표 바깥으로 나간 박스를 잘라서 안으로 들어오게 만듦
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])  # w
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])  # h

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


# 박스를 그리는 함수
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = '{0}'.format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1) # -1은 색상을 fill해줌
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

list(map(lambda x: write(x, loaded_ims), output))


det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("\\")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()




