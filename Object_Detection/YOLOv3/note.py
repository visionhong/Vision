import torch
import torch.nn as nn

# a = torch.tensor([0,1,5])
# b = torch.tensor([[10],[11],[12],[13],[14],[15],[17]])
#
# print(b)
# print(b[~a])
# a = 1
# if not a:
#     print('ho')



# m = nn.ZeroPad2d((0,1,0,1))
# a = nn.MaxPool2d(kernel_size=2, stride=1)
# input = torch.randn(1, 1, 3, 3)
# print(m(input))
# print(a(m(input)))

#
# print(torch.arange(13).repeat(13,1).view([1,1,13,13]))
# print(torch.arange(13).repeat(13,1).t().view([1,1,13,13]))

# anchors = [(10,13),(16,30),(33,23)]
# stride = 32
#
# print(torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors]))
#

# a = torch.tensor([1,2,3])
# print(type(a.data))

# prediction = torch.randn((16,3,13,13,85))
# print(prediction.t().shape)

import cv2
# import matplotlib.pyplot as plt
# a = cv2.imread('data/images/train2014/COCO_train2014_000000500130.jpg')
# a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
# plt.imshow(a)
# plt.show()


import torch
# a = torch.randint((9,4,3))
#
# print(a[...,0])
# print(a)
# a = [{'type':'net'},{}]
# a[-1]['type'] = 'conv'
# print(a)

# print(torch.cuda.get_device_name(0))

print(torch.arange(13).repeat(13, 1).view([1, 1, 13, 13]))
print()
print(torch.arange(13).repeat(13, 1).t().view([1, 1, 13, 13]))





















