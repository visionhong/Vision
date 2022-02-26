import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import torch
import random
from itertools import permutations
import math
from PIL import Image
from skimage.measure import compare_ssim

# 중복검사
root = 'C:\\Users\\hong\\Desktop\\레몬'
id_list = os.listdir('C:\\Users\\hong\\Desktop\\레몬')
image_list = []
for i in range(len(id_list)):
    try:
        img = cv2.imread(os.path.join(root, id_list[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_list.append(img)
    except:
        image_list.append(np.array(Image.open(os.path.join(root, id_list[i]))))



print(len(image_list))
for i in range(len(id_list)):
    for j in range(i+1, len(id_list)):
        if image_list[i].shape == image_list[j].shape:
            (score, diff) = compare_ssim(image_list[i], image_list[j], full=True, multichannel=True)

            if score == 1:
                print(f"i:{id_list[i]} j:{id_list[j]}")






#
# score, diff = compare_ssim(apple1, water, full=True, multichannel=True)
# print(score)