import numpy as np
import torch
from dataset import YOLODataset
import config
import pdb
from util import generalized_intersection_over_union
import torch.nn as nn
import math
import torch
import numpy as np
import os
label1 = np.array([[0, 1, 0,0,150,150], [0,1,0,0,100,100]])
print(np.full((label1.shape[0], 1), 0.4))
print(np.hstack((label1, np.full((label1.shape[0], 1), 0.4))))