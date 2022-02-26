import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import re
import pdb
import torch.nn as nn

x = (np.arange(0, 64) + 0.5) * 8
y = (np.arange(0, 64) + 0.5) * 8
shift_x, shift_y = np.meshgrid(x, y)

print(np.vstack((
    shift_x.ravel(), shift_y.ravel(),
    shift_x.ravel(), shift_y.ravel()
)).transpose())