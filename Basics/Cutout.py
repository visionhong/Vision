import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from albumentations_tutorial.utils import *

class Cutout(object):
    """
    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image
        length (int): The length (in pixels) of each square patch
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        :param img: Tensor image of size (C, H, W)
        :return: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

if __name__ == '__main__':

    img_path = "images/000127.jpg"
    original = np.array(Image.open(img_path)) / 255.0

    imglist = [original]

    img = torch.from_numpy(original).permute(2,0,1)
    cut = Cutout(1, 150)
    for i in range(5):
        output = cut(img)
        output = np.array(output.permute(1, 2, 0))
        imglist.append(output)

    plot_examples(imglist)
