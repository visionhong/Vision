import torch
import torch.nn as nn

import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from albumentations_tutorial.utils import plot_examples
import os



class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):  # root_dir = cat_dogs folder
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(self.root_dir, name))
            self.data += zip(files, [index]*len(files))
        print("self.data", self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']

        return image, label


transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ]
)

dataset = ImageFolder('cat_dogs', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

image_list = []

for (image, label) in dataloader:
    image = image.squeeze().permute(1,2,0)  # plot_examples 함수를 쓰기위해 batch_size 없애고 차원 변환
    image_list.append(image)


print(len(image_list))
plot_examples(image_list)















