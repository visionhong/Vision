import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(img, size):
    image = F.interpolate(img.unsqueeze(0), size=size, mode='nearest').squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))  # glob.glob으로 folder_path안에있는 파일에 .이 있으면 리스트로 가져옴
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx % len(self.files)]  # 인덱스가 범위 밖으로 넘어가는걸 막기위한 처리?
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1,5))

        # Apply transforms
        if self.transform is not None:
            img, _ = self.transform((img, boxes))

        return img_path, img


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()  # /images/train2014/COCO_train2014_000000000009.jpg ~~

        self.label_files = [
            path.replace('images','labels').replace('.png','.txt').replace('.jpg','.txt')
            for path in self.img_files  # /labels/train2014/COCO_train2014_000000000009.txt ~~
        ]
        print(self.img_files[0])
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32  # 320
        self.max_size = self.img_size + 3 * 32  # 512
        self.batch_count = 0
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        # image
        try:
            img_path = self.img_files[idx % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # label
        try:
            label_path = self.label_files[idx % len(self.img_files)].rstrip()

            #Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boxes = np.loadtxt(label_path).reshape(-1,5)

        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # transform
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))

            except:
                print(f"Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))


        # Selects new image size every tenth batch
        # multi input size를 32 배수로 랜덤하게 10 배치마다 적용
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        # 여기서 이제 labels 에 첫번째자리에 인덱스가 붙여짐(이전까지는 모든 이미지의 인덱스가 0)
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)




if __name__ == "__main__":

    import pdb
    a = ListDataset('../data/coco/trainvalno5k.txt', True, 416)