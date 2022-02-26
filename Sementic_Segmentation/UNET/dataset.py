import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)  # gray scale
        mask[mask == 255.0] = 1.0  # mask는 애초에 픽셀값이 0아니면 1이기 때문에 255만 1로 scaling

        if self.transform is not None:
            augmentations = self.transform(image= image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

print(np.array(Image.open('E:\\Computer Vision\\data\\Carvana\\train_masks\\0cdf5b5d0ce1_01_mask.gif').convert('L')).shape)