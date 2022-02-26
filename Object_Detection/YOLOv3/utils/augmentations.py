import imgaug.augmenters as iaa
from .transforms import *

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.2)),
            iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.AddToBrightness((-30, 30)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ], random_order=True)


AUGMENTATION_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])

if __name__ == '__main__':

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2

    img_path = '../data/images/train2014/COCO_train2014_000000000025.jpg'
    label_path = '../data/labels/train2014/COCO_train2014_000000000025.txt'
    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    boxes = np.loadtxt(label_path).reshape(-1, 5)
    print(img.shape)
    print(boxes)
    img, boxes = AUGMENTATION_TRANSFORMS((img, boxes))
    print(img.shape)
    print(boxes)

    img = np.array(img.permute(1, 2, 0))


    boxes = np.array(boxes)
    print(boxes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

