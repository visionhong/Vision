import torch
import ttach as tta
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

image_path = "../dogs/Golden retriever/dog1.jpg"
image = np.array(Image.open(image_path)) / 255  # 이미지를 읽고 min max scaling
image = cv2.resize(image, (384, 384))
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

print(image.shape)
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90]),
        # tta.Scale(scales=[1, 2]),
        # tta.FiveCrops(384, 384),
        tta.Multiply(factors=[0.7, 1]),
    ]
)

model = timm.create_model("vit_base_patch16_384", pretrained=True)
# model = timm.create_model("seresnet50", pretrained=True)

imagenet_labels = dict(enumerate(open('classes.txt')))
fig = plt.figure(figsize=(20, 20))
columns = 3
rows = 3

for i, transformer in enumerate(transforms):  # custom transforms or e.g. tta.aliases.d4_transform()
    augmented_image = transformer.augment_image(image)

    output = model(augmented_image)
    predicted = imagenet_labels[output.argmax(1).item()].strip()

    augmented_image = np.array((augmented_image*255).squeeze()).transpose(1, 2, 0).astype(np.uint8)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(augmented_image)
    plt.title(predicted)

plt.show()
