import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import pdb

from albumentations_tutorial.utils import plot_examples

# 골든 리트리버 50장 / 스웨덴 엘크 하운드 1장
# Methods for dealing with imbalanced datasets:
# 1. Oversampling
# 2. Class weighting

# loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, 50]))


def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    # class_weights = [1, 50]  # [1/50, 1] 가능
    class_weights = []
    for root, subdir, files in os.walk(root_dir):  # 재사용성
        if len(files) > 0:
            class_weights.append(1/len(files))  # [1/50, 1]

    sample_weights = [0] * len(dataset)

    for index, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[index] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)  # shuffle must be False when use sampler
    return loader


def main():
    images = []
    loader = get_loader(root_dir="dogs", batch_size=8)

    num_retrievers = 0
    num_elkhounds = 0
    for epoch in range(10):
        for data, labels in loader:
            for i in range(len(data)):
                num_retrievers += torch.sum(labels == 0)
                num_elkhounds += torch.sum(labels == 1)

            if epoch == 5:
                images.append(np.array(data[i].permute(1, 2, 0)))


    print(f"num_retrievers: {num_retrievers}")
    print(f"num_elkhounds: {num_elkhounds}")
    plot_examples(images)


if __name__ == "__main__":
    main()
