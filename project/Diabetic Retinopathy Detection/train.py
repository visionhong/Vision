# 데이터가 너무 커서 캐글 노트북에서 진행해야할듯

import torch
from torchvision.models import MobileNetV2
from torch import nn, optim
import os
import config
import pdb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
    get_csv_for_blend,
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for idx, (data, targets, _) in enumerate(loop):
        pdb.set_trace()
        data, targets = data.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(data)  # (batch,1)
            loss = loss_fn(outputs, targets.unsqueeze(1).float())  # target (batch) -> (batch, 1)

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")

def main():
    path = 'E:\\Computer Vision\\data\\Diabetic Retinopathy'

    train_ds = DRDataset(
        images_folder=os.path.join(path, 'train'),
        path_to_csv=os.path.join(path, 'train.csv'),
        transform=config.train_transforms,
    )
    val_ds = DRDataset(
        images_folder=os.path.join(path, 'train'),
        path_to_csv=os.path.join(path, 'val.csv'),
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder=os.path.join(path, 'train'),
        path_to_csv=os.path.join(path, 'test.csv'),
        transform=config.val_transforms,
        train=False  # inference
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=6,
        shuffle=False
    )
    loss_fn = nn.MSELoss()

    # model = EfficientNet.from_pretrained('efficientnet-b3')
    model = MobileNetV2(num_classes=1)
    # model._fc = nn.Linear(1536, 1)

    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)


    for epoch in range(config.NUM_EPOCHS):
        check_accuracy(model, val_loader)

        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        # get on train
        # preds, labels = check_accuracy(train_loader, model, config.DEVICE)
        # print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"b3_{epoch}.pth.tar")

if __name__ == "__main__":
    main()



