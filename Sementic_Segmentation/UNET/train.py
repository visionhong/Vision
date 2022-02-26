import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import pdb


from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyper parameters etc.
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
num_epochs = 3
num_workers = 2
image_height = 160  # 1280 originally
image_width = 240  # 1918 originally
pin_memory = True
load_model = False
train_img_dir = 'E:\\Computer Vision\\data\\Carvana\\train_images'
train_mask_dir = 'E:\\Computer Vision\\data\\Carvana\\train_masks'
val_img_dir = 'E:\\Computer Vision\\data\\Carvana\\val_images'
val_mask_dir = 'E:\\Computer Vision\\data\\Carvana\\val_masks'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        '''
        data shape : (8, 3, 160, 240)
        targets shape : (8, 160, 240)  # gray scale이라서 채널이 없음을 인지
        '''
        data = data.to(device)

        targets = targets.float().unsqueeze(1).to(device)  # add channel dimension
        #pdb.set_trace()
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())




def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    train_loader, val_loader = get_loaders(train_img_dir,train_mask_dir,val_img_dir,val_mask_dir,
                                           batch_size, train_transform, val_transform, num_workers, pin_memory)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    check_accuracy(val_loader, model, device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=device)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder='saved_images', device=device)


if __name__ == '__main__':
    main()







