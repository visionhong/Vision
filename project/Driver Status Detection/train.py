import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.optim as optim

from config import CFG

from model import Yolov4

from tqdm import tqdm
from util import (
    cells_to_bboxes,
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    seed_everything,
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
    init_logger
)
from loss import YOLOLoss
from iou_focal_loss import Loss
from torch.utils.tensorboard import SummaryWriter
import pdb



def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(CFG.device)

        y0, y1, y2 = (
            y[0].to(CFG.device),
            y[1].to(CFG.device),
            y[2].to(CFG.device)
        )
        if CFG.amp:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )

            # OHEM 추가해보기 ex) https://gist.github.com/erogol/c37628286f8efdb62b3cc87aad382f9e

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss


def train_loop(df, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    train_loader, valid_loader = get_loaders(df, fold)

    model = Yolov4(yolov4conv137weight='yolov4.conv.137.pth', n_classes=CFG.num_classes, inference=True)
    model.to(CFG.device)

    # # backbone weight freeze
    # for name, param in model.named_parameters():
    #     if name in ['layers.0.conv.weight']:
    #         break
    #     param.requires_grad = False

    optimizer = optim.Adam(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    # loss_fn = YOLOLoss()
    loss_fn = Loss()

    scaler = torch.cuda.amp.GradScaler()  # FP16
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )
    writer = SummaryWriter(
        f"runs/driver"
    )

    if CFG.load_model:
        print("Model Loading!")
        load_checkpoint(
            'checkpoint.pth.tar', model, optimizer
        )

    scaled_anchors = (
            torch.tensor(CFG.anchors) * torch.tensor(CFG.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(CFG.device)

    best_map = 0
    train_step = 0
    val_step = 0
    best_loss = np.inf
    for epoch in range(CFG.epochs):
        start_time = time.time()
        print(f"Epoch:{epoch + 1}")
        train_mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        scheduler.step(train_mean_loss)
        writer.add_scalar("Training loss", train_mean_loss, global_step=train_step)
        train_step += 1

        valid_mean_loss = check_class_accuracy(valid_loader, model, loss_fn, scaled_anchors, CFG.conf_threshold, CFG.device)
        writer.add_scalar("validation loss", valid_mean_loss, global_step=val_step)
        val_step += 1
        elapsed = time.time() - start_time  # 1 epoch동안 걸린 시간 (train + validation)

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {train_mean_loss:.4f}  avg_val_loss: {valid_mean_loss:.4f}  time: {elapsed:.0f}s')

        if valid_mean_loss < best_loss:
            best_loss = valid_mean_loss
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict()},
                       os.path.join(CFG.output_dir, f'fold{fold}_best_loss.pth'))

        if (epoch+1) % 3 == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                valid_loader,
                model,
                iou_threshold=CFG.NMS_IOU_THRESH,
                anchors=CFG.anchors,
                threshold=CFG.conf_threshold,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=CFG.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=CFG.num_classes,
            )

            if mapval > best_map:
                best_score = mapval
                LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict()},
                           os.path.join(CFG.output_dir, f'fold{fold}_best_mAP.pth'))

            print(f"MAP: {mapval.item()}")

def main(df):
    start = time.time()
    oof_df = pd.DataFrame()

    for fold in range(CFG.n_fold):
        train_loop(df, fold)

    end = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    # parser.add_argument('--img-size', type=int, default=416, help='[train, test] image sizes')
    # parser.add_argument('--num-classes', type=int, default=11, help='number of classes')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--pretrained-weight', type=str, default='darknet53_pretrained.pth.tar', help='pretrained weights file name')
    parser.add_argument('--backbone', type=str, default='darknet53', help='backbone network')
    parser.add_argument('--server', action='store_true', help='use gpu server')  # train시에 선언만 하면 True
    opt = parser.parse_args()

    seed_everything(42)
    if opt.server:  # colab
        CFG.path = '/content/sample_driving/'

    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    LOGGER = init_logger()

    with open(os.path.join(CFG.path, 'sample_labels.json')) as json_file:
        json_data = json.load(json_file)
    df = pd.DataFrame(json_data['annotation'])

    fold = KFold(CFG.n_fold, shuffle=True, random_state=42)
    for n, (train_idx, val_idx) in enumerate(fold.split(df, df['file_name'])):
        df.loc[val_idx, 'fold'] = int(n)


    main(df)





