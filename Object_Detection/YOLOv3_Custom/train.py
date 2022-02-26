import argparse
import config
import torch
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
import numpy as np
from model import YOLOv3
from tqdm import tqdm
from util import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    seed_everything,
    mixup_data,
    mixup_criterion
)
from loss import YOLOLoss
from torch.utils.tensorboard import SummaryWriter
import pdb



def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)

        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        # x, y_a, y_b, lam = mixup_data(x, y)
        # #
        # y_a0, y_a1, y_a2 = (
        #     y_a[0].to(config.DEVICE),
        #     y_a[1].to(config.DEVICE),
        #     y_a[2].to(config.DEVICE)
        # )
        # y_b0, y_b1, y_b2 = (
        #     y_b[0].to(config.DEVICE),
        #     y_b[1].to(config.DEVICE),
        #     y_b[2].to(config.DEVICE)
        # )
        # ''' img show'''
        #
        # inp = x[1].cpu().numpy().transpose((1, 2, 0))
        # mean = np.array([0.6274, 0.5588, 0.4318])
        # std = np.array([0.2747, 0.2731, 0.3063])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)
        # plt.imshow(inp)
        # plt.show()
        # pdb.set_trace()

        with torch.cuda.amp.autocast():
            out = model(x)  # [(2, 3, 13, 13, 16), (2, 3, 26, 26, 16), (2, 3, 52, 52, 16)]
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            # loss = (
            #     mixup_criterion(loss_fn, out[0], y_a0, y_b0, lam, scaled_anchors[0])  # 13x13
            #     + mixup_criterion(loss_fn, out[1], y_a1, y_b1, lam, scaled_anchors[1])  # 26x26
            #     + mixup_criterion(loss_fn, out[2], y_a2, y_b2, lam, scaled_anchors[2])  # 52x52
            # )

        # OHEM 추가해보기 ex) https://gist.github.com/erogol/c37628286f8efdb62b3cc87aad382f9e

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES, backbone=opt.backbone, pretrained_weight=opt.pretrained_weight).to(config.DEVICE)
    # # backbone weight freeze
    # for name, param in model.named_parameters():
    #     if name in ['layers.0.conv.weight']:
    #         break
    #     param.requires_grad = False


    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()  # FP16
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )
    writer = SummaryWriter(
        f"runs/fruit"
    )

    train_loader, test_loader = get_loaders()

    if config.LOAD_MODEL:
        print("Model Loading!")
        load_checkpoint(
            'checkpoint.pth.tar', model, optimizer
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(config.DEVICE)

    best_map = 0
    train_step = 0
    val_step = 0
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch:{epoch+1}")
        train_mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        scheduler.step(train_mean_loss)
        writer.add_scalar("Training loss", train_mean_loss, global_step=train_step)
        train_step += 1

        if (epoch+1) % 5 == 0:
            print("On Test loader:")
            valid_mean_loss = check_class_accuracy(model, loss_fn, test_loader, scaled_anchors, threshold=config.CONF_THRESHOLD)
            writer.add_scalar("validation loss", valid_mean_loss, global_step=val_step)
            val_step += 1


            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")

            if config.SAVE_MODEL:
                if best_map < mapval.item():
                    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
                    best_map = mapval.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    # parser.add_argument('--img-size', type=int, default=416, help='[train, test] image sizes')
    # parser.add_argument('--num-classes', type=int, default=11, help='number of classes')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    # parser.add_argument('--weight-decay', type=float, default=1e-4, help='l2 normalization')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--pretrained-weight', type=str, default='darknet53_pretrained.pth.tar', help='pretrained weights file name')
    parser.add_argument('--backbone', type=str, default='darknet53', help='backbone network')
    parser.add_argument('--load-model', action='store_true', help='load checkpoint')  # train시에 선언만 하면 True
    # parser.add_argument('--conf-threshold', type=float, default=0.6, help='')
    # parser.add_argument('--map-iou-threshold', type=float, default=0.5, help='')
    # parser.add_argument('--nms-iou-threshold', type=float, default=0.45, help=''
    opt = parser.parse_args()

    if opt.batch_size > 8:  # colab에서만
        with open('data.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        config.TRAIN_DIR = data['train']
        config.VAL_DIR = data['val']
        config.NUM_CLASSES = data['nc']
        config.CLASSES = data['names']
        config.BATCH_SIZE = opt.batch_size
        config.LEARNING_RATE = opt.lr
        config.NUM_EPOCHS = opt.epochs
        config.LOAD_MODEL = opt.load_model
        # config.IMAGE_SIZE = opt.img_size


    torch.backends.cudnn.benchmark = True  # 32batch size에서 epoch당 약 6분 차이남
    '''
    내장된 cudnn 자동 튜너를 활성화하여, 하드웨어에 맞게 사용할 최상의 알고리즘(텐서 크기나 conv 연산에 맞게?)을 찾는다.
    입력 이미지 크기가 자주 변하지 않는다면, 초기 시간이 소요되지만 일반적으로 더 빠른 런타임의 효과를 볼 수 있다.
    그러나, 입력 이미지 크기가 반복될 때마다 변경된다면 런타임성능이 오히려 저하될 수 있다.
    '''

    main()





