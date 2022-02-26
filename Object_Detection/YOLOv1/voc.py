import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import random
import numpy as np
import cv2

# label_txt : image name(0), x1(1,6),y1(2,7),x2(3,8),y2(4,9),class(5,10)반복
# ex ) ['007732.jpg', '341', '217', '487', '375', '8', '114', '209', '183', '298', '8', '237', '110', '320', '176', '19']

class VOCDataset(Dataset):

    def __init__(self, is_train, image_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.is_train = is_train
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        # if isinstance(label_txt, list) or isinstance(label_txt, tuple):
        #     # cat multiple list files together
        #     # This is useful for VOC2007/2012 combination.
        #     tmp_file = '/data/label.txt'
        #     os.system('cat %s > %s' % (' '.join(label_txt), tmp_file))
        #     label_txt = tmp_file

        self.paths, self.boxes, self.labels = [], [], []

        with open(label_txt) as f:
            lines = f.readlines()

        for line in lines:
            splitted = line.strip().split()

            fname = splitted[0]
            path = os.path.join(image_dir, fname)
            self.paths.append(path)  # 이미지 경로

            num_boxes = (len(splitted) -1) //5  # 물체 수
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(splitted[i * 5 + 1])
                y1 = float(splitted[i * 5 + 2])
                x2 = float(splitted[i * 5 + 3])
                y2 = float(splitted[i * 5 + 4])
                c = int(splitted[i * 5 + 5])
                box.append([x1, y1, x2, y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        boxes = self.boxes[idx].clone()  # .clone() 텐서복사 [n, 4]
        labels = self.labels[idx].clone()  # [n,]

        if self.is_train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)

            img = self.random_blur(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)

            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # For debug.
        debug_dir = 'tmp/voc_tta'
        os.makedirs(debug_dir, exist_ok=True)
        img_show = img.copy()
        box_show = boxes.numpy().reshape(-1)
        n = len(box_show) // 4
        for b in range(n):
            pt1 = (int(box_show[4*b + 0]), int(box_show[4*b + 1]))
            pt2 = (int(box_show[4*b + 2]), int(box_show[4*b + 3]))
            cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0,255,0), thickness=1)
        cv2.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes)  # normalize (x1, y1, x2, y2)
        target = self.encode(boxes, labels) # [S,S,5 x B + C] target 이미지의 정보

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / 255.0  # normalize from -1.0 to 1.0
        imt = self.to_tensor(img)

        return img, target

    def __len__(self):
        return self.num_samples

    # 하나의 이미지의 물체에 대한 중심위치(i,j), x,y,w,h, confidence return하는 함수
    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
               Args:
                   boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
                   labels: (tensor) [c_obj1, c_obj2, ...]
               Returns:
                   An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """

        S, B, C = self.S, self.B, self.C  # 7, 2, 20
        N = 5 * B * C

        target = torch.zeros(S,S,N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2]
        boxes_xy = (boxes[:, 2:] - boxes[:, :2]) / 2.0
        for b in range(boxes.size(0)):  # bounding box 하나씩
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])  # 그리드 셀 번호(i(행),j(열))
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0

            # TBM, remove redundant dimensions from target tensor.
            # To remove there, loss implementation also has to be modified.

            for k in range(B):
                s = 5 * k  # s는 0과 5
                target[j, i, s:s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4] = 1.0  # confidence
            target[j, i, B*5 + label] = 1.0  # 물체의 클래스에 1.0

        return target

    # 50% 확률로 좌우반전
    def random_flip(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes

        h, w, _ = img.shape

        img = np.fliplr(img)  # 픽셀값 좌우 반전

        x1, x2 = boxes[:, 0], boxes[:,2]
        x1_new = w - x2
        x2_new = w - x1
        boxes[:,0], boxes[:,2] = x1_new, x2_new

        return img, boxes

    # 50% 확률로 w 방향만 random scale하는 함수
    def random_scale(self, img, boxes):
        if random.random() > 0.5:
            return img, boxes

        scale = random.uniform(0.8, 1.2)  # 이 사이의 실수
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(int(w*scale), h), interpolation = cv2.INTER_LINEAR)  # w 스케일만 변경

        scale_tensor = torch.FloatTensor([[scale, 1.0, scale, 1.0]]).expand_as(boxes)  # box개수만큼 저 텐서를 늘림
        boxes = boxes * scale_tensor

        return img, boxes


    def random_blur(self, bgr):
        if random.random() > 0.5:
            return bgr

        ksize = random.choice([2,3,4,5])  # bluring할 kernel size
        bgr = cv2.blur(bgr, (ksize, ksize))
        return bgr

    # 밝기 랜덤 조정
    def random_brightness(self, bgr):
        if random.random() > 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr


    def random_hue(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.8, 1.2)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_saturation(self, bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr


    def random_shift(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        # [x2,y2] + [x1,y1] / 2 = [cx,cy]
        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h, w, c = img.shape
        img_out = np.zeros((h, w, c), dtype=img.dtype)
        mean_bgr = self.mean[::-1]  # 역순
        img_out[:, :] = mean_bgr  # img_out 전체에 mean_bgr로 도배

        # 움직일 거리 랜덤으로 추출
        dx = random.uniform(-w*0.2, w*0.2)
        dy = random.uniform(-h*0.2, h*0.2)
        dx, dy = int(dx), int(dy)

        if dx >= 0 and dy >= 0:
            img_out[dy:, dx:] = img[:h-dy, :w-dx]  # 오른쪽 아래로 이동하고 나머지는 평균값
        elif dx >=0 and dy < 0:
            img_out[:h+dy, dx:] = img[-dy:, :w-dx]  # 오른쪽 위
        elif dx < 0 and dy >=0:
            img_out[dy:, :w+dx] = img[:h-dy, -dx:]  # 왼쪽 아래
        elif dx < 0 and dy < 0:
            img_out[:h+dy, :w+dx] = img[-dy:, -dx:]  # 왼쪽 위

        center = center + torch.FloatTensor([[dx, dy]]).expand_as(center)  # dx,dy값만큼 센터이동 # [n,2]
        # dx,dy만큼 움직인 물체의 센터가 이미지 안에있는지 밖에있는지 확인
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
        mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after shift

        boxes_out = boxes[mask.expand_as(boxes)].view(-1,4) # [m,4]  # 중심이 이미지 안에있는것만 인덱싱
        if len(boxes_out) == 0:  # 중심이 이미지 안에 있는 박스가 없다면 그냥 shift하지않고 넘어감
            return img, boxes, labels
        shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out)  # [m, 4]

        boxes_out = boxes_out + shift  # 박스를 dx,dy만큼 움직임
        # 박스를 움직였을때 박스가 이미지 밖으로 나가는 것을 처리하는 과정
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)] # shift해서 살아남은 물체들만 인덱싱

        return img_out, boxes_out, labels_out


    def random_crop(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        # [x2,y2] + [x1,y1] / 2 = [cx,cy]
        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h_orig, w_orig, _ = img.shape
        h = random.uniform(0.6 * h_orig, h_orig)
        w = random.uniform(0.6 * w_orig, w_orig)
        y = random.uniform(0, h_orig - h)  # 좌상단에만 Crop?
        x = random.uniform(0, w_orig - w)
        h, w, x, y = int(h), int(w), int(x), int(y)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)  # 센터이동
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
        mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after crop.

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4)
        # 아무 중심도 살아남지 못했으면 넘어감
        if len(boxes_out) == 0:
            return img, boxes, labels

        shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_out) # [m, 4]

        boxes_out = boxes_out - shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]
        img_out = img[y:y+h, x:x+w, :]

        return img_out, boxes_out, labels_out

