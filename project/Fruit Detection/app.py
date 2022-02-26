import os
import cv2
import imutils
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import config
from model import YOLOv3
from util import cells_to_bboxes, non_max_suppression, show_image, my_non_max_suppression
import pdb
import time
from flask import Flask, render_template, Response, jsonify, url_for,request, redirect
import pandas as pd
import secrets
from flask_ngrok import run_with_ngrok



datas = pd.read_excel('static/market_price.xlsx', engine='openpyxl')
app = Flask(__name__, static_url_path='/static')  # static 경로 선언
run_with_ngrok(app)

# secret Key
app.config["SECRET_KEY"] = secrets.token_hex(16)

detected_labels = []
def price(labels):
    total_price = 0
    list = labels

    for label in list:
        for i in range(datas.shape[0]):
            try:
                name = datas.loc[i, '과일']
                unit_price = datas.loc[i, '가격']

                if name == label["name"] :
                    price = (label["cnt"] * int(unit_price))
                    label["price"] = price
                    total_price += price
                    break

            except KeyError as k:
                print(k)
                break
            i += 1

    return list, total_price

@app.route('/_stuff', methods=['GET',"POST"])
def stuff():
    global detected_labels

    if request.method == "POST":
        detected_labels = []
        return jsonify(result=detected_labels)
    detected_labels,total_price = price(detected_labels)
    return jsonify(result=detected_labels, total_price= total_price)

cam = False  # 캠이 메인에 있지 않을 경우 활성화 하지 않기 위해서 False을 초기값으로 둔다

@app.route('/pay', methods = ['GET',"POST"])
def pay():
    global cam
    cam = True
    if request.method == "POST":
        result = request.form
        return render_template("item_pay.html",pay = result)

@app.route('/add', methods= ["GET", "POST"])
def add():
    global cam
    cam=True
    if request.method == "POST":
        data = request.form
        return render_template("item_add.html", datas=datas, data=data)

@app.route('/')
def index():
    return render_template('main_home.html')

@app.route('/main')
def main():
    return redirect("http://127.0.0.1:7000/")

@app.route('/video')
def video():
    global cam
    cam = False
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pay_card')
def pay_card():
    return render_template('pay_card.html')

@app.route('/pay_cash')
def pay_cash():
    return render_template('pay_cash.html')


def detect(save_img=False):
    global detected_labels
    global cam

    # pTime = 0
    if not cam:
        while True:
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image pre-processing before predict
            img = config.inference_transforms(image=img)['image']
            img = img.to(config.DEVICE)
            if img.ndimension() == 3:  # channels, width, height
                img = img.unsqueeze(0)  # 1 batch

            with torch.no_grad():
                output = model(img)

            boxes = []
            for i in range(3):
                anchor = scaled_anchors[i]  # tensor(3, 2)
                boxes += cells_to_bboxes(
                    output[i], is_preds=True, S=output[i].shape[2], anchors=anchor
                )[0]  # batch 제외 (num_anchors * S * S, 6)

            #boxes = non_max_suppression(boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD, box_format='midpoint')
            boxes = my_non_max_suppression(boxes, iou_threshold=0.3, threshold=config.CONF_THRESHOLD, score_threshold=0.3, box_format='midpoint', method='greedy')
            boxes = [box for box in boxes if box[1] > 0.3]  # nms에서 0.0인 confidence가 안사라지는 박스들이 있음
            # print(len(boxes))
            # boxes : [[class_pred, prob_score, x1, y1, x2, y2], ...]

            image = show_image(frame, boxes, colors)
            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            # cv2.putText(image, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2,lineType=cv2.LINE_AA)

            if boxes:
                boxes = np.array(boxes)
                det_cls = boxes[:, 0]
                det_cls_list = [config.CLASSES[int(i)] for i in det_cls]
                detected_labels = []
                for i in set(det_cls_list):  # 중복요소 제거
                    num = det_cls_list.count(i)
                    detected_labels.append({"name": i, "cnt": num})

            # image = imutils.resize(image, width=500)
            (flag, encodedImage) = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-file', type=str, default="", help='video path')
    opt = parser.parse_args()

    # cap = cv2.VideoCapture(opt.video_file)
    if opt.video_file:
        cap = cv2.VideoCapture(opt.video_file)
    else:
        cap = cv2.VideoCapture(0)
    torch.backends.cudnn.benchmark = True
    model = YOLOv3(num_classes=config.NUM_CLASSES, backbone='darknet53').to(config.DEVICE)
    checkpoint = torch.load('checkpoint.pth.tar', map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    colors = [[0, 0, 255], [0, 138, 255], [0, 153, 207], [0, 74, 36], [135, 254, 186], [46, 252, 255], [164, 0, 115],
              [164, 175, 46], [164, 175, 255], [2, 1, 70], [56, 232, 187]]

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # (3, 3, 2)
    scaled_anchors = scaled_anchors.to(config.DEVICE)

    app.run()