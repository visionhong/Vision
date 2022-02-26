import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2

def soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    '''
    기존 nms와 다른점 ?
    -> iteration마다 score값에 weight값을 곱하므로 score가 어떻게 달라질지 모르기 때문에 반복문 이전에 score순으로 정렬하는 것이 아닌
    매번 최고의 score를 찾고 저장하는 방식으로 진행된다.

    '''
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:

        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]  # 첫번째 row와 score가 가장 큰 row의 위치 전환
        retained_box.append(dets[0, :-1])  # 보존할 박스의 bbox,score값

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])  # 가장 큰 score를 가진 box와 나머지와의 iou 계산 준비
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0


        dets[1:, 4] *= weight  # iou가 클수록 기존 score보다 많이 줄어 듬
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]  # 0.001보다 클 경우 살아남음
        dets = dets[retained_idx + 1, :]  # 현재 iter에서의 top score(index 0)를 제외한 살아남은 박스만 다음 반복에 포함

    return np.vstack(retained_box)




if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.62],
                      [250, 250, 420, 420, 0.7],
                      [220, 220, 320, 330, 0.82],
                      [100, 100, 210, 210, 0.62],
                      [230, 240, 325, 330, 0.71],
                      [220, 230, 315, 340, 0.8]], dtype=np.float32)
    print(boxes)
    retained_boxes = soft_nms(boxes, method='gaussian', score_thr=0.01)
    # retained_boxes = soft_nms(boxes, method='greedy', iou_thr=0.45, score_thr=0.001)

    print(len(retained_boxes))

    image = np.zeros((500, 500, 3))
    for i in range(len(retained_boxes)):
        x1y1 = tuple(retained_boxes[i][0:2].astype('int'))
        x2y2 = tuple(retained_boxes[i][2:4].astype('int'))

        cv2.rectangle(image, pt1=x1y1, pt2=x2y2, color=(255, 0, 0), thickness=1)

    plt.imshow(image)
    plt.show()