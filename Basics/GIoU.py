# Generalized Intersection over Union

import torch

def bbox_overlaps_giou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
    Returns:
        gious(ndarray): shape (n, k)
    """


    #bboxes1 = torch.FloatTensor(bboxes1)
    #bboxes2 = torch.FloatTensor(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    inter = torch.clamp((inter_max_xy - inter_min_xy), 0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), 0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area
    ious = (inter_area / union) - (closure - union) / closure  # GIou = Iou - (outer - union/outer)  -1~1
    ious = torch.clamp(ious, min=-1.0, max=1.0)

    if exchange:
        ious = ious.T
    return ious


if __name__ == '__main__':
    bboxes1 = torch.tensor([[0, 0, 100, 100]])
    bboxes2 = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]])
    print(bbox_overlaps_giou(bboxes1, bboxes2))

