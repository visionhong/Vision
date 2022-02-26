import torch
import torch.nn as nn
import pdb
from util import intersection_over_union

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # BCEWithLogitsLoss = BCE + sigmoid
        # pdb.set_trace()
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # binary loss
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 물체와 배경간의 class imbalance 문제해결
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.focal_loss = FocalLoss(self.bce, gamma=2, alpha=0.25)

    def forward(self, predictions, target, anchors):
        '''
        Args:
            predictions: tensor (batch, 3, S, S, 12)  12: 5(conf,x,y,w,h) + 7(predict_label)
            target: tensor (batch, 3, S, S, 6)  6: conf,x,y,w,h,true_label
            anchors: tensor (3, 2) 각 그리드셀에 맞는 scaled_anchors

        Returns:
            loss
        '''
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # ======================= #
        #       OBJECT LOSS       #
        # ======================= #
        anchors = anchors.reshape(1, 3, 1, 1, 2)  # w와 h를 가진 3개의 anchor가 모든 셀에서 계산하기위해 broad casting을 사용
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # sigmoid(tx) + Cx 가 아닌 sigmoid(tx)만 있는 이유는 차원이 (N, 3, 13, 13, 17) 에서 이미 13x13으로 나누어져 있기 때문
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()  # 실제 박스와 예측이 얼마나 겹쳤는가
        # detach: gradient가 전파되지 않는 텐서생성

        real_target = target.clone()
        real_target[..., 0:1][obj] = ious * target[..., 0:1][obj]

        object_loss = self.focal_loss(predictions[..., 0:1], real_target[..., 0:1])


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        predictions[..., 3:5] = torch.exp(predictions[..., 3:5]) * anchors

        ciou = intersection_over_union(predictions[..., 1:5][obj],
                                       target[..., 1:5][obj],
                                       box_format='midpoint',
                                       CIoU=True)
        box_loss = (1.0 - ciou).mean()


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.ce(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

        return object_loss + box_loss + class_loss




if __name__ == '__main__':
    from util import seed_everything
    from config import CFG
    seed_everything(42)
    # prediction = torch.rand(2, 3, 19, 19, 12)
    # target = torch.rand(2, 3, 19, 19, 6)
    # prediction = prediction[..., 0:1]
    # target = target[..., 0:1]
    # prediction = prediction.view(-1, 1)
    # target =  target.view(-1, 1)
    # print(prediction.shape)
    # # print(prediction[..., 0:1].shape)
    # fl = FocalLoss(nn.BCEWithLogitsLoss())
    # loss = fl(prediction, target)
    # print(loss)

    scaled_anchors = (
            torch.tensor(CFG.anchors) * torch.tensor(CFG.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(CFG.device)
    print(scaled_anchors.shape)