import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor((alpha, 1-alpha))
        if isinstance(alpha, list): self.alpha = torch.Tensor((alpha))
        self.size_average = size_average

    def forward(self, input, target):
        # pdb.set_trace()
        if input.dim() > 2:  # (n, c, h, w)
            input = input.permute(0,2,3,1)  # (n, h, w, c)
            input = input.contiguous().view(-1, input.size(3))  # (n x h x w, c)
        target = target.view(-1, 1)  # (12800) -> (12800, 1)

        logpt = F.log_softmax(input, dim=1)  # log(pt) = log(softmax)
        logpt = logpt.gather(1, target)  # Y * log(pt)
        logpt = logpt.view(-1)  # (12800, 1) -> (12800)
        pt = Variable(logpt.exp())  # logpt에 다시 e를 취해서 softmax 까지 계산한 상태로 돌아감

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))  # (12800)  / target 0: alpha  / target 1(true) : 1-alpha
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt  # gamma가 0이면 일반적인 cross entropy
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
