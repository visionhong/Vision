import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super(DistillationLoss, self).__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        outputs_kd = None

        if not isinstance(outputs, torch.Tensor):#  outputs가 튜플인 경우
            outputs, outputs_kd = outputs  # training일 시에 cls_embed와 dist_embed를 return 받음
        base_loss = self.base_criterion(outputs, labels)  # student output과 true label간의 loss
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)  # 여기서 teacher model의 output을 구함

        if self.distillation_type == 'soft':
            T = self.tau  # soft distillation에서 label smoothing을 위한 하이퍼 파라미터
            distillation_loss = F.kl_div(  # Kullback-Leibler divergence
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs/ T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
