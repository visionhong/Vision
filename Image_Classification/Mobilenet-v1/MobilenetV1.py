import torch.nn as nn
from torchsummary import summary


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()
        self.ch_in = ch_in
        self.n_classes = n_classes

        def conv_st(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(  # 224x224x3
            conv_st(self.ch_in, 32, 2),  # 112x112x32
            conv_dw(32, 64, 1),  # 112x112x64
            conv_dw(64, 128, 2),  # 56x56x128
            conv_dw(128, 128, 1),  # 56x56x128
            conv_dw(128, 256, 2),  # 28x28x256
            conv_dw(256, 256, 1),  # 28x28x256
            conv_dw(256, 512, 2),  # 14x14x512
            conv_dw(512, 512, 1),  # 14x14x512
            conv_dw(512, 512, 1),  # 14x14x512
            conv_dw(512, 512, 1),  # 14x14x512
            conv_dw(512, 512, 1),  # 14x14x512
            conv_dw(512, 512, 1),  # 14x14x512
            conv_dw(512, 1024, 2),  # 7x7x1024
            conv_dw(1024, 1024, 1),  # 7x7x1024
            nn.AdaptiveAvgPool2d(1)  # 1x1x1024
        )
        self.fc = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # model check
    model = MobileNetV1(ch_in=3, n_classes=1000)
    summary(model, input_size=(3, 224, 224), device='cpu')