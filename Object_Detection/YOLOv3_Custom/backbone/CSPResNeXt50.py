#ResNext : https://mblogthumb-phinf.pstatic.net/MjAxNzA2MDVfODYg/MDAxNDk2NjQ4MzE4Njc5.FZ5d8hWjqA0r9CSdo4QjuuWLdt0MXNQlfQx17X1H364g.Nc4rFkfM-dNMPPhmoi1RvMtPgA2WCgecEPwWyPJXWxAg.PNG.sogangori/table1.PNG?type=w2
# cardinality : 집합원의 갯수

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pdb

class BN_Conv2d_Leaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, bias=False):
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return F.leaky_relu(self.seq(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, cardinality, group_width, stride=1):
        super(ResidualBlock, self).__init__()
        self.out_channels = cardinality * group_width  # 32 * 4 = 128
        self.conv1 = BN_Conv2d_Leaky(in_channels, self.out_channels, 1, 1, 0)  # channel reduction
        self.conv2 = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 3, stride, 1, groups=cardinality)  # 그룹별 conv
        self.conv3 = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 1, 1, 0)  # point wise conv
        self.bn = nn.BatchNorm2d(self.out_channels)

        # make shortcut
        layers = []
        if in_channels != self.out_channels:
            layers.append(nn.Conv2d(in_channels, self.out_channels, 1, 1, 0))  # skip connection을 위한 채널맞춤
            layers.append(nn.BatchNorm2d(self.out_channels))
        if stride != 1:
            layers.append(nn.AvgPool2d(stride))  # skip connection을 위한 feature_size 맞춤
        self.shortcut = nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.bn(out)
        out += self.shortcut(x)
        return F.leaky_relu(out)


class Stem(nn.Module):
    def __init__(self, in_channels, num_blocks, cardinality, group_width, stride=2):
        super(Stem, self).__init__()
        self.c0 = in_channels // 2  # 64
        self.c1 = in_channels - self.c0  # 64
        self.hidden_channels = cardinality * group_width  # 128
        self.out_channels = self.hidden_channels * 2  # 256
        self.trans_part0 = nn.Sequential(
            BN_Conv2d_Leaky(self.c0, self.hidden_channels, 1, 1, 0),  # 64 -> 128
            nn.AvgPool2d(stride),  # 1
        )
        self.block = self.__make_block(num_blocks, self.c1, cardinality, group_width, stride)
        self.trans_part1 = BN_Conv2d_Leaky(self.hidden_channels, self.hidden_channels, 1, 1, 0)  # 128, 128
        self.trans = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 1, 1, 0)  # 256, 256

    def __make_block(self, num_blocks, in_channels, cardinality, group_width, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [stride,1,1,1...]  # 1, 1, 1
        channels = [in_channels] + [self.hidden_channels] * (num_blocks - 1)  # 64, 128, 128
        # num_block 만큼 반복해서 sequential에 담음
        return nn.Sequential(*[ResidualBlock(c, cardinality, group_width, s) for c, s in zip(channels, strides)])


    def forward(self, x):
        x0 = x[:, :self.c0, :, :]  # 64
        x1 = x[:, self.c0:, :, :]  # 64
        out0 = self.trans_part0(x0)  # 64 -> 128
        out1 = self.trans_part1(self.block(x1))
        before_concat = out1
        out = torch.cat((out0, out1), 1)  # concatenate (128+128=256)
        return self.trans(out), before_concat


class CSP_ResNeXt(nn.Module):
    def __init__(self, num_blocks, cadinality, group_width):
        super(CSP_ResNeXt, self).__init__()
        self.conv0 = BN_Conv2d_Leaky(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = BN_Conv2d_Leaky(64, 128, 1, 1, 0)

        self.stem0 = Stem(cadinality * group_width, num_blocks[0], cadinality, group_width, stride=1)  # 128, 3, 32, 4
        self.stem1 = Stem(cadinality * group_width * 2, num_blocks[1], cadinality, group_width*2)  # 256, 4, 32, 8
        self.stem2 = Stem(cadinality * group_width * 4, num_blocks[2], cadinality, group_width*4)  # 512, 6, 32, 16
        self.stem3 = Stem(cadinality * group_width * 8, num_blocks[3], cadinality, group_width*8)  # 1024, 3, 32, 32
        self.conv2 = BN_Conv2d_Leaky(2048, 1024, 1, 1, 0)
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(cadinality*group_width*16, num_classes)

    def forward(self, x):

        out = self.conv0(x)
        out = self.pool1(out)
        out = self.conv1(out)
        out, not_use1 = self.stem0(out)
        out, concat1 = self.stem1(out)  # concat1: feature pyramid network 에 쓰일 concatenate layer  (n, 256, 52, 52)
        out, concat2 = self.stem2(out)  # concat2: (n, 512, 26, 26)
        out, not_use2 = self.stem3(out)  # (n, 2048, 13, 13)
        out = self.conv2(out)  # backbone의 최종 채널인 1024를 만들기 위한 1x1 conv  (n, 1024, 13, 13)
        # pdb.set_trace()
        # out = self.global_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out, concat1, concat2

def csp_resnext_50_32x4d():
    return CSP_ResNeXt([3, 4, 6, 3], cadinality=32, group_width=4)


if __name__ == '__main__':
    model = csp_resnext_50_32x4d()
    out, concat1, concat2 = model(torch.rand((2, 3, 416, 416)))
    print(out.shape)
    print(concat1.shape)
    print(concat2.shape)


    print(len(model.state_dict().keys()))
    # print(model)
    # summary(model, input_size=(3, 416, 416), device='cpu')

    import timm

    pre_model = timm.create_model('cspresnext50', pretrained=True)
    pre_key = list(pre_model.state_dict().keys())
    print(len(pre_key))