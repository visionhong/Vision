import torch
from torch import nn
import torch.nn.functional as F
import pdb
import torchsummary

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BN_Conv_Mish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return Mish()(out)


class ResidualBlock(nn.Module):
    def __init__(self, channels, inner_channels=None):
        super(ResidualBlock, self).__init__()
        if inner_channels is None:
            inner_channels = channels
        self.conv1 = BN_Conv_Mish(channels, inner_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(inner_channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out) + x   # activation 전에 skip connection이 필요하므로 BN_Conv_Mish 함수를 사용하지 않음
        return Mish()(out)


# Downsample conv + Residual block  with CSP
class CSPFirst(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirst, self).__init__()
        self.dsample = BN_Conv_Mish(in_channels, out_channels, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_channels, out_channels, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_channels, out_channels, 1, 1, 0)
        self.block = ResidualBlock(out_channels, out_channels//2)
        self.trans_cat = BN_Conv_Mish(2*out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.block(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSPStem(nn.Module):
    def __init__(self, in_channels, out_channels, num_block):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_channels, out_channels, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_channels, out_channels//2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_channels, out_channels//2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels//2) for _ in range(num_block)])
        self.trans_cat = BN_Conv_Mish(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSP_DarkNet(nn.Module):
    def __init__(self, num_blocks, num_classes=1000):
        super(CSP_DarkNet, self).__init__()
        channels = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3, 32, 3, 1, 1)
        self.neck = CSPFirst(32, channels[0])
        self.body = nn.Sequential(
            *[CSPStem(channels[i], channels[i+1], num_blocks[i]) for i in range(4)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[4], num_classes)

    def forward(self, x):
        pdb.set_trace()
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def csp_darknet_53(num_classes=1000):
    return CSP_DarkNet([2, 8, 8, 4], num_classes)




if __name__ =="__main__":
    model = csp_darknet_53(num_classes=1000)
    out = model(torch.rand(2, 3, 256, 256))
    print(out.shape)
    # print(model)
    # torchsummary.summary(model, input_size=(3,256,256), device='cpu')

