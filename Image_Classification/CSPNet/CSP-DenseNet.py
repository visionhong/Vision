import torch
from torch import nn
import torch.nn.functional as F
import pdb
import torchsummary

# DenseNet Architecture : https://miro.medium.com/max/770/1*RUiMzddMbQ0rx_CCqT2k8Q.png

class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=False, activation=True):
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, bias=bias),
                  nn.BatchNorm2d(out_channels),
                  ]
        if activation:
            layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):  # 32, 6, 32
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):  # 6
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0),  # 32, 32*4, 1, 1, 0
                BN_Conv2d(4 * self.k, self.k, 3, 1, 1)  # 32*4, 32, 3, 1, 1
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            # print("feature:", feature.shape)
            # print("out:", out.shape)
            out = torch.cat((feature, out), 1)
        return out


class CSP_DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, k, part_ratio=0.5):  # 64, 6, 32
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)  # 32
        self.part2_chnls = in_channels - self.part1_chnls  # 32
        self.dense = DenseBlock(self.part2_chnls, num_layers, k)  # 32, 6, 32

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        part2 = self.dense(part2)
        # print("part1:",part1.shape)
        # print("part2:", part2.shape)
        out = torch.cat((part1, part2), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, layers, k, theta, num_classes, part_ratio=0):
        super(DenseNet, self).__init__()
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio == 0 else CSP_DenseBlock

        self.conv = BN_Conv2d(3, 2*k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2*k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transition(self, in_chls):
        out_chls = int(self.theta * in_chls)  # 128, 256, 512
        return nn.Sequential(
            BN_Conv2d(in_chls, out_chls, 1, 1, 0),
            nn.AvgPool2d(2)
        ), out_chls

    def __make_blocks(self, k0):
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):  # 4
            # Dense Block with CSP
            layers_list.append(self.Block(k0, self.layers[i], self.k))  # (64, 128, 256, 512), (6, 12, 24, 16), 32
            patches = k0 + self.layers[i] * self.k  # 256, 512, 1024, 1024

            # transition layer (3 times)
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        out = self.conv(x)  # (batch, 64, 112, 112)
        out = F.max_pool2d(out, 3, 2, 1)  # (batch, 64, 56, 56)
        out = self.blocks(out)  # (batch, 1024, 7, 7)
        out = F.avg_pool2d(out, 7)   # (batch, 1024, 1, 1)
        out = out.view(out.size(0), -1)  # (batch, 1024)
        out = F.softmax(self.fc(out), dim=1)  # (batch, 1000)
        return out


def csp_densenet_121(num_classes=1000):
    return DenseNet(layers=[6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)



if __name__ == "__main__":
    model = csp_densenet_121()
    out = model(torch.rand((2, 3, 224, 224)))
    print(out.shape)
    # torchsummary.summary(model=model, input_size=(3, 224, 224), device='cpu')

