import torch
import torch.nn as nn
from math import ceil
import pdb
from torchsummary import summary
from efficientnet_pytorch import EfficientNet as Effi

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    'b0': (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    'b1': (0.5, 240, 0.2),
    'b2': (1, 260, 0.3),
    'b3': (2, 300, 0.3),
    'b4': (3, 380, 0.4),
    'b5': (4, 456, 0.4),
    'b6': (5, 528, 0.5),
    'b7': (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,  # groups=1이면 일반적인 Conv, groups=in_channels 일때만 Depthwise Conv 수행
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),  # 각 채널에 대한 score (0~1)
        )

    def forward(self, x):
        return x * self.se(x)  # input channel x 채널의 중요도


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,  # squeeze excitation
            survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,  # Depthwise Conv
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),  # point wise conv
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        '''
        vanishing gradient로 인해 학습이 느리게 되는 문제를 완화시키고자 stochastic depth 라는 randomness에 기반한 학습 방법
        Stochastic depth란 network의 depth를 학습 단계에 random하게 줄이는 것을 의미
        복잡하고 큰 데이터 셋에서는 별다를 효과를 보지는 못한다고 함
        '''
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor  # torch.div으로 감싼 연산은 stochastic_depth 논문에 나와있음.

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        # self.pool = nn.AdaptiveAvgPool2d(1)  # stage9 pool
        # self.classifier = nn.Sequential(  # stage9 FC
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(last_channels, num_classes),
        # )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)  # B0의 32는 첫레이어의 channel
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]  # stage 1
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels*width_factor) / 4)  # SqueezeExcitation reduction에서 4로 잘 나눠지도록 처리
            # pdb.set_trace()
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):  # stage 2~8
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride if layer == 0 else 1,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels


        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)  # stage9 Conv 1x1
        )

        return nn.Sequential(*features)

    def forward(self, x):
        # x = self.pool(self.features(x))
        # return self.classifier(x.view(x.shape[0], -1))  # flatten
        return self.features(x)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    version = 'b0'
    phi, res, drop_rate = phi_values[version]
    batch_size, num_classes = 4, 10
    x = torch.randn((batch_size, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
      ).to(device)

    print(model(x).shape)
    summary(model, input_size=(3, 416, 416))
