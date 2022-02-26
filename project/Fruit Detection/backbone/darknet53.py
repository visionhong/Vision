import torch
import torch.nn as nn
import pdb

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)

# def Conv(in_num, out_num, kernel_size=3, stride=1, padding=1):
#     return nn.Sequential(
#         nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(out_num),
#         nn.LeakyReLU(),
#     )

class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = in_channels // 2

        self.layer1 = Conv(in_channels, reduced_channels, 1, 1, 0)
        self.layer2 = Conv(reduced_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block, device, pretrained_weight='', ):
        super(Darknet53, self).__init__()
        self.device = device

        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.residual_block1 = self._make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = Conv(64, 128, 3, 2, 1)
        self.residual_block2 = self._make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = Conv(128, 256, 3, 2, 1)
        self.residual_block3 = self._make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = Conv(256, 512, 3, 2, 1)
        self.residual_block4 = self._make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = Conv(512, 1024, 3, 2, 1)
        self.residual_block5 = self._make_layer(block, in_channels=1024, num_blocks=4)
        # self.global_avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

        if pretrained_weight:
            print('Loading Pretrained model!')
            self.load_pretrained_layers(pretrained_weight)


    def _make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        concat1 = out
        out = self.conv5(out)
        out = self.residual_block4(out)
        concat2 = out
        out = self.conv6(out)
        out = self.residual_block5(out)
        # out = self.global_avg_pool(out)
        # out = out.view(-1, 1024)
        # out = self.fc(out)

        return out, concat1, concat2

    def load_pretrained_layers(self, pretrained_weight):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        check_point = torch.load(pretrained_weight, map_location=self.device)
        pretrained_state_dict = check_point['state_dict']
        pretrained_param_names = list(check_point['state_dict'].keys())

        for i, param in enumerate(param_names):  # fc layer 전까지 weight 적용  108
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def darknet53_model(device, pretrained_weight):
    return Darknet53(DarkResidualBlock, device=device, pretrained_weight=pretrained_weight)


if __name__ == '__main__':
    import torchsummary as summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = darknet53_model(device, '../darknet53_pretrained.pth.tar')
    inputs = torch.rand((4, 3, 416, 416))
    outputs, concat1, concat2 = model(inputs)

    print(outputs.shape)
    print(concat1.shape)
    print(concat2.shape)
    print("Success!!")
    # print(model)
    # summary.summary(model, input_size=(3, 416, 416), device='cpu')
