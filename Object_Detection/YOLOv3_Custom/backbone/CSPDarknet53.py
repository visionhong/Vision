# CSPDarknet53 Architecture : https://www.programmersought.com/images/85/4c9f529e5b6e35f6bb8056f9119f867d.png


import torch
from torch import nn
import torch.nn.functional as F
import pdb
import torchsummary

# Mish: https://eehoeskrap.tistory.com/440
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


# Downsample conv + Residual block with CSP
class CSPFirst(nn.Module):
    def __init__(self, in_channels, out_channels):  # 32, 64
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
    def __init__(self, num_blocks, down_pretrained_weight=True):
        super(CSP_DarkNet, self).__init__()
        channels = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3, 32, 3, 1, 1)
        self.neck = CSPFirst(32, channels[0])
        self.body1 = CSPStem(channels[0], channels[1], num_blocks[0])
        self.body2 = CSPStem(channels[1], channels[2], num_blocks[1])
        self.body3 = CSPStem(channels[2], channels[3], num_blocks[2])
        self.body4 = CSPStem(channels[3], channels[4], num_blocks[3])

        self._initialize_weights()
        if down_pretrained_weight:
            print('Loading CSP_DarkNet Pretrained model!')
            self.load_pretrained_layers()

    def forward(self, x):
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body1(out)
        out = self.body2(out)
        concat1 = out
        out = self.body3(out)
        concat2 = out
        out = self.body4(out)

        return out, concat1, concat2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        import timm
        pretrained_state_dict = timm.create_model('cspdarknet53', pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:40]):  # body~fc layer 전까지 weight 적용  108
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i+42]]
            print()
            print(param ,'------', pretrained_param_names[i+42])
            print(state_dict[param].shape, '-----',pretrained_state_dict[pretrained_param_names[i+42]].shape)

       # body1.trans_0.conv.weight
        self.load_state_dict(state_dict)




def csp_darknet_53( down_pretrained_weight=True):
    return CSP_DarkNet([2, 8, 8, 4], down_pretrained_weight)



if __name__ =="__main__":
    model = csp_darknet_53(down_pretrained_weight=True)
    out,concat1, concat2 = model(torch.rand(2, 3, 416, 416))
    print(out.shape)
    print(concat1.shape)
    print(concat2.shape)
    # key = list(model.state_dict().keys())
    # print(key)

    import timm

    pre_model = timm.create_model('cspdarknet53', pretrained=True)
    torchsummary.summary(pre_model, input_size=(3, 416, 416), device='cpu')
    # pre_key = list(pre_model.state_dict().keys())
    # print(len(pre_key))
    # # print(pre_model)
    # print(pre_model.CrossStage)
    # print(model)
    # torchsummary.summary(pre_model, input_size=(3, 416, 416), device='cpu')

    #
    # print(model.state_dict()['conv0.conv.weight'])
    # print(pre_model.state_dict()['stem.conv1.conv.weight'])

