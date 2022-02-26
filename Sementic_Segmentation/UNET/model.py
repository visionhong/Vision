import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

class DoubleConv(nn.Module):
    '''
    Unet은 채널이 바뀔때마다 항상 conv연산 2번을 함
    '''
    def __init__(self, in_chaannels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chaannels, out_channels, 3, 1, 1, bias=False),  # Batch Normalization 을 사용할때는 보통 bias를 False를 사용
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features = [64,128,256,512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature)) # skip connection에 의해서 또 더해질 테니까 in_channels 에 2를 곱해줌

        self.bottlenect = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottlenect(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # max pool을 했을때 홀수값이 남으면 floor처리 즉 날려버리기 때문에(ex) 101 maxpool->  50)
                x = TF.resize(x, size=skip_connection.shape[2:])  # x의 w,h을 resize

            concat_skip = torch.cat((skip_connection, x), 1)  # batch, channel, width, height
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



def test():
    model = UNET(in_channels=3, out_channels=3)
    summary(model, input_size=(3, 161, 161), device='cpu')
    x = torch.randn((3, 3, 161, 161))
    preds = model(x)
    print(x.shape)
    print(preds.shape)

    assert preds.shape == x.shape


if __name__ == '__main__':
    test()
