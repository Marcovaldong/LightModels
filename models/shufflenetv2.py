import torch
import torch.nn as nn
from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, splits_left=2, groups=2):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.splits_left = splits_left
        self.groups = groups

        if stride == 2:
            self.left = nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=True, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU(inplace=True)
            ])
            self.right = nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=True, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU(inplace=True)
            ])
        else:
            in_channels = in_channels - in_channels // splits_left
            self.right = nn.Sequential(*[
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.stride == 2:
            x_left, x_right = x, x
            # print '2', x_left.size()
            x_left = self.left(x_left)
            x_right = self.right(x_right)
        elif self.stride == 1:
            x_left, x_right = torch.split(x, [self.in_channels//self.splits_left,
                                              self.in_channels//self.splits_left], dim=1)
            x_right = self.right(x_right)

        x = torch.cat([x_left, x_right], dim=1)

        # channel_shuffle
        N, C, H, W = x.size()
        
        g = self.groups
        x = x.view(N, g, C / g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        return x

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, net_scale=1.0, splits_left=2):
        super(ShuffleNetV2, self).__init__()
        if net_scale == 0.5:
            self.out_channels = [24, 48, 96, 192, 1024]
        elif net_scale == 1:
            self.out_channels = [24, 116, 232, 464, 1024]
        elif net_scale == 1.5:
            self.out_channels = [24, 176, 352, 704, 1024]
        elif net_scale == 2:
            self.out_channels = [24, 244, 488, 976, 2048]
        self.splits_left = splits_left
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_planes = 24

        self.stage2 = self._make_player(stage=1, num_blocks=self.out_channels[1])
        self.stage3 = self._make_player(stage=2, num_blocks=self.out_channels[2])
        self.stage4 = self._make_player(stage=3, num_blocks=self.out_channels[3])

        self.conv2 = nn.Conv2d(self.out_channels[3], self.out_channels[4], kernel_size=1,
                               stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(self.out_channels[4], num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        x = self.max_pool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_player(self, stage, num_blocks):
        layers = []
        layers.append(Bottleneck(self.out_channels[stage-1], self.out_channels[stage],
                                 stride=2, splits_left=self.splits_left))
        for i in range(self.out_channels[stage]):
            layers.append(Bottleneck(self.out_channels[stage], self.out_channels[stage],
                                     stride=1, splits_left=self.splits_left))
        return nn.Sequential(*layers)

def test():
    net = ShuffleNetV2()
    x = Variable(torch.randn(1, 3, 224, 224))
    y = net(x)
    print y.size()

if __name__ == '__main__':
    test()
