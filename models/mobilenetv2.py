import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Bottleneck(nn.Module):
    """expand -> depthwise conv -> pointwise conv"""
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU6(True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if self.stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x) + out if self.stride == 1 else out
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # self.drop1 = nn.Dropout2d(0.5, inplace=True)
        self.relu = nn.ReLU6(True)
        self.layer1 = self._make_layer(32, 16, 1, 1, 1)
        # self.layer1 = Bottleneck(32, 16, 1, 1)
        self.layer2 = self._make_layer(16, 24, 2, 1) # chang stride 2 to 1 for CIFAR-10
        self.layer3 = self._make_layer(24, 32, 3, 2)
        self.layer4 = self._make_layer(32, 64, 4, 2)
        self.layer5 = self._make_layer(64, 96, 3, 1)
        self.layer6 = self._make_layer(96, 160, 3, 2)
        self.layer7 = self._make_layer(160, 320, 1, 1)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.drop2 = nn.Dropout2d(0.5, inplace=True)
        self.linear = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, in_channels, out_channels, repeat, stride, expansion=6):
        layers = []
        for i in range(1, repeat+1):
            layers.append(Bottleneck(in_channels, out_channels, expansion=expansion, stride=stride))
            in_channels, stride = out_channels, 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.relu(self.drop2(self.bn2(self.conv2(out))))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def name(self):
        return 'MobileNetV2'

def test():
    net = MobileNetV2()
    x = Variable(torch.randn(1, 3, 32, 32))
    y = net(x)
    print(y.size())

# test()