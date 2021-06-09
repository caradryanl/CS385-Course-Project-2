import torch
import torch.nn as nn
import torch.nn.functional as F 

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# raw implementation of resnet18
class ResNet18_raw(nn.Module):

    def __init__(self, cls=10):
        super(ResNet18_raw, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer_basic(64, 64, 1)
        self.layer2 = self._make_layer_basic(64, 128, 2)
        self.layer3 = self._make_layer_basic(128, 256, 2)
        self.layer4 = self._make_layer_basic(256, 512, 2)
        self.downsample1 = self._make_layer_downsample(64, 64, 2)
        self.downsample2 = self._make_layer_downsample(64, 128, 2)
        self.downsample3 = self._make_layer_downsample(128, 256, 2)
        self.downsample4 = self._make_layer_downsample(256, 512, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, cls)
        
    # make layer for basic blocks
    def _make_layer_basic(self, in_features, out_feature, depth=2):
        layers = []

        # Add the first layer group: conv/2 with bn and relu
        layers.append(conv3x3(in_features, out_feature, stride=2))
        layers.append(nn.BatchNorm2d(out_feature))
        layers.append(nn.ReLU(inplace=True))

        for i in range(depth - 1):
            layers.append(conv3x3(in_features, out_feature, stride=1))
            layers.append(nn.BatchNorm2d(out_feature))
            if i != depth - 1:
                layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def _make_layer_downsample(self, in_feature, out_feature, stride=2):
        downsample = nn.Sequential(
            conv1x1(in_feature, out_feature, stride),
            nn.BatchNorm2d(out_feature),
        )
        return downsample

    # forward
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = F.relu(self.layer1(x) + self.downsample1(x))
        x = F.relu(self.layer2(x) + self.downsample2(x))
        x = F.relu(self.layer3(x) + self.downsample3(x))
        x = F.relu(self.layer4(x) + self.downsample4(x))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        res = self.fc(x)

        return res

# Basic Block 
class BasicBlock(nn.Module):

    def __init__(self, in_feature, out_feature, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_feature, out_feature, stride)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_feature, out_feature)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.downsample = downsample
        self.stride = stride
        if in_feature != out_feature or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_feature, out_feature, stride),
                nn.BatchNorm2d(out_feature),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_feature, out_feature, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        width = int(out_feature)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_feature, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_feature)
        self.bn3 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if in_feature != out_feature or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_feature, out_feature, stride),
                nn.BatchNorm2d(out_feature),
            )
        self.stride = stride

    def forward(self, x):
        i = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            i = self.downsample(i)

        out += i
        out = self.relu(out)

        return out

# Standard implementation of ResNet18
class ResNet18(nn.Module):
    def __init__(self, cls_num=10, input_size=32):
        super(ResNet18, self).__init__()
        if input_size%32 != 0:
            raise ValueError("The input_size must match the format of K*32.")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = BasicBlock(64, 64, 1)
        self.layer2 = BasicBlock(64, 128, 2)
        self.layer3 = BasicBlock(128, 256, 2)
        self.layer4 = BasicBlock(256, 512, 2)
        self.avgpool = nn.AvgPool2d(input_size//32)
        self.fc = nn.Linear(512, cls_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        res = self.fc(x)

        return res

class ResNet50(nn.Module):
    def __init__(self, cls_num=10, input_size=32):
        super(ResNet50, self).__init__()
        if input_size%32 != 0:
            raise ValueError("The input_size must match the format of K*32.")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 3, stride=2)
        self.layer2 = self._make_layer(Bottleneck, 64, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 128, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, 512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(input_size//32)
        self.fc = nn.Linear(512, cls_num)

    def _make_layer(self, block, in_features, out_features, blocks, stride=1):
        layers = []
        layers.append(block(in_features, out_features, stride, None))
        for _ in range(1, blocks):
            layers.append(block(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        res = self.fc(x)

        return res

