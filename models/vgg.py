import torch
import torch.nn as nn
import torch.nn.functional as F 

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)

setting = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
    512, 512, 512, 'M', 512, 512, 512, 'M',]}


class VGG16(nn.Module):

    def __init__(self, cls_num=10):
        super(VGG16, self).__init__()

        layers = []
        layers.append(conv3x3(3, 64, 1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(64, 64, 1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(64, 128, 1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(128, 128, 1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(128, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(256, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.model = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, cls_num)
        )


    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x

class VGG19(nn.Module):

    def __init__(self, cls_num=10):
        super(VGG19, self).__init__()

        layers = []
        layers.append(conv3x3(3, 64, 1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(64, 64, 1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(64, 128, 1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(128, 128, 1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(128, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(conv3x3(256, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(512, 512, 1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.model = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, cls_num)
        )


    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x



