import torch
import torch.nn as nn
import torch.nn.functional as F 

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)

'''
    To adjust to the size of F-MNIST and CIFAR10, we modified the structure of AlexNet
    referring to https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py
    
'''

class AlexNet(nn.Module):

    def __init__(self, cls_num=10):
        super(AlexNet, self).__init__()
        layers = []
        layers.append(conv3x3(3, 64, 2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2))
        
        layers.append(conv3x3(64, 192, 1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(conv3x3(192, 384, 1))
        layers.append(nn.ReLU(inplace=True))

        layers.append(conv3x3(384, 256, 1))
        layers.append(nn.ReLU(inplace=True))

        layers.append(conv3x3(256, 256, 1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2))

        self.model = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, cls_num),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
