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
        layers.append(conv3x3(3, 24, 2))
        layers.append(nn.MaxPool2d(kernel_size=2))

        # [N, 64, 16, 16]
        
        layers.append(conv3x3(24, 96, 1))
        layers.append(nn.MaxPool2d(kernel_size=2))

        # [N, 192, 8, 8]

        layers.append(conv3x3(96, 192, 1))

        # [N, 384, 4, 4]

        layers.append(conv3x3(192, 192, 1))

        # [N, 256, 4, 4]

        layers.append(conv3x3(192, 96, 1))
        layers.append(nn.MaxPool2d(kernel_size=2))


        # [N, 96, 2, 2]

        self.model = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(384, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Linear(1024, cls_num),
        )

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
