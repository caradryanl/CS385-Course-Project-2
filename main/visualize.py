import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.resnet import ResNet18
from models.vgg16 import VGG16
from models.alexnet import AlexNet
from models.grad_cam import FeatureExtractor, GradCam

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--checkpoint', default='./checkpoints/', type=str)
parser.add_argument('--gpu-id', default='1', type=str)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main():
    # Raise Model
    if args.arch == 'resnet':
        model = ResNet18()
    elif args.arch == 'alexnet':
        model = AlexNet()
    elif args.arch == 'vgg':
        model = VGG16()
    else:
        raise NotImplementedError("Arch {} is not implemented.".format(args.arch))
    checkpoint_path = os.path.join(args.checkpoint, args.dataset+'/', args.arch+'/model_best.pth.tar')
    print("=> loading model: '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4912393, 0.4820985, 0.44652376], std=[0.24508634, 0.24272567, 0.26051667])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4912393, 0.4820985, 0.44652376], std=[0.24508634, 0.24272567, 0.26051667])
        ])
    elif args.dataset == 'fmnist':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2888097,], std=[0.3549146,])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2888097,], std=[0.3549146,])
        ])
    else:
        raise NotImplementedError("Dataset {} is not implemented.".format(args.dataset))

    
    
    return

if __name__ == '__main__':
    main()