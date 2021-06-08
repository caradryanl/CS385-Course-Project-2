import os
import argparse
import cv2
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.resnet import ResNet18
from models.vgg16 import VGG16
from models.alexnet import AlexNet
from models.grad_cam import FeatureExtractor, GradCam

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='vgg', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--checkpoint', default='./checkpoints/', type=str)
parser.add_argument('--gpu-id', default='1', type=str)
parser.add_argument('--inputdir', default='./grad_cam/input/', type=str)
parser.add_argument('--outputdir', default='./grad_cam/output/', type=str)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main():    
    # Raise Model
    if args.arch == 'resnet':
        model = ResNet18()
        solver = GradCam(model, model.layer2, "3")
    elif args.arch == 'alexnet':
        model = AlexNet()
        solver = GradCam(model, model.layer2, "3")
    elif args.arch == 'vgg':
        model = VGG16()
        solver = GradCam(model, model.model, "18")
    else:
        raise NotImplementedError("Arch {} is not implemented.".format(args.arch))
    model = nn.DataParallel(model).cuda()
    model.eval()
    checkpoint_path = os.path.join(args.checkpoint, args.dataset+'/', args.arch+'/model_best.pth.tar')
    print("=> loading model: '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    '''
    model_dict = model.module.state_dict()
    checkpoint = checkpoint['state_dict']
    model_dict.update(chec)
    model.module.load_state_dict(model_dict)
    '''

    # Dataset
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4912393, 0.4820985, 0.44652376], std=[0.24508634, 0.24272567, 0.26051667])
        ])
    elif args.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2888097,], std=[0.3549146,])
        ])
    else:
        raise NotImplementedError("Dataset {} is not implemented.".format(args.dataset))

    for img_path in os.listdir(args.inputdir):
        # load the image
        img_path_abs = os.path.join(args.inputdir, img_path)
        img = cv2.imread(img_path_abs)
        img = np.float32(img) / 255 # 0-255 to 0-1
        img = img[:, :, ::-1]   #BGR to RGB
        # img = img.transpose((2, 0, 1)) # [32, 32, 3] to [3, 32, 32]

        # generate heatmap
        #print(x.shape)
        x = transform(img.copy())
        x = x.unsqueeze(0)
        gray_cam = solver(x)
        heatmap = solver.heatmap(img, gray_cam)

        # save the heatmap
        name = 'GradCam_' + args.arch + '_' + args.dataset + '_' + img_path + '.jpg'
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = np.uint8(heatmap * 255)

        print(heatmap.shape)

        cv2.imwrite(os.path.join(args.outputdir, name), heatmap)
    
    return

if __name__ == '__main__':
    main()