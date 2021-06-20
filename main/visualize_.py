import os
import argparse
import cv2
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.vgg import VGG16
from models.grad_cam import FeatureExtractor, GradCam

parser = argparse.ArgumentParser()
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
    model = VGG16()
    solver = GradCam(model, model.model, "15")
    model = nn.DataParallel(model).cuda()
    model.eval()
    checkpoint_path = os.path.join('./checkpoints/cifar10/vgg16/model_best.pth.tar')
    print("=> loading model: '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    '''
    model_dict = model.module.state_dict()
    checkpoint = checkpoint['state_dict']
    model_dict.update(chec)
    model.module.load_state_dict(model_dict)
    '''

    big_heatmap = np.zeros((320, 320, 3))
    big_img = np.zeros((320, 320, 3))

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for cls_num, cls_path in enumerate(os.listdir(args.inputdir)):
        cls_path_abs = os.path.join(args.inputdir, cls_path)
        for img_id, img_path in enumerate(os.listdir(cls_path_abs)):
            # load the image
            img_path_abs = os.path.join(cls_path_abs, img_path)
            # print(img_path_abs)
            img = cv2.imread(img_path_abs)
            
            img = np.float32(img)
            big_img[cls_num*32:(cls_num+1)*32, img_id*32:(img_id+1)*32, :] = img
            img = img/255  # 0-255 to 0-1
            # print(img.shape)
            img = img[:, :, ::-1]   #BGR to RGB
            

            # generate heatmap
            #print(x.shape)
            x = transform(img.copy())
            x = x.unsqueeze(0)
            gray_cam = solver(x)
            heatmap = solver.heatmap(img, gray_cam)

            # save the heatmap
            name = 'GradCam_' + cls_path +  '_' + img_path + '.jpg'
            heatmap = np.clip(heatmap, 0, 1)
            heatmap = np.uint8(heatmap * 255)
            big_heatmap[cls_num*32:(cls_num+1)*32, img_id*32:(img_id+1)*32, :] = heatmap

            print(heatmap.shape)

            cv2.imwrite(os.path.join(args.outputdir, cls_path, name), heatmap)
    cv2.imwrite('./heatmap.jpg', big_heatmap)
    cv2.imwrite('./img.jpg', big_img)
    
    return

if __name__ == '__main__':
    main()