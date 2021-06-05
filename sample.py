import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import os
import sys
from mpl_toolkits.mplot3d import Axes3D  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert( 0,  '/slstore/liangchumeng/DPR/')
from models.imagenet.resnet import ResNetDCT_Upscaled_Static
import datasets.cvtransforms as transforms
import torchvision.transforms as nntransforms
from datasets import train_y_mean, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets import train_dct_subset_mean, train_dct_subset_std
from datasets import train_upscaled_static_mean, train_upscaled_static_std
from datasets import train_upscaled_static_dct_direct_mean, train_upscaled_static_dct_direct_std
from datasets import train_upscaled_static_dct_direct_mean_interp, train_upscaled_static_dct_direct_std_interp
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # for each layer, fetch the gradient
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                # x has been replaced by grad_x
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            print(name)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                if (len(x.shape) == 4):
                    if  x.shape[2] ==1 & x.shape[3] ==1:
                        print('reshaping')
                        x = x.squeeze()
                print(x.shape)
                x = module(x)

        return target_activations, x

def preprocess_image(args,img):
    if args.model == 'resnet50':
        normalize = nntransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        preprocessing = nntransforms.Compose([
            nntransforms.ToTensor(),
            normalize,
        ])
        return preprocessing(img.copy()).unsqueeze(0)
    elif args.model == 'dct':
        input_size1 = 512
        input_size2 = 448
        preprocessing =    transforms.Compose([
            transforms.Resize(input_size1),
            transforms.CenterCrop(input_size2),
            transforms.Upscale(upscale_factor=2),
            transforms.TransformUpscaledDCT(),
            transforms.ToTensorDCT(),
            #transforms.SubsetDCT(channels=args.subset, pattern=args.pattern),
            transforms.SubsetDCT(channels=192),
            transforms.Aggregate(),
            transforms.NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=192#,
                #pattern=args.pattern
            )
        ])
        return preprocessing(img.copy())[0].unsqueeze(0).cuda()
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            if isinstance(output,tuple):
                output = output[0]
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/slstore/liangchumeng/datasets/New_ImagenetSubset_100/train/n01440764/n01440764_10026.JPEG',                      help='Input image path')
    parser.add_argument('--adv-path', type=str, default='/slstore/liangchumeng/datasets/fgsm_New_ImagenetSubset_100/train/n01440764/n01440764_10026.JPEG',                      help='Input image path')
    parser.add_argument('--mpath', type=str, default='all_no_adv_dct.pth.tar',                      help='Input image path')
    parser.add_argument('-M', '--model', type = str, default = 'resnet50', help = 'Model name')
    parser.add_argument('--fc', type = int, default = 100, help = 'number of fc')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for ResNet50 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                target_layer_names=["2"], use_cuda=args.use_cuda)



    img = cv2.imread(args.image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(args, img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)
    print('####alert',grayscale_cam.shape)
    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    name2 = 'gb' + args.model + '.jpg'
    name3 = 'cam_gb' + args.model + '.jpg'
    cv2.imwrite(name2, gb)
    cv2.imwrite(name3, cam_gb)
    name1 = "cam" + args.model + '.jpg'

    cv2.imwrite(name1, cam)
    
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                    target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.adv_path, 1)
    print(img.shape)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(args,img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    name2 = 'adv_' + name2
    name3 = 'adv_' + name3
    cv2.imwrite(name2, gb)
    cv2.imwrite(name3, cam_gb)
    name1 = 'adv_' + name1

    cv2.imwrite(name1, cam)
