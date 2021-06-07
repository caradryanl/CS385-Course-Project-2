import torch
import torch.nn as nn
import numpy as np
import cv2

# Get the feature of the target layer
class FeatureExtractor(object):
    
    def __init__(self, model, target_layer):
        '''
            model: a CNN model object
            target_layer: a str, appointing the name of target_layer 
        '''
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def __call__(self, x):
        '''
            x: [N, 3, H, W]
            feature: [N, C, H_C, W_C]
            grad: [N, C, H_C, W_C]
            output: [N, K]
        '''
        feature = None
        self.gradients = None

        # Do the inference step by step, get the feature of the target layer 
        for name, module in self.model._modules.items():
            # forward for one step
            x = module(x)
            # if is the target layer, register the grad and feat
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                feature = x
        # return the feat, grad, and x
        return feature, x

    def save_gradient(self, grad):
        self.gradients = grad

    def get_gradient(self):
        return self.gradients

class GradCam(object):

    def __init__(self, model, target_layer):
        self.model = model
        self.extractor = FeatureExtractor(model, target_layer)

    def __call__(self, img):
        '''
            img: an image of size [1, 3, H, W] need to convert to numpy format
        '''
        
        img = img.cuda()
        feat, output = self.extractor(img)

        # cls_idx: [1, 1]
        self.model.zero_grad()
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][np.argmax(output, dim=1)[0, 0]] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()    
        one_hot = torch.sum(one_hot * output)
        one_hot.backward()

        # Now, the gradient has been updated
        grad = self.extractor.get_gradient().cpu().numpy()[0, :]
        grad = np.mean(grad, axis=(1, 2))
        feat = feat.cpu().numpy()[0, :]

        res = np.zeros(feat.shape[1: ], dtype=np.float32)
        for c in range(feat[0]):
            res[:, :] += grad[c, :, :] * feat[c, :, :]

        res = np.maximum(res, 0)
        res = cv2.resize(res, img.shape[2:])
        res = res - np.min(res)
        res = res / np.max(res)

        return res

    def heatmap(img, gray_cam):
        gray_cam = cv2.resize(gray_cam, (img.shape[0], img.shape[1]))
        
        # cv2 receive 0-255 scale image as input
        color_cam = cv2.applyColorMap(np.uint8(255 * gray_cam), cv2.COLORMAP_JET)
        color_cam = np.float32(color_cam) / 255
        cam = color_cam + np.float32(img)
        # back to 0-1 scalue
        cam = cam / np.max(cam)

        return cam





