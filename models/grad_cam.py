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
            #print(name)
            #print(x.shape)
            x = module(x)
            # if is the target layer, register the grad and feat
            if name == self.target_layer:
                print(name)
                x.register_hook(self.save_gradient)
                feature = x
        # return the feat, grad, and x
        return feature, x

    def save_gradient(self, grad):
        self.gradients = grad

    def get_gradient(self):
        return self.gradients

class GradCam(object):

    def __init__(self, model, module, target_layer):
        self.model = model.cuda()
        self.module = module.cuda()
        self.extractor = FeatureExtractor(self.module, target_layer)

    def __call__(self, img):
        '''
            img: an image of size [1, 3, H, W] need to convert to numpy format
        '''
        x = img.cuda()
        for name, module in self.model._modules.items():
            #print(name)
            #print(x.shape)
            if module == self.module:
                feat, output = self.extractor(x)
                x = output
                print("This!")
            elif "avgpool" in name.lower():
                # avgpool
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                # reshape
                if (len(x.shape) == 4):
                    if  x.shape[2] ==1 & x.shape[3] ==1:
                        x = x.squeeze()
                #print(x.shape)
                x = module(x)
        
        # cls_idx: [1, 1]
        self.model.zero_grad()
        x_ = x.detach().cpu().numpy()
        one_hot = np.zeros((x_.shape[-1]), dtype=np.float32)
        #print(x_.shape)
        #print(np.argmax(x_, axis=0))
        one_hot[np.argmax(x_, axis=0)] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()    
        one_hot = torch.sum(one_hot * output)
        one_hot.backward()

        # Now, the gradient has been updated
        grad = self.extractor.get_gradient().cpu().numpy()[0, :]
        # grad = np.mean(grad, axis=(1, 2))
        # print(feat.shape)
        feat = feat.detach().cpu().numpy()[0, :]

        res = np.zeros(feat.shape[1: ], dtype=np.float32)
        for c in range(feat.shape[0]):
            res[:, :] += grad[c, :, :] * feat[c, :, :]

        res = np.maximum(res, 0)
        res = cv2.resize(res, img.shape[2:])
        res = res - np.min(res)
        res = res / np.max(res)

        return res

    def heatmap(self, img, gray_cam):
        gray_cam = cv2.resize(gray_cam, (img.shape[0], img.shape[1]))
        
        # cv2 receive 0-255 scale image as input
        color_cam = cv2.applyColorMap(np.uint8(255 * gray_cam), cv2.COLORMAP_JET)
        color_cam = np.float32(color_cam) / 255
        cam = color_cam + np.float32(img)
        # back to 0-1 scalue
        cam = cam / np.max(cam)

        return cam





