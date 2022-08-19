from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import transforms, models

import time
import torch.nn as nn

class PaintBender(nn.Module):
    def __init__(self, core, target_path, style_path, device, time_limit, content_weight=1, style_weight=1000,
                 snap_every=500):
        super().__init__()

        self.target_path = target_path
        self.style_path = style_path
        # Backbone Preporation
        self.core = core
        for param in self.core.parameters():
            param.requires_grad_(False)
        self.style_weights = {'conv1_1': 1.0,
                              'conv2_1': 1.0,
                              'conv3_1': 1.0,
                              'conv4_1': 1.0,
                              'conv5_1': 1.0}

        self.content_total_weight = content_weight
        self.style_total_weight = style_weight
        self.show_every = snap_every
        self.counter = 0
        self.device = device
        self.time_limit = time_limit
        self.prev_style_loss = float("inf")

    def forward(self, target):
        tar_img, st_img = PaintBender.load_image(self.target_path, self.style_path, self.device)
        style_features = self.get_vgg19_features(st_img)
        content_features = self.get_vgg19_features(tar_img)
        target_features = self.get_vgg19_features(target)
        style_grams = {layer: PaintBender.gram_matrix(style_features[layer]) for layer in style_features}
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        #Calculate style loss from layers features
        style_loss = 0
        for layer in self.style_weights:
            target_feature = target_features[layer]
            target_gram = PaintBender.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
        #Compute total loss
        total_loss = self.content_total_weight * content_loss + self.style_total_weight * style_loss
        curr_time = (time.time() - self.start_time) / 60.0
        #Save snapshot
        if (self.counter % self.snap_every == 0 or curr_time > self.time_limit):
            print('Total loss: ', total_loss.item())
            print("Content Loss: ", content_loss.item())
            print("Style Loss: ", style_loss.item())

            print("Time(minites): ", curr_time)
            plt.imshow(PaintBender.im_convert(target))
            plt.show()
            # Time Limit Check and Loss Descending Check
            if (self.prev_style_loss < style_loss.item() or curr_time > self.time_limit):
                print("=============================")
                print("Time to Break!")
                print("=============================")
                return False
            self.prev_style_loss = style_loss.item()
        self.counter += 1
        return total_loss

    def create_target(self):
        '''Create trainable tensor from target image'''
        tar_img, st_img = PaintBender.load_image(self.target_path, self.style_path, self.device)
        tar_img.requires_grad_(True).to(self.device)
        return tar_img

    def load_image(target_path, style_path, device, max_size=250, shape=None):
        ''' Load and transform an image'''
        target_image = Image.open(target_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')
        if max(target_image.size) > max_size:
            size = max_size
        else:
            size = max(target_image.size)
        target_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        target_image = target_transform(target_image).unsqueeze(0).to(device)

        style_transform = transforms.Compose([
            transforms.Resize([target_image.shape[2], target_image.shape[3]]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        style_image = style_transform(style_image).unsqueeze(0).to(device)

        return target_image, style_image

    def im_convert(tensor):
        """ Remove normalization from tensor and tranform to image [0...1]"""

        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    def get_vgg19_features(self, image):
        '''Extract features dictionary from layers of vgg19'''
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

        features = {}
        for name, layer in self.core._modules.items():
            image = layer(image)
            if name in layers:
                features[layers[name]] = image

        return features

    def gram_matrix(tensor):
        """ Calculate the Gram Matrix of a given tensor """

        # get the batch_size, depth, height, and width of the Tensor
        _, d, h, w = tensor.size()

        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(d, h * w)

        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())

        return gram