import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
import torchvision.models as models

import copy


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    imsize = 512 if torch.cuda.is_available() else 256

    image_transforms = tt.Compose(
        [
            tt.Resize(imsize),
            tt.CenterCrop(imsize),
            tt.ToTensor()
        ]
    )
    image = image_transforms(image).unsqueeze(0)

    return image


# def im_convert(tensor):
#     image = tensor.to('cpu').clone().detach()
#     image = np.rollaxis(image.numpy().squeeze(), 0, 3)
#     image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

#     image = image.clip(0, 1)

#     return image

def get_gram_matrix(tensor):
    c, n, h, w = tensor.size()
    tensor = tensor.view(c * n, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram.div(c * n * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = get_gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = get_gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, image):
        return (image - self.mean) / self.std


class NST:
    def __init__(self, style_layers=None, content_layers=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        if style_layers is None:
            self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        else:
            self.style_layers = style_layers

        if content_layers is None:
            self.content_layers = ['conv_4']
        else:
            self.content_layers = content_layers

    def model_init(self, style_image, content_image):
        # Copy model
        cnn_copy = copy.deepcopy(self.cnn)

        normalization = Normalization(self.mean, self.std).to(self.device)
        
        content_losses = []
        style_losses = []

        # Build new model
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn_copy.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                # Replacing MaxPool to AvgPool for higher quality results
                layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) 
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)

        # Remove useless layers
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i+1)]    

        return model, style_losses, content_losses

    def transform_image(self, content_path, style_path, n_epochs, style_weight=1000000, content_weight=1):
        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

        # Loading images
        style_image = load_image(style_path).to(self.device)
        content_image = load_image(content_path).to(self.device)

        # Building model
        model, style_losses, content_losses = self.model_init(style_image, content_image)
        model.requires_grad_(False).to(self.device)

        # Building target
        target = content_image.clone().requires_grad_(True).to(self.device)

        # optimizer initialization
        optimizer = optim.LBFGS([target.requires_grad_()])

        # Main train loop
        run = [0]
        while run[0] <= n_epochs:

            def closure():
                with torch.no_grad():
                    target.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(target)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    os.system('clear')
                    print(f'Epoch: {run[0]}/{n_epochs}')
                    print('Total Loss: {:4f} (Style: {:2f} | Content: {:2f})'.format(loss.item(), style_score, content_score))
                    print()
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        with torch.no_grad():
            target.data.clamp_(0, 1)

        print('Transform complete, checkout \'results\' folder')
        
        return target


        
